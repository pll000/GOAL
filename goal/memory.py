"""Target-only feature/prediction banks and mean teacher updates."""
import math
from numbers import Integral, Real
import torch
import torch.nn.functional as F
from .losses import entropy


class TargetMemory:
    def __init__(self, size, dim, classes, device="cpu"):
        self.features = torch.zeros(size, dim, device=device)
        self.probs = torch.zeros(size, classes, device=device)
        self.ready = torch.zeros(size, dtype=torch.bool, device=device)
        # q remains the GCSS EMA distribution. These buffers separately retain
        # the latest instantaneous prediction used to update each target item.
        self.teacher_probs = torch.zeros(size, classes, device=device)
        self.teacher_ready = torch.zeros(size, dtype=torch.bool, device=device)

    @torch.no_grad()
    def update(self, indices, features, probs, momentum=0.9):
        idx = indices.to(self.features.device)
        features = F.normalize(features.detach(), dim=-1).to(self.features.device)
        probs = probs.detach().to(self.probs.device)
        old = self.ready[idx]
        self.teacher_probs[idx] = probs
        self.teacher_ready[idx] = True
        self.features[idx] = features
        self.probs[idx] = torch.where(old[:, None], momentum * self.probs[idx] + (1-momentum) * probs, probs)
        self.ready[idx] = True

    @torch.no_grad()
    def neighbors(self, indices, features, k, include_self=True):
        if not bool(self.ready.all()):
            raise RuntimeError("memory bank must be fully initialized before search")
        idx = indices.to(self.features.device)
        similarity = F.normalize(features.detach(), dim=-1).to(self.features.device) @ self.features.T
        # Explicit index mask matters: the closest feature need not be oneself.
        similarity.scatter_(1, idx[:, None], -torch.inf)
        k = min(k, len(self.features)-1)
        nearest = similarity.topk(k, dim=-1).indices
        if include_self:
            nearest = torch.cat([idx[:, None], nearest], 1)
        return nearest, self.features[nearest], self.probs[nearest]

    @torch.no_grad()
    def reciprocal_weights(self, indices, neighbor_indices, k=5, nonreciprocal_weight=0.1,
                           return_mask=False):
        """Return 1 for reverse-top-k matches, otherwise the configured floor.

        Searches use current normalized bank features, exclude each reverse
        query's own index, and run once per unique requested neighbor. Self
        entries always receive factor 1. With return_mask=True, also return a
        boolean reciprocity mask (independent of the floor, including floor=1).
        This method never changes the bank.
        k=0 means no reverse neighbors; training uses that setting to bypass
        this method entirely. k above N-1 is capped to the available items.
        """
        if isinstance(k, bool) or not isinstance(k, Integral) or k < 0:
            raise ValueError("reciprocal k must be a nonnegative integer")
        if (isinstance(nonreciprocal_weight, bool) or not isinstance(nonreciprocal_weight, Real)
                or not math.isfinite(nonreciprocal_weight) or not 0 <= nonreciprocal_weight <= 1):
            raise ValueError("nonreciprocal_weight must be finite and in [0, 1]")
        if indices.ndim != 1 or neighbor_indices.ndim != 2 or len(indices) != len(neighbor_indices):
            raise ValueError("indices [B] and neighbor_indices [B,K] must have matching batch sizes")
        integer_types = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
        if indices.dtype not in integer_types or neighbor_indices.dtype not in integer_types:
            raise ValueError("reciprocal search indices must be integer tensors")
        if not bool(self.ready.all()):
            raise RuntimeError("memory bank must be fully initialized before reciprocal search")
        idx = indices.detach().to(device=self.features.device, dtype=torch.long)
        near = neighbor_indices.detach().to(device=self.features.device, dtype=torch.long)
        size = len(self.features)
        if bool(((idx < 0) | (idx >= size)).any()) or bool(((near < 0) | (near >= size)).any()):
            raise ValueError("reciprocal search index outside memory bank")
        is_self = near == idx[:, None]
        reciprocal = is_self.clone()
        weights = self.features.new_full(near.shape, float(nonreciprocal_weight))
        weights.masked_fill_(is_self, 1.)
        k = min(int(k), max(0, size - 1))
        if k == 0 or near.numel() == 0:
            return (weights, reciprocal) if return_mask else weights
        # Only non-self entries need reverse searches. Repeated neighbors in
        # different anchors reuse the same bank query and its top-k result.
        requested = near[~is_self]
        if requested.numel() == 0:
            return (weights, reciprocal) if return_mask else weights
        unique, inverse = torch.unique(requested, sorted=True, return_inverse=True)
        features = self.features.detach()
        similarity = features[unique] @ features.T
        similarity.scatter_(1, unique[:, None], -torch.inf)
        reverse = similarity.topk(k, dim=-1).indices
        anchors = idx[:, None].expand_as(near)[~is_self]
        mutual = (reverse[inverse] == anchors[:, None]).any(-1)
        reciprocal[~is_self] = mutual
        weights[~is_self] = torch.where(mutual, weights.new_tensor(1.),
                                        weights.new_tensor(float(nonreciprocal_weight)))
        return (weights, reciprocal) if return_mask else weights

    def entropy_threshold(self, rho_source="prediction_ema"):
        if not bool(self.ready.all()):
            raise RuntimeError("incomplete memory")
        if rho_source == "prediction_ema":
            return entropy(self.probs).mean()
        if rho_source == "teacher":
            if not bool(self.teacher_ready.all()):
                raise RuntimeError("teacher rho requires a fully initialized instantaneous teacher bank")
            return entropy(self.teacher_probs).mean()
        raise ValueError("rho_source must be 'prediction_ema' or 'teacher'")

    def state_dict(self):
        return {"features": self.features, "probs": self.probs, "ready": self.ready,
                "teacher_probs": self.teacher_probs, "teacher_ready": self.teacher_ready}

    def load_state_dict(self, state, rho_source="prediction_ema"):
        if rho_source not in ("prediction_ema", "teacher"):
            raise ValueError("rho_source must be 'prediction_ema' or 'teacher'")
        teacher_keys = [key in state for key in ("teacher_probs", "teacher_ready")]
        if any(teacher_keys) and not all(teacher_keys):
            raise ValueError("checkpoint contains an incomplete teacher bank")
        if rho_source == "teacher" and (not all(teacher_keys) or not bool(state["teacher_ready"].all())):
            raise ValueError("teacher-rho resume requires a complete saved instantaneous teacher bank; legacy EMA-only checkpoints cannot be used")
        for key in ("features", "probs", "ready"):
            getattr(self, key).copy_(state[key])
        if all(teacher_keys):
            for key in ("teacher_probs", "teacher_ready"):
                if getattr(self, key).shape != state[key].shape:
                    raise ValueError("checkpoint teacher bank shape mismatch: " + key)
                getattr(self, key).copy_(state[key])
        else:
            # An old EMA checkpoint cannot reconstruct historical instantaneous
            # outputs. Default-rho recovery remains valid without inventing them.
            self.teacher_probs.zero_()
            self.teacher_ready.zero_()


@torch.no_grad()
def update_teacher(teacher, student, momentum=0.9):
    for dest, src in zip(teacher.parameters(), student.parameters()):
        dest.mul_(momentum).add_(src, alpha=1-momentum)
    # A teacher in eval mode needs usable BN statistics; EMA floating buffers,
    # copy counters, which have no sensible fractional interpretation.
    for dest, src in zip(teacher.buffers(), student.buffers()):
        if dest.is_floating_point():
            dest.mul_(momentum).add_(src, alpha=1-momentum)
        else:
            dest.copy_(src)
