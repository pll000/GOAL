"""GOAL equations with explicitly selectable repairs. No target labels enter here."""
import math
import torch
import torch.nn.functional as F


def entropy(prob):
    return -(prob * prob.clamp_min(1e-8).log()).sum(-1)


@torch.no_grad()
def neighbor_weights(anchor_features, neighbor_features, neighbor_probs,
                     anchor_probs=None, top_dims=5, mode="shannon"):
    """Eq.2-4. Absolute-value ranks; overlap is intersection/top_dims."""
    dim = anchor_features.shape[-1]
    nd = min(top_dims, dim)
    if nd < 1:
        raise ValueError("top_dims must be positive")
    a = anchor_features.abs().topk(nd, dim=-1).indices
    b = neighbor_features.abs().topk(nd, dim=-1).indices
    overlap = (a[:, None, :, None] == b[:, :, None, :]).any(-1).float().mean(-1)
    if mode == "shannon":
        uncertainty = entropy(neighbor_probs)
    elif mode in ("cross_entropy", "cross_entropy_clamped"):
        if anchor_probs is None:
            raise ValueError("literal Eq.4 requires anchor_probs")
        uncertainty = -(anchor_probs[:, None] * neighbor_probs.clamp_min(1e-8).log()).sum(-1)
    else:
        raise ValueError("unknown entropy mode: " + mode)
    confidence = 1.0 - uncertainty / math.log(neighbor_probs.shape[-1])
    if mode != "cross_entropy":
        confidence = confidence.clamp(0, 1)
    return confidence * (1 + overlap)


def attraction(prob, neighbor_probs, weights=None):
    dots = (prob[:, None] * neighbor_probs.detach()).sum(-1)
    if weights is not None:
        dots = dots * weights.detach()
    return -dots.sum(-1).mean()


def neighbor_cross_entropy(logits, neighbor_probs, weights=None):
    """Optional normalized-neighbor CE; the existing attraction is unchanged.

    logits: [B, C]; neighbor_probs: [B, K, C]; nonnegative weights: [B, K].
    Neighbor probabilities and weights are detached. Each row's target is the
    weighted neighbor mean (uniform when weights is None). A zero-weight row
    contributes zero loss/gradient, but remains in the batch mean denominator.
    log_softmax preserves gradients for confidently incorrect student logits.
    """
    q = neighbor_probs.detach()
    if weights is None:
        target = q.mean(dim=1)
    else:
        w = weights.detach()
        total = w.sum(dim=-1, keepdim=True)
        # Do not clamp small positive totals: rescaling weights must leave the
        # target unchanged. Replacing an exact zero denominator avoids 0 / 0.
        denominator = torch.where(total != 0, total, torch.ones_like(total))
        normalized = w / denominator
        target = (normalized.unsqueeze(-1) * q).sum(dim=1)
        target = torch.where(total != 0, target, torch.zeros_like(target))
    return -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def dispersion(prob, reduction="sum"):
    """AaD sums off-diagonal predictions; GOAL Eq.5 instead divides by B-1."""
    if prob.shape[0] < 2:
        return prob.sum() * 0
    pair = prob @ prob.T
    offdiag_sum = pair.sum() - pair.diagonal().sum()
    loss = offdiag_sum / prob.shape[0]
    if reduction == "mean":
        loss = loss / (prob.shape[0] - 1)
    elif reduction != "sum":
        raise ValueError("dispersion reduction must be sum or mean")
    return loss


def topology_loss(features, student_probs, teacher_probs, rho,
                  feature_normalize=True, mask_mode="edges", detach_reliability=True,
                  reduction="valid_edges"):
    """Eq.6-9 over minibatch undirected edges, averaged over valid edges.

    Masking entire residual avoids pulling nonedges together. The literal option
    masks only teacher distances and averages ALL off-diagonal pairs instead.
    all_pairs changes only the denominator, keeping masked nonedge gradients zero.
    """
    if reduction not in ("valid_edges", "all_pairs"):
        raise ValueError("topology reduction must be valid_edges or all_pairs")
    q = teacher_probs.detach()
    if features.shape[0] < 2:
        return features.sum() * 0, {"edges": 0, "selected_nodes": 0}
    p = student_probs.detach() if detach_reliability else student_probs
    reliability = torch.sigmoid(-(p - q).abs().sum(-1))
    with torch.no_grad():
        confident = entropy(q) < rho
        labels = q.argmax(-1)
        valid = confident[:, None] & confident[None, :] & (labels[:, None] == labels[None, :])
        upper = torch.ones_like(valid).triu(1)
        mask = valid & upper
        teacher_distance = torch.cdist(q, q)
    z = F.normalize(features, dim=-1) if feature_normalize else features
    student_distance = torch.cdist(z, z)
    node_weights = reliability[:, None] + reliability[None, :]
    if mask_mode == "literal":
        error = (student_distance - teacher_distance * valid).abs()
        active = upper
    elif mask_mode == "edges":
        error = (student_distance - teacher_distance).abs()
        active = mask
    else:
        raise ValueError("unknown topology mask mode")
    denominator = upper.sum() if reduction == "all_pairs" else active.sum()
    loss = (node_weights * error)[active].sum() / denominator.clamp_min(1)
    return loss, {"edges": int(mask.sum()), "selected_nodes": int(confident.sum())}
