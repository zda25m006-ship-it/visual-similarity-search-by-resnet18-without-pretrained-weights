import torch
import torch.nn as nn


def pairwise_euclidean_distance(embeddings, eps=1e-12):
    """Compute pairwise Euclidean distance matrix for a batch of embeddings."""
    dot = torch.matmul(embeddings, embeddings.t())
    sq_norm = torch.diag(dot)
    sq_dist = sq_norm.unsqueeze(1) - 2.0 * dot + sq_norm.unsqueeze(0)
    sq_dist = torch.clamp(sq_dist, min=0.0)
    zero_mask = (sq_dist == 0.0).float()
    sq_dist = sq_dist + zero_mask * eps
    dist = torch.sqrt(sq_dist)
    return dist * (1.0 - zero_mask)


class TripletLoss(nn.Module):
    """Triplet margin loss with online batch-hard or random mining."""

    def __init__(self, margin=0.3, mining="hard"):
        super().__init__()
        assert mining in ("hard", "random")
        self.margin = float(margin)
        self.mining = mining

    def forward(self, embeddings, labels):
        device = embeddings.device
        b = embeddings.size(0)
        dist_mat = pairwise_euclidean_distance(embeddings)

        lbl_col = labels.unsqueeze(1)
        lbl_row = labels.unsqueeze(0)
        eye = torch.eye(b, dtype=torch.bool, device=device)

        pos_mask = (lbl_col == lbl_row) & (~eye)
        neg_mask = lbl_col != lbl_row

        valid = (pos_mask.sum(dim=1) > 0) & (neg_mask.sum(dim=1) > 0)
        if valid.sum() == 0:
            zero = (embeddings * 0.0).sum()
            return zero, {"triplet_acc": 0.0, "active_triplets": 0.0, "mean_d_ap": 0.0, "mean_d_an": 0.0}

        if self.mining == "hard":
            pos_d = dist_mat.clone()
            pos_d[~pos_mask] = -1e9
            d_ap_all, _ = pos_d.max(dim=1)

            sh_mask = neg_mask & (dist_mat > d_ap_all.unsqueeze(1))
            has_sh = sh_mask.any(dim=1)

            neg_sh = dist_mat.clone()
            neg_sh[~sh_mask] = 1e9
            d_an_sh, _ = neg_sh.min(dim=1)

            neg_all = dist_mat.clone()
            neg_all[~neg_mask] = 1e9
            d_an_hard, _ = neg_all.min(dim=1)

            d_an_all = torch.where(has_sh, d_an_sh, d_an_hard)
            d_ap = d_ap_all[valid]
            d_an = d_an_all[valid]
        else:
            ap_list, an_list = [], []
            for idx in torch.where(valid)[0]:
                p_idxs = torch.where(pos_mask[idx])[0]
                n_idxs = torch.where(neg_mask[idx])[0]
                rp = p_idxs[torch.randint(len(p_idxs), (1,), device=device)]
                rn = n_idxs[torch.randint(len(n_idxs), (1,), device=device)]
                ap_list.append(dist_mat[idx, rp])
                an_list.append(dist_mat[idx, rn])
            d_ap = torch.cat(ap_list)
            d_an = torch.cat(an_list)

        losses = torch.clamp(d_ap - d_an + self.margin, min=0.0)
        center_reg = embeddings.mean(dim=0).pow(2).sum()
        loss = losses.mean() + 0.5 * center_reg

        with torch.no_grad():
            valid_triplets = pos_mask.unsqueeze(2) & neg_mask.unsqueeze(1)
            correct_triplets = (dist_mat.unsqueeze(2) < dist_mat.unsqueeze(1)) & valid_triplets
            total_t = valid_triplets.sum().item()
            acc = float(correct_triplets.sum().item()) / float(total_t) if total_t > 0 else 0.0
            active = (losses > 1e-6).float().mean().item()

        return loss, {
            "triplet_acc": acc,
            "active_triplets": active,
            "mean_d_ap": d_ap.mean().item(),
            "mean_d_an": d_an.mean().item(),
        }


@torch.no_grad()
def evaluate_validation_triplets(model, val_loader, margin=0.3, device=torch.device("cpu")):
    """Compute validation triplet loss and percentage of triplets with d(a,p) < d(a,n)."""
    model.eval()
    criterion = TripletLoss(margin=margin, mining="hard")

    total_loss = 0.0
    total_hard_acc = 0.0
    correct_all = 0
    count_all = 0
    steps = 0

    for images, labels, _ in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        emb = model(images)

        loss, stats = criterion(emb, labels)
        total_loss += loss.item()
        total_hard_acc += stats["triplet_acc"]
        steps += 1

        dist_mat = pairwise_euclidean_distance(emb)
        b = labels.size(0)
        eye = torch.eye(b, dtype=torch.bool, device=device)
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)) & (~eye)
        neg_mask = labels.unsqueeze(1) != labels.unsqueeze(0)

        valid_triplets = pos_mask.unsqueeze(2) & neg_mask.unsqueeze(1)
        correct_triplets = (dist_mat.unsqueeze(2) < dist_mat.unsqueeze(1)) & valid_triplets
        correct_all += int(correct_triplets.sum().item())
        count_all += int(valid_triplets.sum().item())

    n = max(1, steps)
    all_acc = float(correct_all) / float(count_all) if count_all > 0 else 0.0
    return {
        "val_loss": total_loss / n,
        "val_hard_triplet_acc": total_hard_acc / n,
        "val_all_triplet_acc": all_acc,
    }


class NTXentLoss(nn.Module):
    """Normalized temperature-scaled cross-entropy contrastive loss."""

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, embeddings, labels):
        device = embeddings.device
        b = embeddings.size(0)
        sim = torch.matmul(embeddings, embeddings.t()) / self.temperature
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - sim_max.detach()

        eye = torch.eye(b, dtype=torch.bool, device=device)
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)) & (~eye)

        exp_logits = torch.exp(logits) * (~eye).float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        pos_counts = pos_mask.sum(dim=1)
        valid = pos_counts > 0
        if valid.sum() == 0:
            return (embeddings * 0.0).sum(), {"triplet_acc": 0.0, "active_triplets": 0.0}

        mean_log_pos = (pos_mask.float() * log_prob).sum(dim=1)[valid] / pos_counts[valid]
        loss = -mean_log_pos.mean()

        with torch.no_grad():
            _, stats = TripletLoss(margin=0.3, mining="hard")(embeddings, labels)
        return loss, stats


TripletMarginLossFromScratch = TripletLoss
NTXentLossFromScratch = NTXentLoss
