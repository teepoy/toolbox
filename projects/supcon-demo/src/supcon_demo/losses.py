from __future__ import annotations

import torch


def supervised_contrastive_loss(
    features: torch.Tensor, labels: torch.Tensor, temperature: float
) -> torch.Tensor:
    batch_size, num_views, _ = features.shape
    contrast_features = features.reshape(batch_size * num_views, -1)
    contrast_labels = labels.repeat_interleave(num_views)

    similarity = torch.matmul(contrast_features, contrast_features.T) / temperature
    logits_max, _ = similarity.max(dim=1, keepdim=True)
    logits = similarity - logits_max.detach()

    self_mask = torch.eye(
        batch_size * num_views, device=features.device, dtype=torch.bool
    )
    positive_mask = contrast_labels.unsqueeze(0) == contrast_labels.unsqueeze(1)
    positive_mask = positive_mask & ~self_mask

    exp_logits = torch.exp(logits) * (~self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    positive_count = positive_mask.sum(dim=1)
    valid_mask = positive_count > 0
    if not torch.any(valid_mask):
        return logits.new_tensor(0.0)

    mean_log_prob_pos = (positive_mask * log_prob).sum(
        dim=1
    ) / positive_count.clamp_min(1)
    loss = -mean_log_prob_pos[valid_mask].mean()
    return loss
