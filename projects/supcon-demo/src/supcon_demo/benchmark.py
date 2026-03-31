from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn


@torch.no_grad()
def extract_embeddings(
    model, loader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    embeddings = []
    labels = []
    for images, batch_labels in loader:
        images = images.to(device, non_blocking=True)
        batch_embeddings = model.encode(images)
        embeddings.append(batch_embeddings.cpu())
        labels.append(batch_labels.cpu())
    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)


def knn_metrics(
    reference_embeddings: torch.Tensor,
    reference_labels: torch.Tensor,
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    k_values: list[int],
) -> dict:
    similarities = torch.matmul(query_embeddings, reference_embeddings.T)
    metrics = {}
    max_k = max(k_values)
    num_relevant = (
        reference_labels.eq(query_labels.unsqueeze(1)).sum(dim=1).clamp(min=1)
    )

    topk_indices = similarities.topk(k=max_k, dim=1).indices
    topk_labels = reference_labels[topk_indices]

    # First relevant rank across full ranking for MRR.
    full_rank_indices = similarities.argsort(dim=1, descending=True)
    full_rank_labels = reference_labels[full_rank_indices]
    full_relevance = full_rank_labels.eq(query_labels.unsqueeze(1))
    first_relevant_rank = full_relevance.float().argmax(dim=1) + 1
    reciprocal_ranks = 1.0 / first_relevant_rank.float()
    metrics["mrr"] = float(reciprocal_ranks.mean().item())

    discount = 1.0 / torch.log2(torch.arange(2, max_k + 2, dtype=torch.float32))

    for k in k_values:
        label_matches = topk_labels[:, :k].eq(query_labels.unsqueeze(1))

        # Hit@k: probability of at least one same-label retrieval in top-k.
        correct = label_matches.any(dim=1)
        metrics[f"top_{k}_accuracy"] = float(correct.float().mean().item())

        # Average same-label ratio@k: mean(#same-label in top-k / k) across queries.
        same_label_ratio = label_matches.float().mean(dim=1)
        metrics[f"top_{k}_same_label_ratio"] = float(same_label_ratio.mean().item())

        # Precision@k: mean(#same-label in top-k / k) across queries.
        metrics[f"precision_at_{k}"] = float(same_label_ratio.mean().item())

        # Recall@k: mean(#same-label in top-k / #same-label in reference) across queries.
        recall = label_matches.float().sum(dim=1) / num_relevant.float()
        metrics[f"recall_at_{k}"] = float(recall.mean().item())

        # NDCG@k with binary relevance based on label equality.
        gains = label_matches.float() * discount[:k].to(label_matches.device)
        dcg = gains.sum(dim=1)

        ideal_counts = torch.minimum(
            num_relevant, torch.full_like(num_relevant, fill_value=k)
        )
        cumulative_discount = torch.cumsum(discount, dim=0)
        idcg = cumulative_discount[(ideal_counts - 1).long()].to(dcg.device)
        ndcg = dcg / idcg.clamp(min=1e-12)
        metrics[f"ndcg_at_{k}"] = float(ndcg.mean().item())
    return metrics


def train_linear_probe(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    test_embeddings: torch.Tensor,
    test_labels: torch.Tensor,
    probe_config,
    device: torch.device,
) -> dict:
    unique_train_labels = torch.unique(train_labels)
    unique_train_labels, _ = torch.sort(unique_train_labels)
    num_classes = int(unique_train_labels.numel())

    mapped_train_labels = torch.full_like(train_labels, fill_value=-1)
    mapped_test_labels = torch.full_like(test_labels, fill_value=-1)
    for index, label in enumerate(unique_train_labels.tolist()):
        mapped_train_labels[train_labels == label] = index
        mapped_test_labels[test_labels == label] = index

    classifier = nn.Linear(train_embeddings.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=float(probe_config.lr),
        weight_decay=float(probe_config.weight_decay),
    )
    loss_fn = nn.CrossEntropyLoss()

    train_dataset = torch.utils.data.TensorDataset(
        train_embeddings, mapped_train_labels
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(probe_config.batch_size),
        shuffle=True,
    )

    classifier.train()
    for _ in range(int(probe_config.epochs)):
        for batch_embeddings, batch_labels in train_loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(batch_embeddings)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()

    classifier.eval()
    with torch.no_grad():
        logits = classifier(test_embeddings.to(device))
        predictions = logits.argmax(dim=1).cpu()
        known_mask = mapped_test_labels >= 0
        if bool(known_mask.any()):
            accuracy = float(
                (predictions[known_mask] == mapped_test_labels[known_mask])
                .float()
                .mean()
                .item()
            )
        else:
            accuracy = 0.0
    return {"linear_probe_accuracy": accuracy}


def run_benchmark(
    model,
    reference_loader,
    eval_loader,
    config,
    device: torch.device,
    output_path: Path,
    stage_name: str,
) -> dict:
    train_embeddings, train_labels = extract_embeddings(model, reference_loader, device)
    eval_embeddings, eval_labels = extract_embeddings(model, eval_loader, device)

    metrics = {
        "stage": stage_name,
        "knn": knn_metrics(
            train_embeddings,
            train_labels,
            eval_embeddings,
            eval_labels,
            list(config.benchmark.knn_k_values),
        ),
        "linear_probe": train_linear_probe(
            train_embeddings,
            train_labels,
            eval_embeddings,
            eval_labels,
            probe_config=config.benchmark.linear_probe,
            device=device,
        ),
    }
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
