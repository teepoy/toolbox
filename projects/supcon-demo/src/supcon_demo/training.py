from __future__ import annotations

import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from supcon_demo.losses import supervised_contrastive_loss


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type=device.type, enabled=False)


def train_supcon(
    model, train_loader, device: torch.device, config, output_dir: Path
) -> dict:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.train.lr),
        weight_decay=float(config.train.weight_decay),
    )
    scaler = torch.amp.GradScaler(
        enabled=bool(config.train.use_amp and device.type == "cuda")
    )
    model.train()

    history: list[dict] = []
    global_step = 0
    epochs = int(config.train.epochs)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            view_one, view_two, labels = batch
            view_one = view_one.to(device, non_blocking=True)
            view_two = view_two.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            images = torch.cat([view_one, view_two], dim=0)

            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, bool(config.train.use_amp)):
                _, projections = model(images)
                projections = (
                    projections.view(2, labels.shape[0], -1)
                    .permute(1, 0, 2)
                    .contiguous()
                )
                loss = supervised_contrastive_loss(
                    projections,
                    labels,
                    temperature=float(config.train.temperature),
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_value = float(loss.detach().cpu().item())
            epoch_loss += loss_value
            global_step += 1
            if step % int(config.train.log_every_steps) == 0 or step == len(
                train_loader
            ):
                print(
                    f"[train] epoch={epoch + 1}/{epochs} step={step}/{len(train_loader)} loss={loss_value:.4f}"
                )

        average_loss = epoch_loss / max(1, len(train_loader))
        history.append({"epoch": epoch + 1, "loss": average_loss})
        print(f"[train] epoch={epoch + 1} avg_loss={average_loss:.4f}")

    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": OmegaConf.to_container(config, resolve=True),
            "history": history,
            "used_pretrained_weights": bool(model.used_pretrained_weights),
        },
        checkpoint_path,
    )

    metrics = {
        "history": history,
        "checkpoint_path": str(checkpoint_path),
        "used_pretrained_weights": bool(model.used_pretrained_weights),
    }
    (output_dir / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics
