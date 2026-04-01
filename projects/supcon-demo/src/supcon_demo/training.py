from __future__ import annotations

import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from supcon_demo.losses import simclr_loss, supervised_contrastive_loss


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return torch.autocast(device_type="cuda", enabled=False)
    return torch.autocast(device_type=device.type, enabled=False)


def _train_two_view_stage(
    *,
    model,
    train_loader,
    device: torch.device,
    config,
    epochs: int,
    lr: float,
    weight_decay: float,
    temperature: float,
    use_amp: bool,
    log_every_steps: int,
    output_dir: Path,
    metrics_file_name: str,
    checkpoint_file_name: str,
    log_prefix: str,
    loss_fn,
) -> dict:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    model.train()

    history: list[dict] = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            view_one, view_two, labels = batch
            view_one = view_one.to(device, non_blocking=True)
            view_two = view_two.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            images = torch.cat([view_one, view_two], dim=0)

            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, use_amp):
                _, projections = model(images)
                projections = (
                    projections.view(2, labels.shape[0], -1)
                    .permute(1, 0, 2)
                    .contiguous()
                )
                loss = loss_fn(projections, labels, temperature)

            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().cpu().item())
            epoch_loss += loss_value
            if step % log_every_steps == 0 or step == len(train_loader):
                print(
                    f"[{log_prefix}] epoch={epoch + 1}/{epochs} step={step}/{len(train_loader)} loss={loss_value:.4f}"
                )

        average_loss = epoch_loss / max(1, len(train_loader))
        history.append({"epoch": epoch + 1, "loss": average_loss})
        print(f"[{log_prefix}] epoch={epoch + 1} avg_loss={average_loss:.4f}")

    checkpoint_path = output_dir / checkpoint_file_name
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
        "stage": log_prefix,
    }
    (output_dir / metrics_file_name).write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def train_self_supervised(
    model, train_loader, device: torch.device, config, output_dir: Path
) -> dict:
    ssl_config = config.train.self_supervised
    return _train_two_view_stage(
        model=model,
        train_loader=train_loader,
        device=device,
        config=config,
        epochs=int(ssl_config.epochs),
        lr=float(ssl_config.lr),
        weight_decay=float(ssl_config.weight_decay),
        temperature=float(ssl_config.temperature),
        use_amp=bool(config.train.use_amp),
        log_every_steps=int(config.train.log_every_steps),
        output_dir=output_dir,
        metrics_file_name="self_supervised_metrics.json",
        checkpoint_file_name="checkpoint_self_supervised.pt",
        log_prefix="self-supervised",
        loss_fn=lambda projections, _labels, loss_temperature: simclr_loss(
            projections, loss_temperature
        ),
    )


def train_supcon(
    model, train_loader, device: torch.device, config, output_dir: Path
) -> dict:
    metrics = _train_two_view_stage(
        model=model,
        train_loader=train_loader,
        device=device,
        config=config,
        epochs=int(config.train.epochs),
        lr=float(config.train.lr),
        weight_decay=float(config.train.weight_decay),
        temperature=float(config.train.temperature),
        use_amp=bool(config.train.use_amp),
        log_every_steps=int(config.train.log_every_steps),
        output_dir=output_dir,
        metrics_file_name="training_metrics.json",
        checkpoint_file_name="checkpoint.pt",
        log_prefix="train",
        loss_fn=lambda projections, labels, loss_temperature: supervised_contrastive_loss(
            projections,
            labels,
            temperature=loss_temperature,
        ),
    )
    return metrics
