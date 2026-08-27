from __future__ import annotations

import contextlib
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from ct_restore.config import ExperimentConfig
from ct_restore.data.dataset import CTVolumeDataset
from ct_restore.hardware import (
    autocast_dtype,
    configure_torch,
    detect_hardware,
    resolve_config,
)
from ct_restore.losses import RestorationLoss
from ct_restore.models import HybridRestoreNet

# torch.OutOfMemoryError landed in torch 2.5, but pyproject allows torch>=2.3. On the
# older releases the bare name would make the except clause itself raise AttributeError.
OutOfMemory = getattr(torch, "OutOfMemoryError", None) or getattr(
    torch.cuda, "OutOfMemoryError", RuntimeError
)


class EMA:
    """Exponential moving average with a warm-up on the decay.

    A fixed decay of 0.999 has a time constant of ~1000 updates. Gradient accumulation
    makes updates scarce -- 55 volumes at accumulation 8 is 7 per epoch -- so a 50-epoch
    run performs ~350 updates and 0.999**350 = 0.70 of the shadow is still the random
    initialisation. Inference prefers these weights, so the exported model was mostly
    noise while the validation curve looked like a smooth, healthy decline.

    The warm-up caps the decay at ``(1 + step) / (10 + step)`` so the shadow tracks the
    model closely at first and relaxes toward ``decay`` only once enough updates exist
    to support it.
    """

    def __init__(self, model: torch.nn.Module, decay: float, warmup: bool = True) -> None:
        self.decay = decay
        self.warmup = warmup
        self.updates = 0
        self.shadow = copy.deepcopy(model).eval()
        for parameter in self.shadow.parameters():
            parameter.requires_grad_(False)

    def effective_decay(self) -> float:
        if not self.warmup:
            return self.decay
        return min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        source = _unwrap_model(model)
        decay = self.effective_decay()
        self.updates += 1
        for ema_parameter, parameter in zip(
            self.shadow.parameters(), source.parameters(), strict=True
        ):
            ema_parameter.lerp_(parameter.detach(), 1.0 - decay)
        for ema_buffer, buffer in zip(self.shadow.buffers(), source.buffers(), strict=True):
            ema_buffer.copy_(buffer)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, DistributedDataParallel):
        model = model.module
    return getattr(model, "_orig_mod", model)


def _worker_init(_: int) -> None:
    torch.set_num_threads(1)


def _build_dataset(cfg: ExperimentConfig, split: str, allow_unreviewed: bool) -> CTVolumeDataset:
    return CTVolumeDataset(
        cfg.data.manifest,
        split,
        cfg.data.patch_size,
        cfg.data.hu_min,
        cfg.data.hu_max,
        cfg.data.artifact_probability,
        allow_unreviewed=allow_unreviewed,
        seed=cfg.train.seed,
        deterministic=split != "train",
    )


def train(cfg: ExperimentConfig, allow_unreviewed: bool = False) -> Path:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        torch.distributed.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo"
        )
        if torch.cuda.is_available():
            torch.cuda.set_device(rank % torch.cuda.device_count())
    profile = detect_hardware("auto" if distributed else cfg.hardware.device)
    cfg = resolve_config(cfg, profile)
    configure_torch(cfg, profile)
    _seed_everything(cfg.train.seed + rank)
    device = torch.device(profile.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if rank == 0:
        print("hardware=" + json.dumps(profile.to_dict(), sort_keys=True))

    train_data = _build_dataset(cfg, "train", allow_unreviewed)
    try:
        val_data = _build_dataset(cfg, "val", allow_unreviewed)
    except ValueError:
        val_data = None
    sampler = DistributedSampler(train_data, shuffle=True) if distributed else None
    loader_options = {
        "num_workers": cfg.data.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": cfg.data.num_workers > 0,
        "worker_init_fn": _worker_init if cfg.data.num_workers > 0 else None,
    }
    if cfg.data.num_workers > 0:
        loader_options["prefetch_factor"] = cfg.hardware.prefetch_factor
    loader = DataLoader(
        train_data,
        batch_size=cfg.train.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        **loader_options,
    )
    val_loader = (
        DataLoader(
            val_data,
            batch_size=1,
            num_workers=max(0, cfg.data.num_workers // 2),
            pin_memory=device.type == "cuda",
            persistent_workers=cfg.data.num_workers > 1,
        )
        if val_data
        else None
    )
    model = HybridRestoreNet(**cfg.model.__dict__).to(device)
    use_channels_last = cfg.hardware.channels_last_3d and device.type == "cuda"
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last_3d)
    if cfg.train.pretrained:
        state = torch.load(cfg.train.pretrained, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("ema_model", state.get("model", state)), strict=False)
    optimizer_options = {
        "lr": cfg.train.learning_rate,
        "weight_decay": cfg.train.weight_decay,
    }
    if device.type == "cuda":
        optimizer_options["fused"] = True
    try:
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_options)
    except (TypeError, RuntimeError):
        optimizer_options.pop("fused", None)
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_options)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    start_epoch = 0
    best_validation = float("inf")
    resume_state = None
    if cfg.train.resume:
        resume_state = torch.load(cfg.train.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(resume_state["model"])
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        start_epoch = int(resume_state["epoch"]) + 1
        # Restarting best_validation at infinity let the first resumed epoch overwrite
        # best.pt however bad it was.
        best_validation = float(resume_state.get("best_validation", float("inf")))
    # Built after the weights are restored, so a checkpoint without EMA state still
    # seeds the shadow from the resumed model rather than from random initialisation.
    ema = EMA(model, cfg.train.ema_decay)
    if resume_state is not None and "ema_model" in resume_state:
        ema.shadow.load_state_dict(resume_state["ema_model"])
        ema.updates = int(resume_state.get("ema_updates", 0))
    criterion = RestorationLoss(**cfg.loss.__dict__).to(device)
    if cfg.hardware.compile:
        if profile.compile_supported and hasattr(torch, "compile"):
            model = torch.compile(model, mode="default", fullgraph=False, dynamic=False)
            if rank == 0:
                print("torch.compile enabled; first iteration includes compilation warm-up")
        elif rank == 0:
            print("torch.compile requested but unsupported on this runtime; using eager mode")
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(cfg.train.amp and device.type == "cuda" and cfg.hardware.precision == "fp16"),
    )
    checkpoint_dir = Path(cfg.train.checkpoint_dir)
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "resolved_config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    for epoch in range(start_epoch, cfg.train.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for step, batch in enumerate(loader):
            inputs = batch["input"].to(device, non_blocking=True)
            if use_channels_last:
                inputs = inputs.contiguous(memory_format=torch.channels_last_3d)
            target = batch["target"].to(device, non_blocking=True)
            corrupted = batch["corrupted"].to(device, non_blocking=True)
            mask = batch["artifact_mask"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype(cfg.hardware.precision),
                enabled=cfg.train.amp and device.type == "cuda",
            ):
                try:
                    outputs = model(inputs)
                    loss, _ = criterion(outputs, target, corrupted, mask)
                    scaled_loss = loss / cfg.train.grad_accumulation
                except OutOfMemory as exc:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    raise RuntimeError(
                        "CUDA ran out of memory with resolved patch "
                        f"{cfg.data.patch_size} and base_channels={cfg.model.base_channels}. "
                        "Another process may be using VRAM; lower these values in a copied "
                        "config. The failed step was not checkpointed."
                    ) from exc
            updating = (step + 1) % cfg.train.grad_accumulation == 0 or step + 1 == len(loader)
            # Without no_sync, DDP all-reduces gradients on every micro-step even though
            # only the last one in an accumulation window is used.
            sync = contextlib.nullcontext()
            if distributed and not updating:
                sync = model.no_sync()
            try:
                with sync:
                    scaler.scale(scaled_loss).backward()
            except OutOfMemory as exc:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    "CUDA ran out of memory during backpropagation with resolved patch "
                    f"{cfg.data.patch_size}. Lower patch dimensions or base_channels; "
                    "the failed step was not checkpointed."
                ) from exc
            if updating:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
            running += float(loss.detach())
        scheduler.step()

        validation = running / max(1, len(loader))
        if val_loader and (epoch + 1) % cfg.train.validate_every == 0:
            ema.shadow.eval()
            total = 0.0
            with torch.inference_mode():
                for batch in val_loader:
                    inputs = batch["input"].to(device)
                    if use_channels_last:
                        inputs = inputs.contiguous(memory_format=torch.channels_last_3d)
                    target = batch["target"].to(device)
                    corrupted = batch["corrupted"].to(device)
                    mask = batch["artifact_mask"].to(device)
                    outputs = ema.shadow(inputs)
                    loss, _ = criterion(outputs, target, corrupted, mask)
                    total += float(loss)
            validation = total / max(1, len(val_loader))
        if rank == 0:
            source = _unwrap_model(model)
            checkpoint = {
                "epoch": epoch,
                "model": source.state_dict(),
                "ema_model": ema.shadow.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": cfg.to_dict(),
                "hardware_profile": profile.to_dict(),
                "validation_loss": validation,
                "best_validation": min(best_validation, validation),
                "ema_updates": ema.updates,
            }
            torch.save(checkpoint, checkpoint_dir / "last.pt")
            if validation < best_validation:
                best_validation = validation
                torch.save(checkpoint, checkpoint_dir / "best.pt")
            print(
                f"epoch={epoch + 1}/{cfg.train.epochs} train={running / len(loader):.5f} "
                f"validation={validation:.5f}"
            )
    if rank == 0 and ema.updates and ema.effective_decay() < cfg.train.ema_decay:
        print(
            f"Note: {ema.updates} EMA updates is short of ema_decay="
            f"{cfg.train.ema_decay} (effective {ema.effective_decay():.4f}), so the "
            "averaging is light. Warm-up keeps the shadow tracking the model, so this "
            "is informational, not a correctness problem."
        )
    if distributed:
        torch.distributed.destroy_process_group()
    return checkpoint_dir / "best.pt"
