"""模型训练"""
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from data_utils.audio import AudioProcessor
from data_utils.dataset import TTSDataset, discover_thchs30_samples, tts_collate_fn
from data_utils.text import TextProcessor
from models.loss import Tacotron2Loss
from models.tacotron2 import Tacotron2

EPOCHS = 500
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-6
GRAD_CLIP = 1.0
LOG_INTERVAL = 10
SEED = 42
RESUME = False
USE_AMP = True

BEST_METRIC = "mel_postnet"  # mel | mel_postnet | combined
COMBINED_MEL_WEIGHT = 0.9
COMBINED_GATE_WEIGHT = 0.1

_METRIC_NAME = {
    "mel": "val_mel_loss",
    "mel_postnet": "val_mel_postnet_loss",
    "combined": f"{COMBINED_MEL_WEIGHT}*mel+{COMBINED_GATE_WEIGHT}*gate",
}


def _metric_score(metrics: dict[str, float]) -> float:
    if BEST_METRIC == "mel":
        return metrics["mel"]
    if BEST_METRIC == "mel_postnet":
        return metrics["mel_postnet"]
    if BEST_METRIC == "combined":
        return COMBINED_MEL_WEIGHT * metrics["mel"] + COMBINED_GATE_WEIGHT * metrics["gate"]
    raise ValueError(f"未知 BEST_METRIC: {BEST_METRIC}")


def _load_samples(cache_dir: Path, split_dir: Path) -> list:
    path = cache_dir / "samples.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    return discover_thchs30_samples(split_dir, cache_dir)


def _build_loader(samples, text_processor, audio_processor, shuffle: bool) -> DataLoader:
    dataset = TTSDataset(samples, text_processor, audio_processor)
    return DataLoader(
        dataset,
        batch_size=min(BATCH_SIZE, len(dataset)),
        shuffle=shuffle,
        collate_fn=tts_collate_fn,
        num_workers=0,
    )


@torch.no_grad()
def _validate(model, criterion, loader, device) -> dict[str, float]:
    model.eval()
    totals = {"total": 0.0, "mel": 0.0, "mel_postnet": 0.0, "gate": 0.0}
    for batch in loader:
        text = batch["text"].to(device)
        text_lengths = batch["text_lengths"].to(device)
        mel = batch["mel"].to(device)
        mel_lengths = batch["mel_lengths"].to(device)
        with torch.amp.autocast("cuda", enabled=USE_AMP and device.type == "cuda"):
            mel_pred, mel_postnet, gate_pred = model(text, text_lengths, mel)
            losses = criterion(mel_pred, mel_postnet, gate_pred, mel, mel_lengths)
        for k in totals:
            totals[k] += losses[k].item()
    n = max(len(loader), 1)
    return {k: v / n for k, v in totals.items()}


def train() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if not (config.TRAIN_CACHE_DIR / "vocab.json").exists():
        from preprocess import preprocess
        preprocess()

    text_processor = TextProcessor()
    text_processor.load(config.TRAIN_CACHE_DIR / "vocab.json")
    audio_processor = AudioProcessor()

    train_samples = _load_samples(config.TRAIN_CACHE_DIR, config.THCHS30_TRAIN_DIR)
    dev_samples = _load_samples(config.DEV_CACHE_DIR, config.THCHS30_DEV_DIR)
    if not train_samples or not dev_samples:
        raise RuntimeError("训练集或验证集为空，请先运行 preprocess.py")

    train_loader = _build_loader(train_samples, text_processor, audio_processor, shuffle=True)
    val_loader = _build_loader(dev_samples, text_processor, audio_processor, shuffle=False)
    print(f"设备: {device} | 训练 {len(train_loader.dataset)} 段 | 验证 {len(val_loader.dataset)} 段")

    model = Tacotron2(n_symbols=text_processor.n_symbols).to(device)
    criterion = Tacotron2Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and device.type == "cuda")

    start_epoch = 1
    best_score = float("inf")
    latest = config.CHECKPOINT_DIR / "tacotron2_latest.pt"
    if RESUME and latest.exists():
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_score = ckpt.get("best_metric_value", float("inf"))

    step = 0
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for batch in pbar:
            text = batch["text"].to(device)
            text_lengths = batch["text_lengths"].to(device)
            mel = batch["mel"].to(device)
            mel_lengths = batch["mel_lengths"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP and device.type == "cuda"):
                mel_pred, mel_postnet, gate_pred = model(text, text_lengths, mel)
                losses = criterion(mel_pred, mel_postnet, gate_pred, mel, mel_lengths)

            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            train_loss += losses["total"].item()
            step += 1
            if step % LOG_INTERVAL == 0:
                pbar.set_postfix(loss=f"{losses['total'].item():.4f}")

        val_metrics = _validate(model, criterion, val_loader, device)
        score = _metric_score(val_metrics)
        print(
            f"Epoch {epoch} train={train_loss / len(train_loader):.4f} | "
            f"val_mel={val_metrics['mel']:.4f} val_postnet={val_metrics['mel_postnet']:.4f} | "
            f"{_METRIC_NAME[BEST_METRIC]}={score:.4f}"
        )

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "n_symbols": text_processor.n_symbols,
            "val_metrics": val_metrics,
            "best_metric": BEST_METRIC,
            "best_metric_value": best_score,
        }
        torch.save(state, latest)
        if score < best_score:
            best_score = score
            state["best_metric_value"] = best_score
            torch.save(state, config.CHECKPOINT_DIR / "tacotron2_best.pt")
            print(f"保存最佳模型 ({_METRIC_NAME[BEST_METRIC]}={best_score:.4f})")

    print(f"训练完成，最佳 {_METRIC_NAME[BEST_METRIC]}={best_score:.4f}")


if __name__ == "__main__":
    train()
