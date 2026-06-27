"""Tacotron2 模型训练"""
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from data_utils.audio import AudioProcessor
from data_utils.dataset import TTSDataset, discover_samples, tts_collate_fn
from data_utils.text import TextProcessor
from models.loss import Tacotron2Loss
from models.tacotron2 import Tacotron2

# ==================== 配置（按需修改） ====================
ROOT = Path(__file__).resolve().parent
TRAIN_VOICE_DIR = ROOT / "train_data" / "voice"
TRAIN_LABEL_DIR = ROOT / "train_data" / "label"
TRAIN_CACHE_DIR = ROOT / "train_data" / "processed"

VAL_VOICE_DIR = ROOT / "vel_data" / "voice"
VAL_LABEL_DIR = ROOT / "vel_data" / "label"
VAL_CACHE_DIR = ROOT / "vel_data" / "processed"

CHECKPOINT_DIR = ROOT / "checkpoints"

EPOCHS = 500
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-6
GRAD_CLIP = 1.0
LOG_INTERVAL = 10
SEED = 42
RESUME = False  # True 时从 tacotron2_latest.pt 恢复训练
USE_AMP = True  # 混合精度，节省显存

# 最佳模型选型指标（在验证集上，越低越好）
# mel         -> val_mel_loss
# mel_postnet -> val_mel_postnet_loss（推荐）
# combined    -> COMBINED_MEL_WEIGHT * mel + COMBINED_GATE_WEIGHT * gate
BEST_METRIC = "mel_postnet"
COMBINED_MEL_WEIGHT = 0.9
COMBINED_GATE_WEIGHT = 0.1
# ==========================================================

_BEST_METRIC_LABELS = {
    "mel": "val_mel_loss",
    "mel_postnet": "val_mel_postnet_loss",
    "combined": f"{COMBINED_MEL_WEIGHT}*mel+{COMBINED_GATE_WEIGHT}*gate",
}


def _best_metric_score(val_metrics: dict[str, float]) -> float:
    if BEST_METRIC == "mel":
        return val_metrics["mel"]
    if BEST_METRIC == "mel_postnet":
        return val_metrics["mel_postnet"]
    if BEST_METRIC == "combined":
        return COMBINED_MEL_WEIGHT * val_metrics["mel"] + COMBINED_GATE_WEIGHT * val_metrics["gate"]
    raise ValueError(f"未知 BEST_METRIC: {BEST_METRIC}，可选 mel / mel_postnet / combined")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_checkpoint(path: Path, state: dict) -> None:
    torch.save(state, path)


def _load_samples(cache_dir: Path, voice_dir: Path, label_dir: Path) -> list:
    samples_path = cache_dir / "samples.pt"
    if samples_path.exists():
        return torch.load(samples_path, weights_only=False)
    return discover_samples(voice_dir, label_dir, cache_dir)


def load_resources(device: torch.device):
    vocab_path = TRAIN_CACHE_DIR / "vocab.json"
    if not vocab_path.exists():
        print("未找到预处理缓存，正在运行 preprocess ...")
        from preprocess import preprocess

        preprocess()

    text_processor = TextProcessor()
    text_processor.load(vocab_path)

    train_samples = _load_samples(TRAIN_CACHE_DIR, TRAIN_VOICE_DIR, TRAIN_LABEL_DIR)
    if not train_samples:
        raise RuntimeError("没有可用训练样本")

    val_samples = _load_samples(VAL_CACHE_DIR, VAL_VOICE_DIR, VAL_LABEL_DIR)
    if not val_samples:
        raise RuntimeError(f"没有可用验证样本，请检查 {VAL_VOICE_DIR} 与 {VAL_LABEL_DIR}")

    audio_processor = AudioProcessor()

    train_dataset = TTSDataset(train_samples, text_processor, audio_processor, use_cache=True)
    val_dataset = TTSDataset(val_samples, text_processor, audio_processor, use_cache=True)
    print(f"训练片段: {len(train_dataset)}，验证片段: {len(val_dataset)}（每段 {config.SEGMENT_MEL_FRAMES} 帧）")

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(BATCH_SIZE, len(train_dataset)),
        shuffle=False,
        collate_fn=tts_collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(BATCH_SIZE, len(val_dataset)),
        shuffle=False,
        collate_fn=tts_collate_fn,
        num_workers=0,
    )

    model = Tacotron2(n_symbols=text_processor.n_symbols).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    seg_sec = config.SEGMENT_MEL_FRAMES * config.HOP_LENGTH / config.SAMPLE_RATE
    print(f"模型参数量: {param_count / 1e6:.2f}M，每段时长约: {seg_sec:.1f}s")
    print(f"最佳模型指标: {_BEST_METRIC_LABELS[BEST_METRIC]}")

    criterion = Tacotron2Loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    return model, criterion, optimizer, train_loader, val_loader, text_processor


@torch.no_grad()
def validate(
    model: Tacotron2,
    criterion: Tacotron2Loss,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals = {"total": 0.0, "mel": 0.0, "mel_postnet": 0.0, "gate": 0.0}
    n = 0

    for batch in val_loader:
        text = batch["text"].to(device)
        text_lengths = batch["text_lengths"].to(device)
        mel = batch["mel"].to(device)
        mel_lengths = batch["mel_lengths"].to(device)

        with torch.amp.autocast("cuda", enabled=USE_AMP and device.type == "cuda"):
            mel_pred, mel_postnet, gate_pred = model(text, text_lengths, mel)
            losses = criterion(mel_pred, mel_postnet, gate_pred, mel, mel_lengths)

        for k in totals:
            totals[k] += losses[k].item()
        n += 1

    return {k: v / n for k, v in totals.items()}


def train() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    model, criterion, optimizer, train_loader, val_loader, text_processor = load_resources(device)
    start_epoch = 1
    best_metric_value = float("inf")

    resume_path = CHECKPOINT_DIR / "tacotron2_latest.pt"
    if RESUME and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_metric_value = ckpt.get("best_metric_value", ckpt.get("best_val_loss", float("inf")))
        print(
            f"从 {resume_path} 恢复，epoch {start_epoch}，"
            f"best_{_BEST_METRIC_LABELS[BEST_METRIC]}={best_metric_value:.4f}"
        )

    global_step = 0
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and device.type == "cuda")

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]")

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
            global_step += 1

            if global_step % LOG_INTERVAL == 0:
                pbar.set_postfix(loss=f"{losses['total'].item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        val_metrics = validate(model, criterion, val_loader, device)
        metric_score = _best_metric_score(val_metrics)

        print(
            f"Epoch {epoch} | train_loss={avg_train_loss:.4f} | "
            f"val_mel={val_metrics['mel']:.4f} | val_mel_postnet={val_metrics['mel_postnet']:.4f} | "
            f"val_gate={val_metrics['gate']:.4f} | "
            f"{_BEST_METRIC_LABELS[BEST_METRIC]}={metric_score:.4f}"
        )

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "n_symbols": text_processor.n_symbols,
            "train_loss": avg_train_loss,
            "val_metrics": val_metrics,
            "best_metric": BEST_METRIC,
            "best_metric_value": best_metric_value,
        }

        _save_checkpoint(CHECKPOINT_DIR / "tacotron2_latest.pt", state)
        print("已保存最新模型: tacotron2_latest.pt")

        if metric_score < best_metric_value:
            best_metric_value = metric_score
            state["best_metric_value"] = best_metric_value
            _save_checkpoint(CHECKPOINT_DIR / "tacotron2_best.pt", state)
            print(
                f"已保存最佳模型: tacotron2_best.pt"
                f"（{_BEST_METRIC_LABELS[BEST_METRIC]}={best_metric_value:.4f}）"
            )

    print(f"训练完成，最佳 {_BEST_METRIC_LABELS[BEST_METRIC]}: {best_metric_value:.4f}")


if __name__ == "__main__":
    train()
