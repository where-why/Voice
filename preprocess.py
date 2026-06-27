"""数据预处理：构建词表、缓存 Mel 频谱"""
from pathlib import Path

import torch
from tqdm import tqdm

from data_utils.audio import AudioProcessor
from data_utils.dataset import discover_samples
from data_utils.text import TextProcessor

# ==================== 配置（按需修改） ====================
ROOT = Path(__file__).resolve().parent
TRAIN_VOICE_DIR = ROOT / "train_data" / "voice"
TRAIN_LABEL_DIR = ROOT / "train_data" / "label"
TRAIN_CACHE_DIR = ROOT / "train_data" / "processed"

VAL_VOICE_DIR = ROOT / "vel_data" / "voice"
VAL_LABEL_DIR = ROOT / "vel_data" / "label"
VAL_CACHE_DIR = ROOT / "vel_data" / "processed"

FORCE = False  # True 时强制重新处理
# ==========================================================


def _cache_mel_samples(samples, audio_processor, force: bool) -> None:
    for sample in tqdm(samples, desc="提取 Mel 频谱"):
        cache_path = sample["mel_cache"]
        if cache_path.exists() and not force:
            continue
        mel = audio_processor.load_mel(sample["audio_path"])
        torch.save(mel, cache_path)


def preprocess() -> None:
    train_samples = discover_samples(TRAIN_VOICE_DIR, TRAIN_LABEL_DIR, TRAIN_CACHE_DIR)
    if not train_samples:
        raise FileNotFoundError(f"未找到训练数据，请检查 {TRAIN_VOICE_DIR} 与 {TRAIN_LABEL_DIR}")

    val_samples = discover_samples(VAL_VOICE_DIR, VAL_LABEL_DIR, VAL_CACHE_DIR)
    print(f"训练样本: {len(train_samples)} 条，验证样本: {len(val_samples)} 条")

    TRAIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (TRAIN_CACHE_DIR / "mel").mkdir(parents=True, exist_ok=True)
    if val_samples:
        VAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (VAL_CACHE_DIR / "mel").mkdir(parents=True, exist_ok=True)

    vocab_path = TRAIN_CACHE_DIR / "vocab.json"
    text_processor = TextProcessor()
    label_files = [s["label_path"] for s in train_samples]
    if val_samples:
        label_files.extend([s["label_path"] for s in val_samples])

    if vocab_path.exists() and not FORCE:
        text_processor.load(vocab_path)
        print(f"加载已有词表: {vocab_path} ({text_processor.n_symbols} 个符号)")
    else:
        text_processor.build_vocab(label_files)
        text_processor.save(vocab_path)
        print(f"构建词表: {text_processor.n_symbols} 个符号 -> {vocab_path}")

    audio_processor = AudioProcessor()
    _cache_mel_samples(train_samples, audio_processor, FORCE)
    if val_samples:
        _cache_mel_samples(val_samples, audio_processor, FORCE)

    torch.save(train_samples, TRAIN_CACHE_DIR / "samples.pt")
    if val_samples:
        torch.save(val_samples, VAL_CACHE_DIR / "samples.pt")

    print(f"预处理完成\n  训练缓存: {TRAIN_CACHE_DIR}\n  验证缓存: {VAL_CACHE_DIR}")


if __name__ == "__main__":
    preprocess()
