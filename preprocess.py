"""THCHS-30 预处理"""
import torch
from tqdm import tqdm

import config
from data_utils.audio import AudioProcessor
from data_utils.dataset import discover_thchs30_samples
from data_utils.text import TextProcessor

FORCE = False


def _cache_mel(samples, audio_processor: AudioProcessor) -> None:
    for sample in tqdm(samples, desc="提取 Mel"):
        cache_path = sample["mel_cache"]
        if cache_path.exists() and not FORCE:
            if sample.get("mel_frames", 0) <= 0:
                sample["mel_frames"] = torch.load(cache_path, weights_only=True).size(1)
            continue
        mel = audio_processor.load_mel(sample["audio_path"])
        sample["mel_frames"] = mel.size(1)
        torch.save(mel, cache_path)


def preprocess() -> None:
    train_samples = discover_thchs30_samples(config.THCHS30_TRAIN_DIR, config.TRAIN_CACHE_DIR)
    dev_samples = discover_thchs30_samples(config.THCHS30_DEV_DIR, config.DEV_CACHE_DIR)
    if not train_samples:
        raise FileNotFoundError(f"未找到训练集: {config.THCHS30_TRAIN_DIR}")

    for cache_dir in (config.TRAIN_CACHE_DIR, config.DEV_CACHE_DIR):
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "mel").mkdir(exist_ok=True)

    print(f"训练 {len(train_samples)} 条，验证 {len(dev_samples)} 条")

    label_files = [s["label_path"] for s in train_samples]
    label_files.extend(s["label_path"] for s in dev_samples)

    vocab_path = config.TRAIN_CACHE_DIR / "vocab.json"
    text_processor = TextProcessor()
    if vocab_path.exists() and not FORCE:
        text_processor.load(vocab_path)
    else:
        text_processor.build_vocab(label_files)
        text_processor.save(vocab_path)
    print(f"词表: {text_processor.n_symbols} 个符号")

    audio_processor = AudioProcessor()
    _cache_mel(train_samples, audio_processor)
    if dev_samples:
        _cache_mel(dev_samples, audio_processor)

    torch.save(train_samples, config.TRAIN_CACHE_DIR / "samples.pt")
    if dev_samples:
        torch.save(dev_samples, config.DEV_CACHE_DIR / "samples.pt")
    print("预处理完成")


if __name__ == "__main__":
    preprocess()
