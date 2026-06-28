"""数据集、THCHS-30 扫描与 DataLoader"""
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

import config
from data_utils.audio import AudioProcessor
from data_utils.text import TextProcessor


def discover_thchs30_samples(split_dir: Path, cache_dir: Path) -> list[dict[str, Any]]:
    """扫描 THCHS-30 的 train/dev 目录，配对 .wav 与 .wav.trn"""
    if not split_dir.exists():
        return []

    samples: list[dict[str, Any]] = []
    for wav_path in sorted(split_dir.glob("*.wav")):
        trn_path = wav_path.with_name(wav_path.name + ".trn")
        if not trn_path.exists():
            continue
        stem = wav_path.stem
        samples.append(
            {
                "id": stem,
                "audio_path": wav_path.resolve(),
                "label_path": trn_path.resolve(),
                "mel_cache": cache_dir / "mel" / f"{stem}.pt",
                "mel_frames": 0,
            }
        )
    return samples


def _segment_count(mel_frames: int) -> int:
    if mel_frames <= 0:
        return 1
    return (mel_frames + config.SEGMENT_MEL_FRAMES - 1) // config.SEGMENT_MEL_FRAMES


def _load_mel(sample: dict[str, Any], audio_processor: AudioProcessor, use_cache: bool) -> torch.Tensor:
    mel_path = sample.get("mel_cache")
    if use_cache and mel_path and Path(mel_path).exists():
        return torch.load(mel_path, weights_only=True)
    return audio_processor.load_mel(sample["audio_path"])


def _get_segment(
    text_ids: torch.Tensor,
    mel: torch.Tensor,
    seg_idx: int,
    sample_id: str,
) -> dict[str, Any]:
    mel_len = mel.size(1)
    text_len = text_ids.size(0)
    start = seg_idx * config.SEGMENT_MEL_FRAMES
    end = min(start + config.SEGMENT_MEL_FRAMES, mel_len)
    seg_mel = mel[:, start:end]

    t_start = int(start / mel_len * text_len)
    t_end = int(end / mel_len * text_len)
    t_end = max(t_end, t_start + 1)
    seg_text = text_ids[t_start:t_end]
    if seg_text.size(0) > config.SEGMENT_TEXT_LEN:
        seg_text = seg_text[: config.SEGMENT_TEXT_LEN]

    return {"text": seg_text, "mel": seg_mel, "id": f"{sample_id}_seg{seg_idx}"}


class TTSDataset(Dataset):
    """THCHS-30 文本-语音数据集，长音频按顺序切分"""

    def __init__(
        self,
        samples: list[dict[str, Any]],
        text_processor: TextProcessor,
        audio_processor: AudioProcessor,
        use_cache: bool = True,
    ):
        self.samples = samples
        self.text_processor = text_processor
        self.audio_processor = audio_processor
        self.use_cache = use_cache
        self.index: list[tuple[int, int]] = []

        for sample_i, sample in enumerate(samples):
            mel_frames = sample.get("mel_frames") or 0
            if mel_frames <= 0 and use_cache and sample.get("mel_cache") and Path(sample["mel_cache"]).exists():
                mel_frames = torch.load(sample["mel_cache"], weights_only=True).size(1)
            for seg_i in range(_segment_count(mel_frames)):
                self.index.append((sample_i, seg_i))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample_i, seg_i = self.index[idx]
        sample = self.samples[sample_i]
        text = self.text_processor._read_and_normalize(sample["label_path"])
        text_ids = torch.LongTensor(self.text_processor.text_to_sequence(text))
        mel = _load_mel(sample, self.audio_processor, self.use_cache)
        return _get_segment(text_ids, mel, seg_i, sample["id"])


def tts_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    text_lengths = torch.LongTensor([b["text"].size(0) for b in batch])
    mel_lengths = torch.LongTensor([b["mel"].size(1) for b in batch])
    max_text_len = int(text_lengths.max())
    max_mel_len = int(mel_lengths.max())

    texts = torch.zeros(len(batch), max_text_len, dtype=torch.long)
    mels = torch.zeros(len(batch), config.N_MELS, max_mel_len)

    for i, b in enumerate(batch):
        t_len = b["text"].size(0)
        m_len = b["mel"].size(1)
        texts[i, :t_len] = b["text"]
        mels[i, :, :m_len] = b["mel"]

    return {
        "text": texts,
        "text_lengths": text_lengths,
        "mel": mels,
        "mel_lengths": mel_lengths,
        "ids": [b["id"] for b in batch],
    }
