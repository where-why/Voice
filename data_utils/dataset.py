"""TTS 数据集与 DataLoader 整理函数"""
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

import config
from data_utils.audio import AudioProcessor
from data_utils.text import TextProcessor


def _split_into_segments(
    text_ids: torch.Tensor,
    mel: torch.Tensor,
    sample_id: str,
) -> list[dict[str, Any]]:
    """按时间顺序将长样本切分为固定长度片段"""
    mel_len = mel.size(1)
    text_len = text_ids.size(0)
    segments = []
    seg_idx = 0
    start = 0

    while start < mel_len:
        end = min(start + config.SEGMENT_MEL_FRAMES, mel_len)
        seg_mel = mel[:, start:end]

        t_start = int(start / mel_len * text_len)
        t_end = int(end / mel_len * text_len)
        t_end = max(t_end, t_start + 1)
        seg_text = text_ids[t_start:t_end]
        if seg_text.size(0) > config.SEGMENT_TEXT_LEN:
            seg_text = seg_text[: config.SEGMENT_TEXT_LEN]

        segments.append(
            {
                "text": seg_text,
                "mel": seg_mel,
                "id": f"{sample_id}_seg{seg_idx}",
            }
        )
        seg_idx += 1
        start += config.SEGMENT_MEL_FRAMES

    return segments


class TTSDataset(Dataset):
    """文本-语音配对数据集（长音频按顺序切分为多段）"""

    def __init__(
        self,
        samples: list[dict[str, Any]],
        text_processor: TextProcessor,
        audio_processor: AudioProcessor,
        use_cache: bool = True,
    ):
        self.items: list[dict[str, Any]] = []

        for sample in samples:
            text_ids = torch.LongTensor(
                text_processor.text_to_sequence(
                    text_processor._read_and_normalize(sample["label_path"])
                )
            )

            mel_path = sample.get("mel_cache")
            if use_cache and mel_path and Path(mel_path).exists():
                mel = torch.load(mel_path, weights_only=True)
            else:
                mel = audio_processor.load_mel(sample["audio_path"])

            self.items.extend(_split_into_segments(text_ids, mel, sample["id"]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.items[idx]


def tts_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """按 batch 内最大长度 padding"""
    text_lengths = torch.LongTensor([b["text"].size(0) for b in batch])
    mel_lengths = torch.LongTensor([b["mel"].size(1) for b in batch])

    max_text_len = text_lengths.max().item()
    max_mel_len = mel_lengths.max().item()

    texts = torch.zeros(len(batch), max_text_len, dtype=torch.long)
    mels = torch.zeros(len(batch), config.N_MELS, max_mel_len)
    text_padded = torch.ones(len(batch), max_text_len, dtype=torch.bool)

    for i, b in enumerate(batch):
        t_len = b["text"].size(0)
        m_len = b["mel"].size(1)
        texts[i, :t_len] = b["text"]
        mels[i, :, :m_len] = b["mel"]
        text_padded[i, :t_len] = False

    return {
        "text": texts,
        "text_lengths": text_lengths,
        "text_padded": text_padded,
        "mel": mels,
        "mel_lengths": mel_lengths,
        "ids": [b["id"] for b in batch],
    }


def discover_samples(voice_dir: Path, label_dir: Path, cache_dir: Path) -> list[dict[str, Any]]:
    """扫描 voice/ 与 label/ 目录，按文件名配对"""
    audio_exts = {".m4a", ".wav", ".mp3", ".flac"}
    samples = []

    for label_path in sorted(label_dir.glob("*.txt")):
        stem = label_path.stem
        audio_path = None
        for ext in audio_exts:
            candidate = voice_dir / f"{stem}{ext}"
            if candidate.exists():
                audio_path = candidate
                break
        if audio_path is None:
            continue
        samples.append(
            {
                "id": stem,
                "audio_path": audio_path,
                "label_path": label_path,
                "mel_cache": cache_dir / "mel" / f"{stem}.pt",
            }
        )
    return samples
