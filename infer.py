"""TTS 推理：文本 -> Mel -> 波形"""
from pathlib import Path

import torch
import torchaudio

import config
from data_utils.audio import AudioProcessor
from data_utils.text import TextProcessor
from models.tacotron2 import Tacotron2

# ==================== 配置（按需修改） ====================
ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "train_data" / "processed"
CHECKPOINT_DIR = ROOT / "checkpoints"
OUTPUT_DIR = ROOT / "outputs"

TEXT = "乡村振兴，人才是关键。"
CHECKPOINT = CHECKPOINT_DIR / "tacotron2_best.pt"
OUTPUT = OUTPUT_DIR / "output.wav"
# ==========================================================


def synthesize(text: str, checkpoint: Path, output_path: Path) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_path = CACHE_DIR / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError("请先运行 python preprocess.py 进行数据预处理")

    text_processor = TextProcessor()
    text_processor.load(vocab_path)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    n_symbols = ckpt.get("n_symbols", text_processor.n_symbols)
    model = Tacotron2(n_symbols=n_symbols).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    text_ids = torch.LongTensor([text_processor.text_to_sequence(text)]).to(device)
    text_lengths = torch.LongTensor([text_ids.size(1)]).to(device)

    with torch.no_grad():
        mel = model.infer(text_ids, text_lengths)

    audio_processor = AudioProcessor()
    waveform = audio_processor.mel_to_wav(mel.squeeze(0).cpu())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), waveform.unsqueeze(0), config.SAMPLE_RATE)
    duration = waveform.shape[-1] / config.SAMPLE_RATE
    print(f"已生成语音: {output_path} ({duration:.2f}s)")
    return output_path


if __name__ == "__main__":
    synthesize(TEXT, CHECKPOINT, OUTPUT)
