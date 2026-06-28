"""文字转语音推理"""
from pathlib import Path

import torch
import torchaudio

import config
from data_utils.audio import AudioProcessor
from data_utils.text import TextProcessor
from models.tacotron2 import Tacotron2

TEXT = "绿是阳春烟景大块的底色四月的林峦更是绿得鲜活秀媚诗意盎然"
CHECKPOINT = config.CHECKPOINT_DIR / "tacotron2_best.pt"
OUTPUT = config.OUTPUT_DIR / "output.wav"


def synthesize(text: str, checkpoint: Path, output_path: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_path = config.TRAIN_CACHE_DIR / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError("请先运行 preprocess.py")

    text_processor = TextProcessor()
    text_processor.load(vocab_path)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = Tacotron2(n_symbols=ckpt.get("n_symbols", text_processor.n_symbols)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    text_ids = torch.LongTensor([text_processor.text_to_sequence(text)]).to(device)
    text_lengths = torch.LongTensor([text_ids.size(1)]).to(device)

    with torch.no_grad():
        mel = model.infer(text_ids, text_lengths)

    waveform = AudioProcessor().mel_to_wav(mel.squeeze(0).cpu())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), waveform.unsqueeze(0), config.SAMPLE_RATE)
    print(f"已保存: {output_path} ({waveform.shape[-1] / config.SAMPLE_RATE:.2f}s)")


if __name__ == "__main__":
    synthesize(TEXT, CHECKPOINT, OUTPUT)
