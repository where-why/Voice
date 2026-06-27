"""文本预处理：构建词表、文本转 ID 序列"""
import json
import re
from pathlib import Path

import config


class TextProcessor:
    """中文字符级文本处理器"""

    def __init__(self):
        self.symbol_to_id: dict[str, int] = {}
        self.id_to_symbol: dict[int, str] = {}

    @property
    def n_symbols(self) -> int:
        return len(self.symbol_to_id)

    def build_vocab(self, text_files: list[Path]) -> None:
        """从标注文件构建字符词表"""
        chars: set[str] = set()
        for path in text_files:
            text = self._read_and_normalize(path)
            chars.update(text)

        symbols = [config.PAD_TOKEN, config.EOS_TOKEN] + sorted(chars)
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self.id_to_symbol = {i: s for s, i in self.symbol_to_id.items()}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "symbol_to_id": self.symbol_to_id,
                    "id_to_symbol": {str(k): v for k, v in self.id_to_symbol.items()},
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def load(self, path: Path) -> None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.symbol_to_id = data["symbol_to_id"]
        self.id_to_symbol = {int(k): v for k, v in data["id_to_symbol"].items()}

    def _read_and_normalize(self, path: Path) -> str:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        return self.normalize(text)

    @staticmethod
    def normalize(text: str) -> str:
        """去除空白与换行，保留中文及标点"""
        text = text.replace("\n", "").replace("\r", "").replace(" ", "")
        text = re.sub(r"[^\u4e00-\u9fff" + re.escape(config.PUNCTUATION) + r"]", "", text)
        return text

    def text_to_sequence(self, text: str) -> list[int]:
        text = self.normalize(text)
        unk_id = self.symbol_to_id.get(config.PAD_TOKEN, 0)
        seq = [self.symbol_to_id.get(c, unk_id) for c in text]
        seq.append(self.symbol_to_id[config.EOS_TOKEN])
        return seq

    def sequence_to_text(self, sequence: list[int]) -> str:
        chars = []
        for idx in sequence:
            sym = self.id_to_symbol.get(idx, "")
            if sym in (config.PAD_TOKEN, config.EOS_TOKEN):
                continue
            chars.append(sym)
        return "".join(chars)
