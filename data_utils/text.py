"""文本词表与编码"""
import json
import re
from pathlib import Path

import config


class TextProcessor:
    def __init__(self):
        self.symbol_to_id: dict[str, int] = {}
        self.id_to_symbol: dict[int, str] = {}

    @property
    def n_symbols(self) -> int:
        return len(self.symbol_to_id)

    def build_vocab(self, text_files: list[Path]) -> None:
        chars: set[str] = set()
        for path in text_files:
            chars.update(self._read_and_normalize(path))
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
            if path.name.endswith(".trn"):
                text = f.readline().strip().replace(" ", "")
                return self.normalize(text)
            return self.normalize(f.read())

    @staticmethod
    def normalize(text: str) -> str:
        text = text.replace("\n", "").replace("\r", "").replace(" ", "")
        return re.sub(r"[^\u4e00-\u9fff" + re.escape(config.PUNCTUATION) + r"]", "", text)

    def text_to_sequence(self, text: str) -> list[int]:
        text = self.normalize(text)
        pad_id = self.symbol_to_id.get(config.PAD_TOKEN, 0)
        seq = [self.symbol_to_id.get(c, pad_id) for c in text]
        seq.append(self.symbol_to_id[config.EOS_TOKEN])
        return seq
