"""The one tokenizer that counts every budget."""
from __future__ import annotations

from .rerankers.text import WhitespaceTokenizer


class HFTokenizer:
    def __init__(self, name: str, revision: str | None = None):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name, revision=revision)
        self.name = f"{name}@{revision}"

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)


def get_tokenizer(name: str, revision: str | None = None):
    return WhitespaceTokenizer() if name == "whitespace" else HFTokenizer(name, revision)
