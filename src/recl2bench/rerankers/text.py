"""Shared rendering and token budgets. Every model gets identical text.

Budgets are counted with ONE tokenizer (configs/dataset.yaml: budget.tokenizer)
and applied before any model-specific formatting.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Protocol

from .base import ProfileItem, UserProfile


class Tokenizer(Protocol):
    def encode(self, text: str) -> list: ...
    def decode(self, ids: list) -> str: ...


class WhitespaceTokenizer:
    """Test-only stand-in. Real runs use the HF tokenizer named in the config."""
    def encode(self, text):
        return text.split()

    def decode(self, ids):
        return " ".join(ids)


@dataclass
class Truncation:
    text: str
    truncated: bool
    n_tokens: int


def cut(text: str, tok: Tokenizer, max_tokens: int) -> Truncation:
    ids = tok.encode(text)
    if len(ids) <= max_tokens:
        return Truncation(text, False, len(ids))
    return Truncation(tok.decode(ids[:max_tokens]), True, max_tokens)


def _line(it: ProfileItem) -> str:
    attrs = "; ".join(it.attrs[:2])
    return f"- {it.title} (rated {it.rating:g}){' | ' + attrs if attrs else ''}"


def render_profile(p: UserProfile, tok: Tokenizer, max_tokens: int) -> tuple[UserProfile, bool]:
    """Liked first (most recent first), then disliked; empty disliked slot kept explicit."""
    disliked = "\n".join(_line(i) for i in p.disliked) if p.disliked else "- (none)"
    text = ("Items this shopper rated highly:\n" + "\n".join(_line(i) for i in p.liked)
            + "\nItems this shopper rated low:\n" + disliked)
    t = cut(text, tok, max_tokens)
    return replace(p, text=t.text), t.truncated


ITEM_FIELDS = ("title", "features", "description", "store", "categories", "price")


def render_item(meta: dict, tok: Tokenizer, max_tokens: int) -> Truncation:
    parts = []
    for f in ITEM_FIELDS:
        v = meta.get(f)
        if v is None or v == "" or v == []:
            continue
        if isinstance(v, (list, tuple)):
            v = "; ".join(map(str, v))
        parts.append(f"{f.capitalize()}: {v}")
    return cut("\n".join(parts), tok, max_tokens)
