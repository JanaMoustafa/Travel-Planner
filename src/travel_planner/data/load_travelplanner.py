"""Load osunlp/TravelPlanner splits via Hugging Face datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from datasets import Dataset

SplitName = Literal["train", "validation", "test"]


def load_split(
    name: str = "osunlp/TravelPlanner",
    split: SplitName = "validation",
    *,
    token: str | None = None,
) -> "Dataset":
    """Load a named config with train/test swapped.

    The upstream dataset exposes a small labeled train split and a larger test split.
    For this project we remap them so callers see:
    - train -> upstream test (1000 rows)
    - validation -> upstream validation
    - test -> upstream train (45 rows)
    """
    from datasets import load_dataset

    kwargs: dict[str, Any] = {}
    if token:
        kwargs["token"] = token
    upstream_split = {"train": "test", "validation": "validation", "test": "train"}[split]
    ds = load_dataset(name, upstream_split, **kwargs)
    return ds[upstream_split]
