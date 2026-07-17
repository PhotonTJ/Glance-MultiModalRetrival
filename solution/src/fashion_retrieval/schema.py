from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Literal

from pydantic import BaseModel, Field, field_validator


class Garment(BaseModel):
    type: str
    color: str = "unknown"
    pattern: str = "unknown"

    @field_validator("type", "color", "pattern")
    @classmethod
    def normalize(cls, value: str) -> str:
        return value.strip().lower().replace(" ", "_")


class ImageRecord(BaseModel):
    image_id: str
    image_path: str
    caption: str = ""
    garments: list[Garment] = Field(default_factory=list)
    environment: str = "unknown"
    style: list[str] = Field(default_factory=list)
    activity: str = "unknown"
    objects: list[str] = Field(default_factory=list)
    source: str = "unknown"
    split: Literal["train", "validation", "test", "gallery"] = "gallery"

    @field_validator("environment", "activity", "source")
    @classmethod
    def normalize_scalar(cls, value: str) -> str:
        return value.strip().lower().replace(" ", "_")

    @field_validator("style", "objects")
    @classmethod
    def normalize_list(cls, values: list[str]) -> list[str]:
        return sorted({v.strip().lower().replace(" ", "_") for v in values if v.strip()})


class JudgedQuery(BaseModel):
    query_id: str
    text: str
    relevance: dict[str, int]

    @field_validator("relevance")
    @classmethod
    def valid_grades(cls, values: dict[str, int]) -> dict[str, int]:
        if any(v not in (0, 1, 2) for v in values.values()):
            raise ValueError("relevance grades must be 0, 1, or 2")
        return values


def load_jsonl(path: str | Path, model: type[BaseModel]) -> list[BaseModel]:
    with Path(path).open(encoding="utf-8") as handle:
        return [model.model_validate(json.loads(line)) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[BaseModel | dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = row.model_dump() if isinstance(row, BaseModel) else row
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

