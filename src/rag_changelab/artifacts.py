"""Portable JSON artifacts and canonical text-directory snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from .models import Document, Snapshot

T = TypeVar("T", bound=BaseModel)


def read_model(path: str | Path, model: type[T]) -> T:
    return model.model_validate_json(Path(path).read_text(encoding="utf-8"))


def write_model(path: str | Path, value: BaseModel | dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def snapshot_directory(root: str | Path, *, name: str | None = None) -> Snapshot:
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("snapshot input must be a directory")
    documents = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt"} or not path.is_file():
            continue
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"source symlink leaves snapshot directory: {path.name}")
        documents.append(
            Document(
                source_id=path.relative_to(root).as_posix(),
                content=path.read_text(encoding="utf-8"),
            )
        )
    if not documents:
        raise ValueError("snapshot directory has no UTF-8 .md or .txt documents")
    return Snapshot(name=name or root.name, documents=tuple(documents))
