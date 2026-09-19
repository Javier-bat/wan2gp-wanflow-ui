from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from .graph_model import graph_for_json, normalize_graph


class GraphStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def slug(value: Any) -> str:
        result = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value or "workflow").strip()).strip("-_").lower()
        return result[:80] or "workflow"

    def choices(self) -> list[tuple[str, str]]:
        result = []
        for path in sorted(self.root.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                result.append((str(payload.get("name") or path.stem), path.stem))
            except (OSError, ValueError, TypeError):
                result.append((path.stem, path.stem))
        return result

    def save(self, graph: dict[str, Any], name: str) -> tuple[dict[str, Any], Path]:
        normalized = normalize_graph(graph)
        normalized["name"] = str(name or normalized.get("name") or "Workflow").strip()
        slug = self.slug(normalized["name"])
        normalized["updated_at"] = int(time.time())
        target = self.root / f"{slug}.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(graph_for_json(normalized), indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(target)
        return normalized, target

    def load(self, slug: str) -> dict[str, Any]:
        target = self.root / f"{self.slug(slug)}.json"
        if not target.is_file():
            raise FileNotFoundError(target)
        return normalize_graph(json.loads(target.read_text(encoding="utf-8")))

    def delete(self, slug: str) -> None:
        target = self.root / f"{self.slug(slug)}.json"
        if target.is_file():
            target.unlink()

