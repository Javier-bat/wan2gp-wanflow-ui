from __future__ import annotations

import copy
import json
import re
import time
from pathlib import Path
from typing import Any

from .graph_model import normalize_graph


SCHEMA = "wan2gp-wanflow-block-v1"


class BlockStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def slug(value: Any) -> str:
        result = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value or "block").strip()).strip("-_").lower()
        return result[:80] or "block"

    def catalog(self) -> list[dict[str, Any]]:
        result = []
        for path in sorted(self.root.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("schema") != SCHEMA:
                    continue
                result.append({
                    "id": path.stem,
                    "name": str(payload.get("name") or path.stem),
                    "description": str(payload.get("description") or ""),
                    "nodes": payload.get("nodes") or [],
                    "edges": payload.get("edges") or [],
                    "groups": payload.get("groups") or [],
                })
            except (OSError, ValueError, TypeError):
                continue
        return result

    def save_group(self, graph: dict[str, Any], group_id: str, name: str | None = None) -> Path:
        normalized = normalize_graph(graph)
        group = next((item for item in normalized.get("groups") or [] if item.get("id") == group_id), None)
        if not group:
            raise ValueError("The selected group does not exist.")
        node_ids = set(group.get("node_ids") or [])
        nodes = [copy.deepcopy(node) for node in normalized.get("nodes") or [] if node.get("id") in node_ids]
        edges = [copy.deepcopy(edge) for edge in normalized.get("edges") or [] if edge.get("source", {}).get("node") in node_ids and edge.get("target", {}).get("node") in node_ids]
        block_group = copy.deepcopy(group)
        block_group["title"] = str(name or block_group.get("title") or "Reusable block").strip()
        payload = {
            "schema": SCHEMA,
            "version": 1,
            "name": block_group["title"],
            "description": "Reusable Wan2GP v2 subgraph.",
            "updated_at": int(time.time()),
            "nodes": nodes,
            "edges": edges,
            "groups": [block_group],
        }
        target = self.root / f"{self.slug(payload['name'])}.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(target)
        return target

    def delete(self, block_id: str) -> Path:
        target = self.root / f"{self.slug(block_id)}.json"
        if not target.is_file():
            raise FileNotFoundError(target)
        target.unlink()
        return target
