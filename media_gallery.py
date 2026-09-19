from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gradio as gr


MEDIA_EXTENSIONS = [
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff",
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg",
    ".wav", ".mp3", ".flac", ".ogg", ".m4a",
]


def _path(value: Any) -> str:
    if isinstance(value, (str, Path)):
        return str(value)
    if isinstance(value, dict):
        for key in ("path", "name", "file_name", "orig_name"):
            path = value.get(key)
            if path:
                return str(path)
        return ""
    if isinstance(value, (list, tuple)) and value:
        # Gallery values can be returned as (path, caption) pairs.
        return _path(value[0])
    return str(getattr(value, "path", None) or getattr(value, "name", None) or "")


class RuntimeMediaGallery:
    """Small v2-only media gallery with Wan2GP-style slot controls."""

    def __init__(self, label: str = "Runtime media inputs (slot order)"):
        self.label = label
        self.gallery = None
        self.state = None
        self.add_button = None
        self.remove_button = None
        self.left_button = None
        self.right_button = None
        self.clear_button = None

    @staticmethod
    def _items(value) -> list[str]:
        values = value if isinstance(value, (list, tuple)) else ([value] if value else [])
        return [path for value in values if (path := _path(value)) and os.path.isfile(path)]

    @staticmethod
    def _selected(state) -> int | None:
        value = state if isinstance(state, dict) else {}
        selected = value.get("selected")
        return selected if isinstance(selected, int) else None

    def _select(self, state, event: gr.SelectData):
        selected = getattr(event, "index", None)
        if isinstance(selected, (list, tuple)):
            selected = selected[0] if selected else None
        selected = selected if isinstance(selected, int) else None
        state = dict(state or {})
        state["selected"] = selected
        return state

    def _changed(self, value, state):
        state = dict(state or {})
        items = self._items(value)
        if not items and value not in (None, [], ()):
            # Keep the canonical state when Gradio sends gallery metadata
            # that cannot be converted back to local paths.
            items = self._items(state.get("items"))
        selected = self._selected(state)
        if selected is None or not 0 <= selected < len(items):
            selected = len(items) - 1 if items else None
        state.update(items=items, selected=selected)
        return state

    def _add(self, files, state, current):
        state = dict(state or {})
        current = self._items(state.get("items")) if "items" in state else self._items(current)
        incoming = self._items(files)
        seen = {os.path.normcase(os.path.abspath(path)) for path in current}
        incoming = [path for path in incoming if os.path.normcase(os.path.abspath(path)) not in seen]
        selected = self._selected(state)
        insert_at = len(current) if selected is None else selected + 1
        merged = current[:insert_at] + incoming + current[insert_at:]
        state.update(items=merged, selected=(insert_at + len(incoming) - 1 if incoming else selected))
        return gr.update(value=merged, selected_index=state["selected"]), state

    def _remove(self, state, current):
        state = dict(state or {})
        current = self._items(state.get("items")) if "items" in state else self._items(current)
        selected = self._selected(state)
        if selected is None or not 0 <= selected < len(current):
            return gr.update(value=current), state
        current.pop(selected)
        selected = min(selected, len(current) - 1) if current else None
        state.update(items=current, selected=selected)
        return gr.update(value=current, selected_index=selected), state

    def _move(self, delta, state, current):
        state = dict(state or {})
        current = self._items(state.get("items")) if "items" in state else self._items(current)
        selected = self._selected(state)
        target = (selected + delta) if selected is not None else -1
        if selected is None or not 0 <= selected < len(current) or not 0 <= target < len(current):
            return gr.update(value=current, selected_index=selected), state
        current[selected], current[target] = current[target], current[selected]
        state.update(items=current, selected=target)
        return gr.update(value=current, selected_index=target), state

    def _clear(self, state):
        state = dict(state or {})
        state.update(items=[], selected=None)
        return gr.update(value=[], selected_index=None), state

    def mount(self):
        with gr.Column(elem_classes=["w2gp-v2-media-gallery"]):
            self.state = gr.State({"items": [], "selected": None})
            self.gallery = gr.Gallery(
                value=[],
                label=self.label,
                columns=5,
                height=260,
                object_fit="contain",
                preview=True,
                show_fullscreen_button=True,
                file_types=MEDIA_EXTENSIONS,
                interactive=True,
            )
            with gr.Row(equal_height=True, elem_classes=["w2gp-v2-media-controls"]):
                self.add_button = gr.UploadButton("Add", file_types=MEDIA_EXTENSIONS, file_count="multiple", size="sm", min_width=1)
                self.remove_button = gr.Button("Remove", size="sm", min_width=1)
                self.left_button = gr.Button("◀ Left", size="sm", min_width=1)
                self.right_button = gr.Button("Right ▶", size="sm", min_width=1)
                self.clear_button = gr.Button("Clear", size="sm", min_width=1)

        self.gallery.select(self._select, inputs=[self.state], outputs=[self.state], trigger_mode="always_last")
        self.gallery.change(self._changed, inputs=[self.gallery, self.state], outputs=[self.state], trigger_mode="always_last")
        self.add_button.upload(self._add, inputs=[self.add_button, self.state, self.gallery], outputs=[self.gallery, self.state], trigger_mode="always_last")
        self.remove_button.click(self._remove, inputs=[self.state, self.gallery], outputs=[self.gallery, self.state], trigger_mode="always_last")
        self.left_button.click(lambda state, current: self._move(-1, state, current), inputs=[self.state, self.gallery], outputs=[self.gallery, self.state], trigger_mode="always_last")
        self.right_button.click(lambda state, current: self._move(1, state, current), inputs=[self.state, self.gallery], outputs=[self.gallery, self.state], trigger_mode="always_last")
        self.clear_button.click(self._clear, inputs=[self.state], outputs=[self.gallery, self.state], trigger_mode="always_last")
        return self
