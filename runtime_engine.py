from __future__ import annotations

import copy
import json
import math
import os
import re
import string
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Any

from PIL import Image
from shared import resolutions as resolution_api


IMAGE_EXTENSIONS = {".avif", ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}

SOURCE_MODE_CHOICES = [
    ("Automatic", "auto"),
    ("Start image", "start_image"),
    ("Continue video", "continue_video"),
    ("Reference image", "reference_image"),
    ("Reference video", "reference_video"),
    ("Control video", "control_video"),
    ("Do not connect", "none"),
]
TRANSFORMER_RETRY_DELAYS = (2.0, 4.0, 8.0)

STAGE_TYPES = {"generation", "temporal", "spatial", "soundtrack", "concatenate", "ai_analysis"}
STAGE_TYPE_ALIASES = {"rife": "temporal", "mmaudio": "soundtrack", "prismaudio": "soundtrack"}
STAGE_TYPE_LABELS = {
    "generation": "Generation",
    "temporal": "Temporal upsampling",
    "spatial": "Spatial upsampling",
    "soundtrack": "Soundtrack",
    "concatenate": "Concatenate videos",
    "ai_analysis": "AI analysis",
}
AI_MODEL_CHOICES = [
    ("Florence 2 + Llama 3.2 3B | Lowest VRAM | Single image | Standard", 1),
    ("Florence 2 + Llama Joy 8B | High VRAM | Single image | Uncensored, richer", 2),
    ("Qwen3.5VL Abliterated 4B | Medium VRAM | Multiple images | Uncensored, recommended", 3),
    ("Qwen3.5VL Abliterated 9B | Highest VRAM | Multiple images | Uncensored, best quality", 4),
]
AI_MODEL_INFO = {
    1: {
        "name": "Florence 2 + Llama 3.2 3B",
        "vram": "Lowest",
        "uncensored": False,
        "multiple_images": False,
        "summary": "Best for lightweight single-image descriptions and text requests.",
    },
    2: {
        "name": "Florence 2 + Llama Joy 8B",
        "vram": "High",
        "uncensored": True,
        "multiple_images": False,
        "summary": "Richer uncensored captions, but each request accepts only one image.",
    },
    3: {
        "name": "Qwen3.5VL Abliterated 4B",
        "vram": "Medium",
        "uncensored": True,
        "multiple_images": True,
        "summary": "Recommended balance for joint multi-image analysis and text requests.",
    },
    4: {
        "name": "Qwen3.5VL Abliterated 9B",
        "vram": "Highest",
        "uncensored": True,
        "multiple_images": True,
        "summary": "Best reasoning and detail, with the largest RAM and VRAM footprint.",
    },
}
ASPECT_RATIO_CHOICES = [
    ("Initial input (1:1 without input)", "source"),
    ("16:9", "16:9"),
    ("9:16", "9:16"),
    ("1:1", "1:1"),
    ("4:3", "4:3"),
    ("3:4", "3:4"),
    ("3:2", "3:2"),
    ("2:3", "2:3"),
    ("21:9", "21:9"),
]
INPUT_PREVIOUS = "previous"
INPUT_ORIGINAL = "original"
INPUT_SLOT_PREFIX = "input:"
INPUT_TRANSFORM_MEDIA = "media"
INPUT_TRANSFORM_LAST_FRAME = "last_frame"
INPUT_TRANSFORM_CHOICES = [
    ("Use media as-is", INPUT_TRANSFORM_MEDIA),
    ("Extract last video frame", INPUT_TRANSFORM_LAST_FRAME),
]

RUNTIME_ATTACHMENT_KEYS = {
    "audio_guide",
    "audio_guide2",
    "audio_source",
    "custom_guide",
    "image_guide",
    "image_mask",
    "image_start",
    "image_end",
    "image_refs",
    "replace_voice_sample",
    "replace_voice_sample2",
    "video_guide",
    "video_guide2",
    "video_mask",
    "video_source",
}
RUNTIME_TASK_KEYS = {
    "_api",
    "client_id",
    "priority",
}


def normalize_resolution(value: Any) -> str:
    match = re.search(r"(?<!\d)(\d+)\s*[xX]\s*(\d+)(?!\d)", str(value or ""))
    if match is None:
        return ""
    width, height = int(match.group(1)), int(match.group(2))
    return f"{width}x{height}" if width > 0 and height > 0 else ""


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items() if not callable(item)}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value if not callable(item)]
    return str(value)


def clean_settings(settings: dict[str, Any] | None) -> dict[str, Any]:
    cleaned = _json_safe(copy.deepcopy(settings or {}))
    if not isinstance(cleaned, dict):
        return {}
    for key in RUNTIME_TASK_KEYS | RUNTIME_ATTACHMENT_KEYS:
        cleaned.pop(key, None)
    if "resolution" in cleaned:
        normalized_resolution = normalize_resolution(cleaned.get("resolution"))
        if normalized_resolution:
            cleaned["resolution"] = normalized_resolution
    return cleaned


def new_workflow(name: str = "New workflow") -> dict[str, Any]:
    return {
        "version": 11,
        "id": uuid.uuid4().hex,
        "name": str(name or "New workflow").strip(),
        "slug": "",
        "aspect_ratio": "source",
        "unload_on_model_change": False,
        "unload_around_heavy_processing": False,
        "anchor_images": [],
        "input_slots": [],
        "selected_stage_id": "",
        "stages": [],
    }


def normalize_aspect_ratio(value: Any) -> str:
    text = str(value or "source").strip().lower()
    available = {choice[1] for choice in ASPECT_RATIO_CHOICES}
    return text if text in available else "source"


def normalize_workflow(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return new_workflow()
    workflow = new_workflow(str(value.get("name") or "New workflow"))
    workflow["slug"] = str(value.get("slug") or "").strip()
    workflow_key = workflow["slug"] or workflow["name"].casefold()
    workflow["id"] = str(value.get("id") or uuid.uuid5(uuid.NAMESPACE_URL, f"wan2gp-wanflow-ui:{workflow_key}").hex)
    workflow["aspect_ratio"] = normalize_aspect_ratio(value.get("aspect_ratio"))
    workflow["unload_on_model_change"] = bool(value.get("unload_on_model_change", False))
    workflow["unload_around_heavy_processing"] = bool(value.get("unload_around_heavy_processing", False))
    raw_anchors = value.get("anchor_images") or []
    if not isinstance(raw_anchors, (list, tuple)):
        raw_anchors = [raw_anchors]
    workflow["anchor_images"] = [str(path).strip() for path in raw_anchors if str(path or "").strip()]
    workflow["input_slots"] = normalize_input_slots(value.get("input_slots"))
    input_slot_ids = {slot["id"] for slot in workflow["input_slots"]}
    stages = []
    for index, raw_stage in enumerate(value.get("stages") or [], start=1):
        if not isinstance(raw_stage, dict):
            continue
        settings = clean_settings(raw_stage.get("settings"))
        stage_id = str(raw_stage.get("id") or uuid.uuid4().hex)
        raw_stage_type = str(raw_stage.get("stage_type") or "generation").strip()
        stage_type = STAGE_TYPE_ALIASES.get(raw_stage_type, raw_stage_type)
        if stage_type not in STAGE_TYPES:
            stage_type = "generation"
        input_source = str(raw_stage.get("input_source") or INPUT_PREVIOUS).strip()
        if (
            input_source not in {INPUT_PREVIOUS, INPUT_ORIGINAL}
            and not input_source.startswith("stage:")
            and not input_source.startswith(INPUT_SLOT_PREFIX)
        ):
            input_source = INPUT_PREVIOUS
        input_transform = str(raw_stage.get("input_transform") or INPUT_TRANSFORM_MEDIA).strip()
        if input_transform not in {INPUT_TRANSFORM_MEDIA, INPUT_TRANSFORM_LAST_FRAME}:
            input_transform = INPUT_TRANSFORM_MEDIA
        raw_concat_sources = raw_stage.get("concat_sources") or []
        if not isinstance(raw_concat_sources, (list, tuple)):
            raw_concat_sources = [raw_concat_sources]
        raw_references = raw_stage.get("reference_images")
        if raw_references is None:
            raw_references = [raw_stage.get("reference_image")] if raw_stage.get("reference_image") else []
        elif not isinstance(raw_references, (list, tuple)):
            raw_references = [raw_references]
        raw_reference_sources = raw_stage.get("reference_sources") or []
        if not isinstance(raw_reference_sources, (list, tuple)):
            raw_reference_sources = [raw_reference_sources]
        raw_input_references = raw_stage.get("input_references") or []
        if not isinstance(raw_input_references, (list, tuple)):
            raw_input_references = [raw_input_references]
        processor_settings = _json_safe(raw_stage.get("processor_settings") or {})
        if not isinstance(processor_settings, dict):
            processor_settings = {}
        if raw_stage_type == "rife":
            processor_settings.setdefault("temporal_upsampling", "rife4")
        elif raw_stage_type in {"mmaudio", "prismaudio"}:
            processor_settings.setdefault("method", raw_stage_type)
        if stage_type == "ai_analysis":
            ai_model = normalize_ai_model(processor_settings.get("model"))
            processor_settings = {
                "model": ai_model,
                "instruction": str(processor_settings.get("instruction") or "").strip(),
                "image_sources": normalize_ai_image_sources(
                    ai_model,
                    processor_settings.get("image_sources"),
                ),
                "max_tokens": max(32, min(1024, _safe_int(processor_settings.get("max_tokens"), 192))),
            }
        stages.append(
            {
                "id": stage_id,
                "name": str(raw_stage.get("name") or f"Stage {index}").strip(),
                "stage_type": stage_type,
                "output_name": str(raw_stage.get("output_name") or f"output_{index}").strip(),
                "input_source": input_source,
                "input_transform": input_transform,
                "concat_sources": [str(source).strip() for source in raw_concat_sources if str(source or "").strip()],
                "processor_settings": processor_settings,
                "enabled": bool(raw_stage.get("enabled", True)),
                "source_mode": normalize_source_mode(raw_stage.get("source_mode")),
                "resolution_mode": "auto" if raw_stage.get("resolution_mode") == "auto" else "fixed",
                "resolution_tier": normalize_resolution_tier(raw_stage.get("resolution_tier"), settings.get("resolution")),
                "anchor_mode": normalize_anchor_mode(
                    raw_stage.get("anchor_mode"),
                    default="override" if raw_references else "inherit",
                ),
                "reference_images": [str(path).strip() for path in raw_references if str(path or "").strip()],
                "reference_sources": [
                    source for value in raw_reference_sources
                    if (source := normalize_reference_source(value))
                ],
                "input_references": list(dict.fromkeys(
                    source for value in raw_input_references
                    if (source := normalize_input_reference(value)) and source[len(INPUT_SLOT_PREFIX):] in input_slot_ids
                )),
                "end_frame_source": normalize_frame_source(raw_stage.get("end_frame_source")),
                "end_frame_images": normalize_media_paths(raw_stage.get("end_frame_images")),
                "injected_frame_source": normalize_frame_source(raw_stage.get("injected_frame_source")),
                "injected_frame_images": normalize_media_paths(raw_stage.get("injected_frame_images")),
                "injected_frame_positions": str(raw_stage.get("injected_frame_positions") or "1").strip() or "1",
                "settings": settings,
            }
        )
    workflow["stages"] = stages
    selected_id = str(value.get("selected_stage_id") or "")
    workflow["selected_stage_id"] = selected_id if any(stage["id"] == selected_id for stage in stages) else (stages[0]["id"] if stages else "")
    return workflow


def normalize_input_slots(value: Any) -> list[dict[str, str]]:
    raw_slots = value if isinstance(value, (list, tuple)) else []
    slots: list[dict[str, str]] = []
    used_ids: set[str] = set()
    used_names: set[str] = set()
    for index, raw_slot in enumerate(raw_slots, start=1):
        if isinstance(raw_slot, dict):
            name = str(raw_slot.get("name") or "").strip()
            slot_id = str(raw_slot.get("id") or "").strip()
        else:
            name = str(raw_slot or "").strip()
            slot_id = ""
        name = name or f"Input {index}"
        base_name = name
        suffix = 2
        while name.casefold() in used_names:
            name = f"{base_name} {suffix}"
            suffix += 1
        if not slot_id or slot_id in used_ids:
            slot_id = uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"wan2gp-workflow-input:{index}:{name.casefold()}",
            ).hex
            while slot_id in used_ids:
                slot_id = uuid.uuid4().hex
        used_ids.add(slot_id)
        used_names.add(name.casefold())
        slots.append({"id": slot_id, "name": name})
    return slots


def normalize_input_reference(value: Any) -> str:
    normalized = str(value or "").strip()
    if normalized.startswith(INPUT_SLOT_PREFIX) and normalized[len(INPUT_SLOT_PREFIX):]:
        return normalized
    return ""


def is_missing_transformer_error(value: Any) -> bool:
    values = value if isinstance(value, (list, tuple)) else [value]
    return any("no transformer found" in str(item or "").casefold() for item in values)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_ai_model(value: Any) -> int:
    model = _safe_int(value, 3)
    return model if model in AI_MODEL_INFO else 3


def ai_model_supports_multiple_images(value: Any) -> bool:
    return bool(AI_MODEL_INFO[normalize_ai_model(value)]["multiple_images"])


def normalize_ai_image_sources(model_choice: Any, values: Any) -> list[str]:
    if values is None:
        raw_values = []
    elif isinstance(values, (list, tuple)):
        raw_values = values
    else:
        raw_values = [values]
    sources = list(dict.fromkeys(
        source for value in raw_values
        if (source := normalize_ai_image_source(value))
    ))
    return sources if ai_model_supports_multiple_images(model_choice) else sources[:1]


def normalize_ai_image_source(value: Any) -> str:
    normalized = str(value or "").strip()
    if normalized in {INPUT_PREVIOUS, INPUT_ORIGINAL}:
        return normalized
    if normalized.startswith(INPUT_SLOT_PREFIX) and normalized[len(INPUT_SLOT_PREFIX):]:
        return normalized
    return normalize_reference_source(normalized)


def validate_variable_name(value: Any) -> str:
    name = str(value or "").strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError("Variable names must start with a letter or underscore and contain only letters, numbers, and underscores.")
    return name


def substitute_variables(value: Any, variables: dict[str, Any] | None) -> str:
    text = str(value or "")
    try:
        return string.Template(text).substitute({
            str(name): str(content)
            for name, content in dict(variables or {}).items()
        })
    except KeyError as exc:
        raise ValueError(f'Variable "${exc.args[0]}" is not available at this stage.') from exc
    except ValueError as exc:
        raise ValueError(f"Invalid variable expression: {exc}") from exc


def add_runtime_input_slots(
    workflow: dict[str, Any],
    runtime_paths: dict[str, str] | None,
    paths: list[str],
    selected_index: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized = normalize_workflow(workflow)
    slots = normalized["input_slots"]
    slot_ids = {slot["id"] for slot in slots}
    previous = {
        str(slot_id): str(path)
        for slot_id, path in dict(runtime_paths or {}).items()
        if str(slot_id) in slot_ids and str(path or "").strip()
    }
    clean_paths = list(dict.fromkeys(str(path).strip() for path in paths if str(path or "").strip()))
    previous_by_path = {path: slot_id for slot_id, path in previous.items()}
    assigned: dict[str, str] = {}
    used_slots: set[str] = set()
    available_slots = [slot["id"] for slot in slots if slot["id"] not in previous]

    for path in clean_paths:
        slot_id = previous_by_path.get(path)
        if slot_id in used_slots:
            slot_id = None
        if not slot_id:
            slot_id = next((value for value in available_slots if value not in used_slots), None)
        if not slot_id:
            name = unique_input_slot_name(slots, Path(path).stem or f"Input {len(slots) + 1}")
            slot_id = uuid.uuid4().hex
            slots.append({"id": slot_id, "name": name})
        assigned[slot_id] = path
        used_slots.add(slot_id)

    selected_slot_id = ""
    if isinstance(selected_index, int) and 0 <= selected_index < len(clean_paths):
        selected_path = clean_paths[selected_index]
        selected_slot_id = next(
            (slot_id for slot_id, path in assigned.items() if path == selected_path),
            "",
        )
    return normalized, {"paths": assigned, "selected_slot_id": selected_slot_id}


def unique_input_slot_name(slots: list[dict[str, Any]], requested: Any) -> str:
    base = str(requested or "Input").strip() or "Input"
    used = {str(slot.get("name") or "").casefold() for slot in slots}
    if base.casefold() not in used:
        return base
    index = 2
    while f"{base} {index}".casefold() in used:
        index += 1
    return f"{base} {index}"


def normalize_source_mode(value: Any) -> str:
    valid = {choice[1] for choice in SOURCE_MODE_CHOICES}
    normalized = str(value or "auto").strip()
    return normalized if normalized in valid else "auto"


def normalize_anchor_mode(value: Any, default: str = "inherit") -> str:
    normalized = str(value or default).strip().lower()
    if normalized in {"inherit", "override"}:
        return normalized
    if normalized.startswith("stage:") and normalized[6:]:
        return normalized
    return default


def normalize_frame_source(value: Any) -> str:
    normalized = str(value or "none").strip().lower()
    if normalized in {"none", "upload"}:
        return normalized
    if normalized.startswith("stage:") and normalized[6:]:
        return normalized
    if normalized.startswith("last_frame:") and normalized[11:]:
        return normalized
    return "none"


def normalize_reference_source(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized.startswith("stage:") and normalized[6:]:
        return normalized
    if normalized.startswith("last_frame:") and normalized[11:]:
        return normalized
    if normalized.startswith("video:") and normalized[6:]:
        return normalized
    if normalized.startswith("frame:"):
        parts = normalized.split(":")
        if len(parts) == 3 and parts[1] and parts[2].isdigit() and int(parts[2]) >= 1:
            return normalized
    return ""


def normalize_stage_reference_sources(
    values: Any,
    prior_output_kinds: dict[str, str],
    *,
    allow_image_references: bool,
    allow_video_references: bool,
) -> list[str]:
    values = values if isinstance(values, (list, tuple)) else ([values] if values else [])
    output_kinds = {str(stage_id).casefold(): str(kind) for stage_id, kind in prior_output_kinds.items()}
    normalized_sources = []
    for value in values:
        source = normalize_reference_source(value)
        if not source:
            continue
        if source.startswith("stage:"):
            dependency_id, required_kind, allowed = source[6:], "image", allow_image_references
        elif source.startswith("last_frame:"):
            dependency_id, required_kind, allowed = source[11:], "video", allow_image_references
        elif source.startswith("frame:"):
            dependency_id, required_kind, allowed = source.split(":", 2)[1], "video", allow_image_references
        elif source.startswith("video:"):
            dependency_id, required_kind, allowed = source[6:], "video", allow_video_references
        else:
            continue
        if allowed and output_kinds.get(dependency_id.casefold()) == required_kind:
            normalized_sources.append(source)
    return list(dict.fromkeys(normalized_sources))


def normalize_media_paths(value: Any) -> list[str]:
    values = value if isinstance(value, (list, tuple)) else ([value] if value else [])
    return [str(path).strip() for path in values if str(path or "").strip()]


def model_supports_end_frame(model_def: dict[str, Any] | None) -> bool:
    return "E" in str((model_def or {}).get("image_prompt_types_allowed") or "")


def model_supports_injected_frames(model_def: dict[str, Any] | None) -> bool:
    model_def = model_def or {}
    if bool(model_def.get("custom_frames_injection", False)):
        return True
    choices = ((model_def.get("image_ref_choices") or {}).get("choices") or [])
    return any("F" in str(value or "") for _label, value in choices)


def injected_frame_prompt_letters(model_def: dict[str, Any] | None) -> str:
    model_def = model_def or {}
    for choice_def_key in ("guide_custom_choices", "image_ref_choices"):
        choices = ((model_def.get(choice_def_key) or {}).get("choices") or [])
        for choice in choices:
            value = choice[1] if isinstance(choice, (list, tuple)) and len(choice) > 1 else choice
            letters = str(value or "")
            if "F" in letters:
                return letters
    return "FI"


def bind_end_frame(settings: dict[str, Any], frame_path: str, model_def: dict[str, Any] | None) -> dict[str, Any]:
    bound = copy.deepcopy(settings)
    bound.pop("image_end", None)
    bound["image_prompt_type"] = _remove_letters(bound.get("image_prompt_type"), "E")
    if not frame_path:
        return bound
    if not model_supports_end_frame(model_def):
        raise ValueError("This stage model does not support an end frame.")
    if media_kind(frame_path) != "image" or not Path(frame_path).is_file():
        raise ValueError("The selected end frame is missing or is not a valid image.")
    bound["image_end"] = frame_path
    bound["image_prompt_type"] = _add_letter(bound.get("image_prompt_type"), "E")
    return bound


def bind_injected_frames(
    settings: dict[str, Any],
    frame_paths: Any,
    positions: Any,
    model_def: dict[str, Any] | None,
) -> dict[str, Any]:
    bound = copy.deepcopy(settings)
    paths = normalize_media_paths(frame_paths)
    injection_letters = injected_frame_prompt_letters(model_def)
    bound["video_prompt_type"] = _remove_letters(bound.get("video_prompt_type"), injection_letters)
    bound.pop("frames_positions", None)
    if not paths:
        return bound
    if not model_supports_injected_frames(model_def):
        raise ValueError("This stage model does not support positioned frame injection.")
    for path in paths:
        if media_kind(path) != "image" or not Path(path).is_file():
            raise ValueError("An injected frame is missing or is not a valid image.")
    position_tokens = [token for token in re.split(r"[\s,]+", str(positions or "").strip()) if token]
    if len(position_tokens) != len(paths):
        raise ValueError("Provide exactly one frame position for each injected frame.")
    if any(token.lower() not in {"l", "x"} and (not token.isdigit() or int(token) < 1) for token in position_tokens):
        raise ValueError("Injected frame positions must be positive frame numbers, L, or X.")
    existing_refs = normalize_media_paths(bound.get("image_refs"))
    bound["image_refs"] = list(dict.fromkeys(paths + existing_refs))
    bound["frames_positions"] = " ".join(position_tokens)
    for letter in injection_letters:
        bound["video_prompt_type"] = _add_letter(bound.get("video_prompt_type"), letter)
    return bound


def normalize_resolution_tier(value: Any, fallback_resolution: Any = "") -> str:
    normalized = str(value or "").strip().lower()
    if normalized in resolution_api.GROUP_THRESHOLDS:
        return normalized
    fallback = normalize_resolution(fallback_resolution)
    if fallback:
        return resolution_api.categorize_resolution(fallback)
    return "720p"


def available_resolution_tiers(resolution_choices: Any) -> list[str]:
    available = {
        resolution_api.categorize_resolution(str(value))
        for _label, value in list(resolution_choices or [])
        if resolution_api.is_resolution_value(str(value))
    }
    return [tier for tier in reversed(resolution_api.GROUP_THRESHOLDS) if tier in available]


def resolution_for_tier(
    resolution_choices: Any,
    tier: Any,
    aspect_ratio: float,
    block_size: int = 16,
) -> str:
    normalized_tier = normalize_resolution_tier(tier)
    candidates = [
        (str(label), str(value))
        for label, value in list(resolution_choices or [])
        if resolution_api.is_resolution_value(str(value))
        and resolution_api.categorize_resolution(str(value)) == normalized_tier
    ]
    if not candidates:
        raise ValueError(f"The selected model does not support the {normalized_tier} quality tier.")
    ratio = float(aspect_ratio or 0)

    def score(choice: tuple[str, str]) -> tuple[float, int]:
        width, height = resolution_api.parse_resolution(choice[1])
        ratio_score = abs(math.log((width / height) / ratio)) if ratio > 0 else 0.0
        area_score = abs(width * height - resolution_api.GROUP_THRESHOLDS[normalized_tier])
        return ratio_score, area_score

    selected = min(candidates, key=score)[1]
    selected_width, selected_height = resolution_api.parse_resolution(selected)
    if ratio > 0 and abs(math.log((selected_width / selected_height) / ratio)) <= 0.03:
        return selected
    return resolution_for_aspect_ratio(selected, ratio, block_size)


def make_stage(settings: dict[str, Any], name: str = "") -> dict[str, Any]:
    cleaned = clean_settings(settings)
    model_type = str(cleaned.get("model_type") or cleaned.get("base_model_type") or "Model")
    return {
        "id": uuid.uuid4().hex,
        "name": str(name or model_type).strip(),
        "stage_type": "generation",
        "output_name": "output",
        "input_source": INPUT_PREVIOUS,
        "input_transform": INPUT_TRANSFORM_MEDIA,
        "concat_sources": [],
        "processor_settings": {},
        "enabled": True,
        "source_mode": "auto",
        "resolution_mode": "fixed",
        "resolution_tier": normalize_resolution_tier("", cleaned.get("resolution")),
        "anchor_mode": "inherit",
        "reference_images": [],
        "reference_sources": [],
        "input_references": [],
        "end_frame_source": "none",
        "end_frame_images": [],
        "injected_frame_source": "none",
        "injected_frame_images": [],
        "injected_frame_positions": "1",
        "settings": cleaned,
    }


def make_processing_stage(stage_type: str, name: str = "", processor_settings: dict[str, Any] | None = None) -> dict[str, Any]:
    raw_stage_type = str(stage_type or "").strip()
    stage_type = STAGE_TYPE_ALIASES.get(raw_stage_type, raw_stage_type)
    if stage_type not in STAGE_TYPES - {"generation"}:
        raise ValueError(f"Unsupported processing stage type: {stage_type}")
    normalized_processor_settings = _json_safe(processor_settings or {})
    if not isinstance(normalized_processor_settings, dict):
        normalized_processor_settings = {}
    if raw_stage_type == "rife":
        normalized_processor_settings.setdefault("temporal_upsampling", "rife4")
    elif raw_stage_type in {"mmaudio", "prismaudio"}:
        normalized_processor_settings.setdefault("method", raw_stage_type)
    return {
        "id": uuid.uuid4().hex,
        "name": str(name or STAGE_TYPE_LABELS[stage_type]).strip(),
        "stage_type": stage_type,
        "output_name": "output",
        "input_source": INPUT_PREVIOUS,
        "input_transform": INPUT_TRANSFORM_MEDIA,
        "concat_sources": [],
        "processor_settings": normalized_processor_settings,
        "enabled": True,
        "source_mode": "none",
        "resolution_mode": "fixed",
        "resolution_tier": "720p",
        "anchor_mode": "inherit",
        "reference_images": [],
        "reference_sources": [],
        "input_references": [],
        "end_frame_source": "none",
        "end_frame_images": [],
        "injected_frame_source": "none",
        "injected_frame_images": [],
        "injected_frame_positions": "1",
        "settings": {},
    }


def selected_stage(workflow: dict[str, Any]) -> dict[str, Any] | None:
    selected_id = str(workflow.get("selected_stage_id") or "")
    return next((stage for stage in workflow.get("stages", []) if stage.get("id") == selected_id), None)


def media_kind(path: str | os.PathLike[str] | None) -> str:
    suffix = Path(os.fspath(path or "")).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if suffix in AUDIO_EXTENSIONS:
        return "audio"
    return "unknown"


def _add_letter(value: Any, letter: str) -> str:
    text = str(value or "")
    return text if letter in text else text + letter


def _remove_letters(value: Any, letters: str) -> str:
    text = str(value or "")
    return "".join(char for char in text if char not in letters)


def resolve_source_mode(source_mode: str, source_path: str, model_def: dict[str, Any] | None) -> str:
    mode = normalize_source_mode(source_mode)
    if mode != "auto":
        return mode
    kind = media_kind(source_path)
    model_def = model_def or {}
    allowed = str(model_def.get("image_prompt_types_allowed") or "")
    capabilities = ((model_def.get("metadata") or {}).get("capabilities") or {})
    if kind == "image":
        if "S" in allowed:
            return "start_image"
        if capabilities.get("image_to_image") or capabilities.get("reference_images"):
            return "reference_image"
    if kind == "video":
        if "V" in allowed:
            return "continue_video"
        if model_supports_reference_videos(model_def):
            return "reference_video"
        if capabilities.get("video_to_video"):
            return "control_video"
    raise ValueError("This stage model cannot automatically accept the received file type.")


def model_supports_text_only(settings: dict[str, Any] | None, model_def: dict[str, Any] | None) -> bool:
    settings = settings or {}
    model_def = model_def or {}
    if bool(model_def.get("one_image_ref_needed", False) or model_def.get("at_least_one_image_ref_needed", False)):
        return False
    custom_guide = model_def.get("custom_guide")
    if isinstance(custom_guide, dict) and custom_guide.get("required", False):
        return False
    if expected_output_kind(settings, model_def) == "image":
        capabilities = ((model_def.get("metadata") or {}).get("capabilities") or {})
        return bool(capabilities.get("text_to_image", True))
    return "T" in str(model_def.get("image_prompt_types_allowed") or "")


def _choice_letters(model_def: dict[str, Any], key: str, fallback: str = "") -> str:
    definition = model_def.get(key)
    if not isinstance(definition, dict):
        return fallback
    return str(definition.get("letters_filter") or fallback)


def _choice_values(model_def: dict[str, Any] | None, key: str) -> list[str]:
    definition = (model_def or {}).get(key)
    if not isinstance(definition, dict):
        return []
    return [
        str(choice[1] if isinstance(choice, (list, tuple)) and len(choice) > 1 else choice or "")
        for choice in definition.get("choices") or []
    ]


def model_supports_reference_videos(model_def: dict[str, Any] | None) -> bool:
    return any("V" in value and "-" in value for value in _choice_values(model_def, "guide_custom_choices"))


def model_reference_video_limit(model_def: dict[str, Any] | None) -> int:
    choices = _choice_values(model_def, "guide_custom_choices")
    return 2 if any("V" in value and "+" in value and "-" in value for value in choices) else 1


def _replace_guide_choice(value: Any, model_def: dict[str, Any] | None, choice: str) -> str:
    letters = _choice_letters(model_def or {}, "guide_custom_choices")
    updated = _remove_letters(value, letters)
    for letter in choice:
        updated = _add_letter(updated, letter)
    return updated


def bind_source(settings: dict[str, Any], source_path: str, source_mode: str, model_def: dict[str, Any] | None) -> tuple[dict[str, Any], str]:
    bound = copy.deepcopy(settings)
    if normalize_source_mode(source_mode) == "none":
        model_def = model_def or {}
        if not model_supports_text_only(bound, model_def):
            raise ValueError("This stage model requires a media input and cannot run in text-only mode.")
        for key in ("image_start", "video_source", "video_guide", "video_guide2", "image_guide", "video_mask", "image_mask"):
            bound.pop(key, None)
        image_prompt_type = _remove_letters(bound.get("image_prompt_type"), "SVL")
        if not bound.get("image_end"):
            image_prompt_type = _remove_letters(image_prompt_type, "E")
        bound["image_prompt_type"] = image_prompt_type
        video_prompt_type = str(bound.get("video_prompt_type") or "")
        if not bound.get("image_refs"):
            image_ref_letters = _choice_letters(model_def, "image_ref_choices", "KFI")
            video_prompt_type = _remove_letters(video_prompt_type, image_ref_letters)
        guide_letters = _choice_letters(model_def, "guide_custom_choices")
        if guide_letters:
            video_prompt_type = _remove_letters(video_prompt_type, guide_letters)
        bound["video_prompt_type"] = video_prompt_type
        return bound, "none"
    if not source_path:
        raise ValueError("The input file required to connect this stage is missing.")
    mode = resolve_source_mode(source_mode, source_path, model_def)
    kind = media_kind(source_path)
    allowed = str((model_def or {}).get("image_prompt_types_allowed") or "")

    if mode == "start_image":
        if kind != "image":
            raise ValueError("The stage expects a start image, but received another file type.")
        if "S" not in allowed:
            raise ValueError("This stage model does not support a start image.")
        bound["image_start"] = source_path
        bound.pop("video_source", None)
        bound["image_prompt_type"] = _add_letter(_remove_letters(bound.get("image_prompt_type"), "VL"), "S")
    elif mode == "continue_video":
        if kind != "video":
            raise ValueError("The stage expects a video to continue.")
        if "V" not in allowed:
            raise ValueError("This stage model does not support video continuation.")
        bound["video_source"] = source_path
        bound.pop("image_start", None)
        bound["image_prompt_type"] = _add_letter(_remove_letters(bound.get("image_prompt_type"), "SL"), "V")
    elif mode == "reference_image":
        if kind != "image":
            raise ValueError("The stage expects a reference image.")
        existing_refs = bound.get("image_refs") or []
        if isinstance(existing_refs, str):
            existing_refs = [existing_refs]
        bound["image_refs"] = [source_path] + [str(path) for path in existing_refs if str(path) and str(path) != source_path]
        bound.pop("image_start", None)
        bound.pop("video_source", None)
        bound["video_prompt_type"] = _add_letter(bound.get("video_prompt_type"), "I")
    elif mode == "reference_video":
        if kind != "video":
            raise ValueError("The stage expects a reference video.")
        if not model_supports_reference_videos(model_def):
            raise ValueError("This stage model does not support reference videos.")
        bound["video_guide"] = source_path
        bound.pop("video_guide2", None)
        bound.pop("image_start", None)
        bound.pop("video_source", None)
        bound["video_prompt_type"] = _replace_guide_choice(bound.get("video_prompt_type"), model_def, "V-")
    elif mode == "control_video":
        if kind != "video":
            raise ValueError("The stage expects a control video.")
        bound["video_guide"] = source_path
        bound.pop("image_start", None)
        bound.pop("video_source", None)
        bound.pop("video_guide2", None)
        bound["video_prompt_type"] = _replace_guide_choice(bound.get("video_prompt_type"), model_def, "V")
    else:
        raise ValueError(f"Unsupported input connection: {mode}")
    return bound, mode


def add_stage_references(settings: dict[str, Any], reference_paths: Any, model_def: dict[str, Any] | None) -> dict[str, Any]:
    bound = copy.deepcopy(settings)
    if not isinstance(reference_paths, (list, tuple)):
        reference_paths = [reference_paths] if reference_paths else []
    normalized_paths = [str(path or "").strip() for path in reference_paths if str(path or "").strip()]
    if not normalized_paths:
        return bound
    for reference_path in normalized_paths:
        if media_kind(reference_path) != "image" or not Path(reference_path).is_file():
            raise ValueError("A stage reference is missing or is not a valid image.")
    existing_refs = bound.get("image_refs") or []
    if isinstance(existing_refs, str):
        existing_refs = [existing_refs]
    combined = [str(path) for path in existing_refs if str(path)] + normalized_paths
    bound["image_refs"] = list(dict.fromkeys(combined))
    bound["video_prompt_type"] = _add_letter(bound.get("video_prompt_type"), "I")
    if bool((model_def or {}).get("svi2pro", False)):
        bound["video_prompt_type"] = _add_letter(bound["video_prompt_type"], "K")
    return bound


def add_stage_video_references(settings: dict[str, Any], reference_paths: Any, model_def: dict[str, Any] | None) -> dict[str, Any]:
    bound = copy.deepcopy(settings)
    if not isinstance(reference_paths, (list, tuple)):
        reference_paths = [reference_paths] if reference_paths else []
    normalized_paths = [str(path or "").strip() for path in reference_paths if str(path or "").strip()]
    if not normalized_paths:
        return bound
    if not model_supports_reference_videos(model_def):
        raise ValueError("This stage model does not support reference videos.")
    for reference_path in normalized_paths:
        if media_kind(reference_path) != "video" or not Path(reference_path).is_file():
            raise ValueError("A stage reference is missing or is not a valid video.")
    existing = []
    current_mode = str(bound.get("video_prompt_type") or "")
    if "V" in current_mode and "-" in current_mode and bound.get("video_guide"):
        existing.append(str(bound["video_guide"]))
        if "+" in current_mode and bound.get("video_guide2"):
            existing.append(str(bound["video_guide2"]))
    combined = list(dict.fromkeys(existing + normalized_paths))
    limit = model_reference_video_limit(model_def)
    if len(combined) > limit:
        raise ValueError(f"This stage model accepts at most {limit} reference video{'s' if limit != 1 else ''}.")
    bound["video_guide"] = combined[0]
    if len(combined) > 1:
        bound["video_guide2"] = combined[1]
        choice = "V+-"
    else:
        bound.pop("video_guide2", None)
        choice = "V-"
    bound["video_prompt_type"] = _replace_guide_choice(bound.get("video_prompt_type"), model_def, choice)
    return bound


def automatic_resolution(source_path: str, current_resolution: Any, model_def: dict[str, Any] | None) -> str:
    normalized_resolution = normalize_resolution(current_resolution)
    if media_kind(source_path) != "image" or not Path(source_path).is_file():
        return normalized_resolution
    try:
        canvas_width, canvas_height = [int(part) for part in normalized_resolution.split("x", 1)]
        if canvas_width <= 0 or canvas_height <= 0:
            return normalized_resolution
    except (TypeError, ValueError):
        return normalized_resolution
    with Image.open(source_path) as image:
        image_width, image_height = image.size
    if image_width <= 0 or image_height <= 0:
        return normalized_resolution
    block_size = max(1, int((model_def or {}).get("vae_block_size", 16) or 16))
    scale = math.sqrt((canvas_width * canvas_height) / (image_width * image_height))
    width = max(block_size, round(image_width * scale / block_size) * block_size)
    height = max(block_size, round(image_height * scale / block_size) * block_size)
    return f"{width}x{height}"


def media_aspect_ratio(path: str | os.PathLike[str]) -> float:
    normalized_path = os.fspath(path)
    kind = media_kind(normalized_path)
    if kind == "image":
        with Image.open(normalized_path) as image:
            width, height = image.size
    elif kind == "video":
        from shared.utils.utils import get_video_info

        _fps, width, height, _frames = get_video_info(normalized_path)
    else:
        raise ValueError("The workflow aspect ratio can only be derived from an image or video input.")
    width, height = int(width or 0), int(height or 0)
    if width <= 0 or height <= 0:
        raise ValueError("Could not determine the workflow input aspect ratio.")
    return width / height


def workflow_aspect_ratio(value: Any, source_path: str | os.PathLike[str] | None = None) -> float:
    normalized = normalize_aspect_ratio(value)
    if normalized == "source":
        if not source_path:
            return 1.0
        return media_aspect_ratio(source_path)
    width, height = normalized.split(":", 1)
    return int(width) / int(height)


def resolution_for_aspect_ratio(resolution: Any, aspect_ratio: float, block_size: int = 16) -> str:
    normalized = normalize_resolution(resolution)
    if not normalized:
        return ""
    width, height = [int(part) for part in normalized.split("x", 1)]
    ratio = float(aspect_ratio or 0)
    if width <= 0 or height <= 0 or ratio <= 0:
        return normalized
    block = max(1, int(block_size or 1))
    area = width * height
    fraction = Fraction(ratio).limit_denominator(32)
    ratio_width, ratio_height = fraction.numerator, fraction.denominator
    scale_unit = math.lcm(block // math.gcd(block, ratio_width), block // math.gcd(block, ratio_height))
    unit_width, unit_height = ratio_width * scale_unit, ratio_height * scale_unit
    ideal_scale = math.sqrt(area / (unit_width * unit_height))
    candidates = {max(1, math.floor(ideal_scale)), max(1, math.ceil(ideal_scale))}
    scale = min(candidates, key=lambda candidate: abs(unit_width * unit_height * candidate * candidate - area))
    target_width, target_height = unit_width * scale, unit_height * scale
    return f"{target_width}x{target_height}"


def expected_output_kind(settings: dict[str, Any], model_def: dict[str, Any] | None) -> str:
    if bool((model_def or {}).get("audio_only", False)):
        return "audio"
    if int(settings.get("image_mode") or 0) > 0:
        return "image"
    outputs = list((((model_def or {}).get("metadata") or {}).get("main_output") or []))
    if len(outputs) == 1 and outputs[0] in {"image", "video", "audio"}:
        return str(outputs[0])
    return "video"


def select_generated_path(result: Any, expected_kind: str = "") -> str:
    candidates: list[str] = []
    for artifact in list(getattr(result, "artifacts", ()) or ()):
        path = str(getattr(artifact, "path", "") or "").strip()
        if path:
            candidates.append(path)
    for path in list(getattr(result, "generated_files", []) or []):
        path = str(path or "").strip()
        if path and path not in candidates:
            candidates.append(path)
    existing = [path for path in candidates if Path(path).is_file()]
    if expected_kind:
        matching = [path for path in existing if media_kind(path) == expected_kind]
        if matching:
            return matching[-1]
    media = [path for path in existing if media_kind(path) != "unknown"]
    return media[-1] if media else (existing[-1] if existing else "")


def loras_to_text(settings: dict[str, Any]) -> str:
    values = settings.get("activated_loras") or []
    if isinstance(values, str):
        return values
    return "\n".join(str(item) for item in values)


def text_to_loras(value: Any) -> list[str]:
    return [line.strip() for line in str(value or "").splitlines() if line.strip()]


def workflow_for_json(workflow: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_workflow(workflow)
    normalized.pop("selected_stage_id", None)
    return json.loads(json.dumps(normalized, ensure_ascii=True))
