from __future__ import annotations

import copy
from typing import Any

from shared import resolutions as resolution_api
from shared import extra_settings

try:
    from shared.attention import get_override_attention_modes
except Exception:  # pragma: no cover - optional backend availability
    get_override_attention_modes = lambda: []

from .graph_model import GENERATION_NODE_TYPES, node_ports, model_capabilities


STATIC_NODES = [
    ("input_image", "Input Image", "Input", "Provide an image file."),
    ("input_video", "Input Video", "Input", "Provide a video file."),
    ("input_audio", "Input Audio", "Input", "Provide an audio file."),
    ("input_mask_image", "Input Mask Image", "Input", "Provide a static mask image file."),
    ("input_mask_video", "Input Mask Video", "Input", "Provide a video mask file."),
    ("load_media", "Load Media", "Input", "Bind one runtime media upload."),
    ("text", "Text", "Input", "Provide text to a prompt or AI node."),
    ("mask_editor", "Mask Editor", "Input", "Create an image mask with the existing editor."),
    ("magic_mask", "Magic Mask", "Mask", "Generate an image or video mask from object keywords using Wan2GP Magic Mask."),
    ("generate_image", "Generate Image", "Generation", "Text or image-conditioned image generation."),
    ("edit_image", "Edit Image", "Generation", "Image editing node."),
    ("inpaint_image", "Inpaint Image", "Generation", "Image generation with a mask."),
    ("generate_video", "Generate Video", "Generation", "Text, image or video-conditioned video generation."),
    ("edit_video", "Edit Video", "Generation", "Video editing node."),
    ("inpaint_video", "Inpaint Video", "Generation", "Video generation with a mask."),
    ("generate_audio", "Generate Audio", "Generation", "Audio generation node."),
    ("ai_analyze", "AI Analyze", "AI", "Analyze connected images and emit text."),
    ("last_frame", "Last Frame", "Conversion", "Extract the last image from a video."),
    ("extract_frame", "Extract Frame", "Conversion", "Extract a numbered image from a video."),
    ("image_sequence", "Image Sequence", "Conversion", "Encode images as a video."),
    ("video_probe", "Video Probe", "Conversion", "Read media metadata."),
    ("postprocess", "Postprocess", "Postprocess", "Use a dynamically discovered Wan2GP processor."),
    ("ffmpeg_trim", "FFmpeg Trim", "FFmpeg", "Trim a video safely."),
    ("ffmpeg_concat", "FFmpeg Concat", "FFmpeg", "Concatenate videos with normalized streams."),
    ("ffmpeg_mux_audio", "FFmpeg Mux Audio", "FFmpeg", "Replace or add a soundtrack."),
    ("ffmpeg_remove_audio", "FFmpeg Remove Audio", "FFmpeg", "Remove audio from a video."),
    ("ffmpeg_transcode", "FFmpeg Transcode", "FFmpeg", "Transcode media using a safe preset."),
    ("ffmpeg_resize", "FFmpeg Resize", "FFmpeg", "Resize a video."),
    ("ffmpeg_fps", "FFmpeg FPS", "FFmpeg", "Change a video's frame rate."),
    ("ffmpeg_normalize", "FFmpeg Normalize", "FFmpeg", "Normalize video streams for downstream nodes."),
    ("ffmpeg_export", "FFmpeg Export", "FFmpeg", "Copy a media result to a selected output name."),
]

AI_MODELS = [
    {"id": 1, "label": "Florence 2 + Llama 3.2 3B | Lowest VRAM | Single image"},
    {"id": 2, "label": "Florence 2 + Llama Joy 8B | High VRAM | Single image"},
    {"id": 3, "label": "Qwen3.5VL Abliterated 4B | Medium VRAM | Multiple images"},
    {"id": 4, "label": "Qwen3.5VL Abliterated 9B | Highest VRAM | Multiple images"},
]

ASPECT_RATIOS = [
    {"label": "Initial input", "value": "source"},
    *({"label": value, "value": value} for value in ("16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3", "21:9")),
]


def _choices(values: Any) -> list[dict[str, Any]]:
    result = []
    for value in values or []:
        if isinstance(value, dict):
            label = value.get("label", value.get("name", value.get("value", "")))
            item = value.get("value", value.get("id", label))
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            label, item = value[0], value[1]
        else:
            label = item = value
        if str(item or "").strip() or item == "":
            result.append({"label": str(label or item), "value": item})
    return result


def _native_settings(model_def: dict[str, Any]) -> list[dict[str, Any]]:
    """Expose the useful Wan2GP settings without duplicating model rules."""
    result: list[dict[str, Any]] = []
    try:
        definitions = extra_settings.iter_defs(model_def, only_visible=True, guidance_phases=1)
    except Exception:
        definitions = {}
    for key, definition in definitions.items():
        if key in {"resolution", "video_length", "force_fps", "batch_size", "negative_prompt"}:
            continue
        result.append({
            "key": key,
            "label": definition.label,
            "type": definition.type,
            "min": definition.min,
            "max": definition.max,
            "step": definition.step,
            "custom": definition.custom,
            "disabled": key == "sliding_window_size" and bool(model_def.get("sliding_window_size_locked")),
            "default": (model_def.get("sliding_window_defaults") or {}).get("window_default") if key == "sliding_window_size" else None,
        })

    sample_solvers = _choices(model_def.get("sample_solvers"))
    if sample_solvers:
        result.append({"key": "sample_solver", "label": "Sampler / scheduler", "type": "string", "choices": sample_solvers})

    cache_choices = [{"label": "Auto / disabled", "value": ""}]
    for flag, value, label in (
        ("tea_cache", "tea", "Tea Cache"),
        ("mag_cache", "mag", "MAG Cache"),
        ("spectrum_cache", "spectrum", "Spectrum"),
        ("first_block_cache", "first_block", "First Block Cache"),
    ):
        if model_def.get(flag):
            cache_choices.append({"label": label, "value": value})
    if len(cache_choices) > 1:
        result.extend([
            {"key": "skip_steps_cache_type", "label": "Skip steps cache type", "type": "string", "choices": cache_choices},
            {"key": "skip_steps_multiplier", "label": str(model_def.get("skip_steps_multiplier_label") or "Cache threshold"), "type": "number", "choices": _choices(model_def.get("skip_steps_multiplier_choices"))},
            {"key": "skip_steps_start_step_perc", "label": "Skip steps starting moment (%)", "type": "integer", "min": 0, "max": 100, "step": 1},
        ])

    attention_modes = []
    try:
        attention_modes.extend(get_override_attention_modes() or [])
    except Exception:
        pass
    attention_modes.extend(
        key for key, value in (model_def.get("custom_attention_modes") or {}).items()
        if not isinstance(value, dict) or value.get("supported", True)
    )
    attention_choices = [{"label": "Auto / default", "value": ""}]
    seen = set()
    for mode in attention_modes:
        mode = str(mode or "").strip()
        if not mode or mode in seen:
            continue
        seen.add(mode)
        custom = (model_def.get("custom_attention_modes") or {}).get(mode) or {}
        attention_choices.append({"label": str(custom.get("label") or mode), "value": mode})
    if len(attention_choices) > 1:
        result.append({"key": "override_attention", "label": "Override Attention Mode", "type": "string", "choices": attention_choices})
        sparsity = model_def.get("attention_sparsity") or {}
        if "attention_sparsity" not in {item["key"] for item in result} and (sparsity or any((model_def.get("custom_attention_modes") or {}).get(mode, {}).get("supports_sparsity") for mode in seen)):
            result.append({
                "key": "attention_sparsity",
                "label": str(sparsity.get("label") or "Attention sparsity"),
                "type": "number",
                "min": sparsity.get("start", 0.0),
                "max": sparsity.get("end", 4.0),
                "step": sparsity.get("inc", 0.05),
            })
    return result


def _port_list(node: dict[str, Any], model_defs: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    values = []
    for name, port in node_ports(node, model_defs).items():
        values.append({"name": name, **copy.deepcopy(port)})
    return values


def build_catalog(
    model_defs: dict[str, dict[str, Any]] | None = None,
    processes: list[dict[str, Any]] | None = None,
    loras: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    model_defs = model_defs or {}
    loras = loras or {}
    catalog = []
    for node_type, label, category, description in STATIC_NODES:
        sample = {"type": node_type, "params": {}}
        catalog.append({
            "type": node_type,
            "label": label,
            "category": category,
            "description": description,
            "ports": _port_list(sample, model_defs),
        })
    dynamic_processes = []
    for process in processes or []:
        dynamic_processes.append({
            "id": str(process.get("id") or ""),
            "label": str(process.get("label") or process.get("id") or "Processor"),
            "type": str(process.get("type") or ""),
            "media": list(process.get("media") or []),
            "parameters": copy.deepcopy(process.get("parameters") or []),
            "status": str(process.get("status") or "unknown"),
            "reason_disabled": str(process.get("reason_disabled") or ""),
        })
    models = []
    for model_type, model_def in sorted(model_defs.items(), key=lambda item: str(item[0]).casefold()):
        caps = model_capabilities(model_def)
        try:
            resolutions = resolution_api.resolve_resolution_choices(None, model_def, False, model_def.get("vae_block_size", 16))[0]
        except Exception:
            resolutions = []
        tier_order = list(resolution_api.GROUP_THRESHOLDS)
        tiers = sorted(
            {resolution_api.categorize_resolution(str(value)) for _label, value in resolutions},
            key=lambda value: tier_order.index(value) if value in tier_order else len(tier_order),
        )
        models.append({
            "model_type": str(model_type),
            "name": str(model_def.get("name") or model_type),
            "metadata": copy.deepcopy(model_def.get("metadata") or {}),
            "capabilities": caps,
            "svi2pro": bool(model_def.get("svi2pro")),
            "reference_input_label": "Anchor images (each window)" if model_def.get("svi2pro") else "Reference images",
            "lora_supported": not bool(model_def.get("no_lora", False)) and bool(caps.get("lora", True)),
            "loras": list(loras.get(str(model_type), [])),
            "resolution_choices": [{"label": str(label), "value": str(value), "tier": resolution_api.categorize_resolution(str(value))} for label, value in resolutions],
            "resolution_tiers": [{"label": value, "value": value} for value in tiers],
            "native_settings": _native_settings(model_def),
        })
    return {"nodes": catalog, "models": models, "processes": dynamic_processes, "ai_models": copy.deepcopy(AI_MODELS), "aspect_ratios": copy.deepcopy(ASPECT_RATIOS)}
