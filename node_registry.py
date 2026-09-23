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
    ("input_image", "Input Image", "Inputs", "Provide an image file."),
    ("input_video", "Input Video", "Inputs", "Provide a video file."),
    ("input_audio", "Input Audio", "Inputs", "Provide an audio file."),
    ("load_media", "Load Media", "Inputs", "Bind one runtime media upload."),
    ("text", "Text", "Inputs", "Provide text to a prompt or AI node."),
    ("input_mask_image", "Input Mask Image", "Masks", "Provide a static mask image file."),
    ("input_mask_video", "Input Mask Video", "Masks", "Provide a video mask file."),
    ("mask_editor", "Mask Editor", "Masks", "Create an image mask with the existing editor."),
    ("magic_mask", "Magic Mask", "Masks", "Generate an image or video mask from object keywords using Wan2GP Magic Mask."),
    ("generate_image", "Generate Image", "Generation", "Text or image-conditioned image generation."),
    ("edit_image", "Edit Image", "Generation", "Image editing node."),
    ("inpaint_image", "Inpaint Image", "Generation", "Image generation with a mask."),
    ("generate_video", "Generate Video", "Generation", "Text, image or video-conditioned video generation."),
    ("edit_video", "Edit Video", "Generation", "Video editing node."),
    ("inpaint_video", "Inpaint Video", "Generation", "Video generation with a mask."),
    ("generate_audio", "Generate Audio", "Generation", "Audio generation node."),
    ("ai_analyze", "AI Analyze", "AI", "Analyze connected images and emit text."),
    ("prompt_enhancer", "Prompt Enhancer", "AI", "Rewrite a prompt with Wan2GP's model-specific prompt enhancer."),
    ("resolution_config", "Resolution & Aspect", "Configuration", "Override resolution tier and aspect ratio for connected generation nodes."),
    ("lora_stack", "LoRA Stack", "Configuration", "Select multiple model-scoped LoRAs with an individual strength for each one."),
    ("sampling_config", "Sampling / Guidance", "Configuration", "Override common sampling and guidance settings on connected generation nodes."),
    ("attention_config", "Attention & Cache", "Configuration", "Override model-native attention and skip-step cache settings."),
    ("reference_composition", "Reference Composition", "Configuration", "Override the relative size of image references inside the output video."),
    ("last_frame", "Last Frame", "Processing", "Extract the last image from a video."),
    ("extract_frame", "Extract Frame", "Processing", "Extract a numbered image from a video."),
    ("image_sequence", "Image Sequence", "Processing", "Encode images as a video."),
    ("video_probe", "Video Probe", "Processing", "Read media metadata."),
    ("postprocess", "Postprocess", "Processing", "Use a dynamically discovered Wan2GP processor."),
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


def _model_defaults(model_def: dict[str, Any]) -> dict[str, Any]:
    """Expose only generation defaults that are safe to persist in the editor."""
    keys = {
        "num_inference_steps", "steps", "video_length", "cfg_scale", "guidance_scale",
        "sample_solver", "flow_shift", "shift", "denoising_strength", "sliding_window_size",
        "sliding_window_overlap", "skip_steps_cache_type", "skip_steps_multiplier",
        "skip_steps_start_step_perc", "override_attention", "attention_sparsity",
        "image_refs_relative_size",
    }
    source = model_def.get("default_settings") or model_def.get("defaults") or {}
    result = copy.deepcopy(source) if isinstance(source, dict) else {}
    numeric_keys = {
        "num_inference_steps", "steps", "video_length", "cfg_scale", "guidance_scale",
        "flow_shift", "shift", "denoising_strength", "sliding_window_size", "sliding_window_overlap",
        "skip_steps_multiplier", "skip_steps_start_step_perc", "attention_sparsity", "image_refs_relative_size",
    }
    string_keys = {"sample_solver", "skip_steps_cache_type", "override_attention"}

    def valid_default(key: str, value: Any) -> bool:
        if key in numeric_keys:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if key in string_keys:
            return isinstance(value, (str, int, float)) and not isinstance(value, bool)
        return True

    result = {key: value for key, value in result.items() if valid_default(key, value)}
    for key in keys:
        if key in model_def and model_def[key] not in (None, "") and valid_default(key, model_def[key]):
            result[key] = copy.deepcopy(model_def[key])
    return result


def _prompt_enhancer_catalog(model_def: dict[str, Any], resolver=None) -> dict[str, Any]:
    """Expose Wan2GP's native enhancer choices without copying model prompts."""
    result: dict[str, Any] = {}
    for target, audio_only, image_mode in (("image", False, 1), ("video", False, 0), ("audio", True, 0)):
        choices = []
        default = ""
        if callable(resolver):
            try:
                native_choices, default, _definition = resolver(model_def, audio_only, image_mode, False)
                choices = _choices(native_choices)
            except Exception:
                choices = []
        if not choices:
            definition = model_def.get("prompt_enhancer_def")
            if isinstance(definition, dict):
                labels = definition.get("labels") or {}
                mode_letter = "P" if image_mode > 0 else "V"
                for key, label in labels.items():
                    key = str(key or "")
                    if "V" in key or "P" in key:
                        filters = {letter for letter in key if letter in "VP"}
                        if filters and mode_letter not in filters:
                            continue
                    value = key.replace("V", "").replace("P", "")
                    if value:
                        choices.append({"label": str(label or value), "value": value})
                default = str(definition.get("default") or "")
            else:
                allowed = model_def.get("prompt_enhancer_choices_allowed")
                if not isinstance(allowed, (list, tuple)):
                    allowed = ["T"] if audio_only else ["T", "TI"]
                choices = _choices(allowed)
        result[target] = {"choices": choices, "default": default}
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
    prompt_enhancer_resolver=None,
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
        native_settings = _native_settings(model_def)
        base_model_type = str(model_def.get("base_model_type") or model_def.get("architecture") or model_type)
        base_def = model_defs.get(base_model_type) or {}
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
            "base_model_type": base_model_type,
            "base_name": str(base_def.get("name") or base_model_type),
            "lora_family": str(model_def.get("lora_family") or base_model_type),
            "lora_family_name": str(model_def.get("lora_family_name") or model_def.get("family_name") or base_def.get("name") or base_model_type),
            "metadata": copy.deepcopy(model_def.get("metadata") or {}),
            "capabilities": caps,
            "svi2pro": bool(model_def.get("svi2pro")),
            "reference_input_label": "Anchor images (each window)" if model_def.get("svi2pro") else "Reference images",
            "lora_supported": not bool(model_def.get("no_lora", False)) and bool(caps.get("lora", True)),
            "loras": list(loras.get(str(model_type), [])),
            "resolution_choices": [{"label": str(label), "value": str(value), "tier": resolution_api.categorize_resolution(str(value))} for label, value in resolutions],
            "resolution_tiers": [{"label": value, "value": value} for value in tiers],
            "native_settings": native_settings,
            "defaults": _model_defaults(model_def),
            "prompt_enhancer": _prompt_enhancer_catalog(model_def, prompt_enhancer_resolver),
            "attention_supported": bool(model_def.get("custom_attention_modes")) or any(model_def.get(flag) for flag in ("tea_cache", "mag_cache", "spectrum_cache", "first_block_cache")),
        })
    base_models = {}
    for model in models:
        base = str(model.get("lora_family") or model.get("base_model_type") or model["model_type"])
        entry = base_models.setdefault(base, {"model_type": base, "name": model.get("lora_family_name") or model.get("base_name") or base, "loras": []})
        for lora in model.get("loras") or []:
            if lora not in entry["loras"]:
                entry["loras"].append(lora)
    resolution_tiers = [
        {"label": str(tier), "value": str(tier)}
        for tier in resolution_api.GROUP_THRESHOLDS
    ]
    return {
        "nodes": catalog,
        "models": models,
        "base_models": sorted(base_models.values(), key=lambda item: str(item.get("name") or "").casefold()),
        "processes": dynamic_processes,
        "ai_models": copy.deepcopy(AI_MODELS),
        "aspect_ratios": copy.deepcopy(ASPECT_RATIOS),
        "resolution_tiers": resolution_tiers,
    }
