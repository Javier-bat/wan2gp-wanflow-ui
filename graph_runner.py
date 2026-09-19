from __future__ import annotations

import copy
import json
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from postprocessing import catalog as postprocess_catalog
from shared.settings_metadata import clean_metadata_settings

from . import ai_runtime, media_ops, runtime_engine
from .graph_model import GENERATION_NODE_TYPES, effective_edges, node_ports, topological_order, validate_graph


def _runtime_engine():
    return runtime_engine


def _runtime_ai():
    return ai_runtime


def _path(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value or "").strip()


def _number(value: Any, default: int | float) -> int | float:
    try:
        return type(default)(value)
    except (TypeError, ValueError):
        return default


def _substitute(value: Any, variables: dict[str, str]) -> str:
    engine = _runtime_engine()
    return engine.substitute_variables(value, variables) if isinstance(value, str) else str(value or "")


class _Callbacks:
    def __init__(self, callback: Callable[[str, float], None] | None = None):
        self.callback = callback

    def on_status(self, status):
        if self.callback:
            self.callback(str(status or ""), 0.0)

    def on_progress(self, update):
        try:
            progress = float(getattr(update, "progress", 0) or 0)
        except (TypeError, ValueError):
            progress = 0.0
        if self.callback:
            self.callback(str(getattr(update, "status", "") or ""), progress)


def _job_client_ids(job) -> set[str]:
    return {str(value or "").strip() for value in (getattr(job, "webui_client_ids", ()) or ()) if str(value or "").strip()}


def _queue_has_job(state: dict[str, Any], job) -> bool:
    wanted = _job_client_ids(job)
    if not wanted:
        return False
    gen = state.get("gen") if isinstance(state, dict) else None
    queue = gen.get("queue", []) if isinstance(gen, dict) else []
    candidates = list(queue or [])
    if isinstance(gen, dict) and gen.get("inline_queue") is not None:
        candidates.append(gen.get("inline_queue"))
    for item in candidates:
        params = item.get("params", {}) if isinstance(item, dict) else {}
        client_id = params.get("client_id") if isinstance(params, dict) else None
        if str(client_id or "").strip() in wanted:
            return True
    return False


def _wake_webui_queue(api_session, job) -> bool:
    """Wake only the v2-owned queue when the embedded UI missed its trigger."""
    session = getattr(api_session, "_session", None)
    state = getattr(session, "_state", None)
    service = getattr(state, "service", None)
    wake = getattr(service, "_process_inline_queue", None)
    if not isinstance(state, dict) or not callable(wake) or not _queue_has_job(state, job):
        return False
    if bool(getattr(service, "generation_running", False)):
        return False
    threading.Thread(
        target=wake,
        args=(None,),
        name="wan2gp-v2-queue-wakeup",
        daemon=True,
    ).start()
    return True


def _wait_job(job, cancel_check=None, callback=None, api_session=None):
    wake_attempted = False
    submitted_at = time.monotonic()
    while not job.done:
        if cancel_check and cancel_check():
            job.cancel()
            raise InterruptedError("Workflow cancelled.")
        if not wake_attempted and api_session is not None and time.monotonic() - submitted_at >= 3.0:
            wake_attempted = _wake_webui_queue(api_session, job)
        time.sleep(0.08)
    result = job.result()
    if result.cancelled:
        raise InterruptedError("Workflow cancelled.")
    if not result.success:
        errors = list(result.errors or [])
        raise RuntimeError(str(errors[0] if errors else "Wan2GP task failed."))
    return result


def _connected_values(graph: dict[str, Any], node_id: str, port: str, values: dict[str, dict[str, Any]], model_defs: dict[str, dict[str, Any]] | None = None) -> list[Any]:
    result = []
    for edge in effective_edges(graph, model_defs):
        target = edge.get("target") or {}
        if target.get("node") == node_id and target.get("port") == port:
            source = edge.get("source") or {}
            result.append((values.get(source.get("node")) or {}).get(source.get("port")))
    return [value for value in result if value not in (None, "")]


def _first_path(values: list[Any]) -> str:
    for value in values:
        if isinstance(value, (str, Path)) and Path(value).is_file():
            return str(value)
        if isinstance(value, dict) and _path(value.get("path")) and Path(_path(value.get("path"))).is_file():
            return _path(value.get("path"))
    return ""


def _output_path(root: Path, node_id: str, suffix: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{node_id}-{uuid.uuid4().hex[:10]}{suffix}"


def _submit_magic_mask(source: str, params: dict[str, Any], output_root: Path, state: dict[str, Any], cancel_check=None) -> tuple[str, str]:
    """Run Wan2GP's managed Magic Mask implementation for one image or video."""
    from shared import magic_mask
    from shared.utils.download import process_files_def
    from shared.utils.process_locks import acquire_GPU_ressources, release_GPU_ressources
    from shared.utils.utils import has_image_file_extension

    if not source or not Path(source).is_file():
        raise ValueError("Magic Mask needs an existing image or video input.")
    if cancel_check and cancel_check():
        raise InterruptedError("Magic Mask cancelled.")

    keywords = magic_mask.parse_keywords(params.get("keywords") or params.get("objects") or "")
    if not keywords:
        raise ValueError("Magic Mask needs at least one object or person keyword.")
    max_objects = params.get("max_objects", "all")
    if max_objects in (None, "", "all"):
        max_colored_objects = None
    else:
        max_colored_objects = max(1, min(5, int(max_objects)))
    max_time = params.get("max_time")
    if max_time in (None, ""):
        max_time = None
    else:
        max_time = float(max_time)
        if max_time <= 0:
            raise ValueError("Magic Mask max video time must be greater than zero.")

    process_files_def(**magic_mask.query_download_def())
    process_id = magic_mask.PROCESS_ID
    acquired = False
    try:
        acquire_GPU_ressources(state, process_id, magic_mask.PROCESS_NAME)
        acquired = True
        def check_cancel(*_args):
            if cancel_check and cancel_check():
                raise InterruptedError("Magic Mask cancelled.")

        if has_image_file_extension(source):
            _background, video = magic_mask.prepare_image_mask_input(source)
            masks = magic_mask.generate_keyword_masks(
                video,
                keywords,
                progress_callback=check_cancel,
                max_colored_objects=max_colored_objects,
            )
            if getattr(masks, "ndim", 0) == 4 or (getattr(masks, "ndim", 0) == 3 and masks.shape[-1] != 3):
                masks = masks[0]
            mask = magic_mask.finalize_masks(masks, negative_mask=bool(params.get("negative_mask", False)))
            output = _output_path(output_root, str(params.get("node_id") or "magic-mask"), ".png")
            magic_mask.mask_to_image(mask).save(output)
            return "mask_image", str(output)

        video_path, video, fps = magic_mask.prepare_video_mask_input(source, max_time_seconds=max_time)
        masks = magic_mask.generate_keyword_masks(
            video,
            keywords,
            progress_callback=check_cancel,
            max_colored_objects=max_colored_objects,
        )
        mask = magic_mask.finalize_masks(masks, negative_mask=bool(params.get("negative_mask", False)))
        output_root.mkdir(parents=True, exist_ok=True)
        output = magic_mask.save_mask_video(
            video_path,
            mask,
            fps,
            keywords,
            output_dir=str(output_root),
            abort_callback=check_cancel,
        )
        return "mask_video", str(output)
    finally:
        if acquired:
            release_GPU_ressources(state, process_id)


def _submit_generation(plugin, api_session, main_state, node, inputs, variables, workflow_settings, output_root, callback, cancel_check):
    engine = _runtime_engine()
    params = node.get("params") or {}
    settings = engine.clean_settings(copy.deepcopy(params.get("settings") or {}))
    model_type = str(params.get("model_type") or settings.get("model_type") or settings.get("base_model_type") or "")
    if not model_type:
        raise ValueError(f"{node['title']} has no model selected.")
    settings["model_type"] = model_type
    node_type = str(node.get("type") or "")
    if node_type.endswith("_image") or node_type == "generate_image":
        settings["image_mode"] = 1
    elif node_type.endswith("_video") or node_type == "generate_video":
        settings["image_mode"] = 0
    model_def = plugin.get_model_def(model_type)
    if not isinstance(model_def, dict):
        raise ValueError(f"Model '{model_type}' is no longer available.")
    for key in ("prompt", "negative_prompt", "seed", "steps", "num_inference_steps", "video_length", "cfg_scale", "guidance_scale", "sample_solver"):
        if key in params and params[key] not in (None, ""):
            settings[key] = _substitute(params[key], variables) if "prompt" in key else params[key]
    if params.get("steps") not in (None, ""):
        settings["num_inference_steps"] = params["steps"]
    if "activated_loras" in params:
        values = params.get("activated_loras") or []
        if isinstance(values, str):
            values = [values]
        settings["activated_loras"] = [str(value).strip() for value in values if str(value).strip()]
    if "loras_multipliers" in params:
        settings["loras_multipliers"] = str(params.get("loras_multipliers") or "")
    prompt_values = inputs.get("prompt") or []
    if prompt_values:
        settings["prompt"] = _substitute(prompt_values[-1], variables)
    settings.setdefault("repeat_generation", 1)
    settings.setdefault("batch_size", 1)
    source_path = _first_path((inputs.get("image") or []) + (inputs.get("video") or []) + (inputs.get("media") or []))
    source_port = "image" if inputs.get("image") else "video" if inputs.get("video") else ""
    mask = _first_path(inputs.get("mask") or [])
    masked_image = bool(mask and source_port == "image" and node_type in {"generate_image", "edit_image", "inpaint_image"})
    masked_video = bool(mask and source_port == "video" and node_type in {"generate_video", "edit_video", "inpaint_video"})
    if masked_image:
        # Wan2GP's image inpainting contract is image_mode=2 with a Control
        # Image and a MASKED_AREA prompt mode.  image_mode=1 treats the same
        # file as a reference and silently clears image_mask later in wgp.py.
        settings["image_mode"] = 2
        settings["image_guide"] = source_path
        settings["image_mask"] = mask
        settings["image_prompt_type"] = ""
        settings["video_prompt_type"] = str(model_def.get("inpaint_video_prompt_type") or "VAG")
        for key in ("image_start", "image_end", "video_source", "video_guide", "video_guide2"):
            settings.pop(key, None)
    elif masked_video:
        # Video inpainting uses the normal video output mode, but the guide
        # and mask must be supplied together with Wan2GP's V+A mode.
        settings["video_guide"] = source_path
        settings["video_mask"] = mask
        settings["video_prompt_type"] = str(model_def.get("inpaint_video_prompt_type") or "VAG")
        settings.pop("video_source", None)
        for key in ("image_start", "image_end", "image_guide", "image_mask"):
            settings.pop(key, None)
    elif source_path:
        mode = "start_image" if source_port == "image" else "continue_video"
        settings, _resolved = engine.bind_source(settings, source_path, params.get("source_mode") or mode, model_def)
    references = [_path(item) for item in (inputs.get("references") or []) if Path(_path(item)).is_file()]
    if references and (not masked_image or model_def.get("inpaint_with_image_ref", False)):
        settings = engine.add_stage_references(settings, references, model_def)
    reference_videos = [_path(item) for item in (inputs.get("reference_videos") or []) if Path(_path(item)).is_file()]
    if reference_videos:
        settings = engine.add_stage_video_references(settings, reference_videos, model_def)
    if mask and not (masked_image or masked_video):
        settings["video_mask" if source_port == "video" else "image_mask"] = mask
    end_frame = _first_path(inputs.get("end_frame") or [])
    if end_frame:
        settings = engine.bind_end_frame(settings, end_frame, model_def)
    frame_paths = [_path(item) for item in (inputs.get("frames") or []) if Path(_path(item)).is_file()]
    if frame_paths:
        settings = engine.bind_injected_frames(settings, frame_paths, str(params.get("frame_positions") or "1"), model_def)
    if inputs.get("audio"):
        settings["audio_guide"] = _first_path(inputs["audio"])
    try:
        choices, _ = plugin.get_resolution_choices(settings.get("resolution"), model_def)
        aspect_value = params.get("aspect_ratio")
        if not aspect_value:
            aspect_value = (workflow_settings or {}).get("aspect_ratio") or "source"
        source_for_ratio = source_path if str(aspect_value).strip().lower() == "source" else None
        aspect_ratio = engine.workflow_aspect_ratio(aspect_value, source_for_ratio)
        requested_resolution = str(params.get("resolution") or "").strip().lower()
        if requested_resolution and any(str(value).lower() == requested_resolution for _label, value in choices):
            settings["resolution"] = requested_resolution
        else:
            settings["resolution"] = engine.resolution_for_tier(
                choices, params.get("resolution_tier") or "720p", aspect_ratio, int(model_def.get("vae_block_size", 16) or 16),
            )
    except Exception:
        pass
    # Apply the same model-aware visibility/compatibility cleanup used by
    # Wan2GP's main generator. This prevents settings from a previous model
    # from leaking into the selected model's request.
    settings = clean_metadata_settings(
        settings,
        model_def,
        attention_mode=str(settings.get("override_attention") or ""),
    )
    settings.pop("client_id", None)
    settings.pop("priority", None)
    callbacks = _Callbacks(callback)
    job = api_session.submit_task(copy.deepcopy(settings), callbacks=callbacks)
    result = _wait_job(job, cancel_check=cancel_check, callback=callback, api_session=api_session)
    expected = engine.expected_output_kind(settings, model_def)
    return engine.select_generated_path(result, expected)


def _submit_postprocess(api_session, node, source, inputs, callback, cancel_check):
    params = node.get("params") or {}
    process_type = str(params.get("process_type") or "").strip()
    process_id = str(params.get("process_id") or "").strip()
    process = next((item for media_type in ("image", "video", "audio") for item in postprocess_catalog.query_processes(media_type, enabled_only=False) if item.get("id") == process_id), None) if process_id else None
    if process:
        process_type = str(process.get("type") or process_type)
    parameter_values = copy.deepcopy(params.get("parameters") or {})
    if process:
        for parameter in process.get("parameters") or []:
            name = str(parameter.get("name") or "")
            if name and name not in parameter_values and name in params:
                parameter_values[name] = params[name]
        references = [_path(item) for item in (inputs.get("references") or []) if Path(_path(item)).is_file()]
        if references and any(parameter.get("name") == "spatial_upsampler_reference_images" for parameter in process.get("parameters") or []):
            parameter_values["spatial_upsampler_reference_images"] = references
        parameter_values, parameter_error = postprocess_catalog.normalize_parameters(process, parameter_values)
        if parameter_error:
            raise ValueError(f"{node['title']}: {parameter_error}")
    else:
        parameter_values = parameter_values or {}
    method = str(params.get("method") or params.get("temporal_upsampling") or params.get("spatial_upsampling") or "")
    if process:
        method = postprocess_catalog.build_process_value(process, parameter_values) or method
    callbacks = _Callbacks(callback)
    if process_type == "temporal_upsampling":
        job = api_session.submit_media_postprocessing(source, temporal_upsampling=method or "rife4", return_media=True, callbacks=callbacks, **{key: value for key, value in parameter_values.items() if key != "multiplier"})
    elif process_type == "spatial_upsampling":
        job = api_session.submit_media_postprocessing(source, spatial_upsampling=method or "lanczos2", return_media=True, callbacks=callbacks, **{key: value for key, value in parameter_values.items() if key != "multiplier"})
    elif process_type == "soundtrack":
        audio_source = _first_path(inputs.get("audio") or [])
        job = api_session.submit_audio_remux(source, postprocess_audio=method or "mmaudio", audio_source=audio_source or None, postprocess_audio_prompt=parameter_values.get("prompt", params.get("prompt", "")), postprocess_audio_neg_prompt=parameter_values.get("negative_prompt", params.get("negative_prompt", "")), seed=int(_number(parameter_values.get("seed", params.get("seed")), -1)), return_media=True, callbacks=callbacks)
    elif process_type == "voice_replacement":
        job = api_session.submit_audio_remux(source, postprocess_audio=method, replace_voice_sample=parameter_values.get("voice_sample_media_id", params.get("replace_voice_sample")), replace_voice_sample2=parameter_values.get("voice_sample2_media_id", params.get("replace_voice_sample2")), return_media=True, callbacks=callbacks)
    elif process_type == "audio_edit":
        job = api_session.submit_audio_postprocessing(source, postprocess_audio=method, return_media=True, callbacks=callbacks, **parameter_values)
    else:
        raise ValueError(f"Unsupported postprocess type: {process_type or method}.")
    result = _wait_job(job, cancel_check=cancel_check, callback=callback, api_session=api_session)
    expected_kind = "audio" if process_type == "audio_edit" else "video" if process_type in {"soundtrack", "voice_replacement", "temporal_upsampling", "spatial_upsampling"} else ""
    output = _runtime_engine().select_generated_path(result, expected_kind)
    if not output:
        raise RuntimeError("Wan2GP postprocessing completed without a media output.")
    return output


def run_graph(plugin, api_session, main_state: dict[str, Any], graph: dict[str, Any], runtime_inputs: list[str] | None = None, *, output_root: str | Path, callback: Callable[[str, float], None] | None = None, cancel_check: Callable[[], bool] | None = None, node_callback: Callable[[str, str, dict[str, Any] | None], None] | None = None) -> tuple[list[str], dict[str, str]]:
    model_defs = {}
    try:
        model_defs = {str(item.get("model_type")): item for item in plugin.list_model_defs() or [] if isinstance(item, dict)}
    except Exception:
        pass
    errors = validate_graph(graph, model_defs)
    if errors:
        raise ValueError("Workflow graph is invalid:\n- " + "\n- ".join(errors))
    nodes = {node["id"]: node for node in graph.get("nodes") or []}
    values: dict[str, dict[str, Any]] = {}
    variables: dict[str, str] = {}
    outputs: list[str] = []
    runtime_paths = []
    for value in runtime_inputs or []:
        path = getattr(value, "path", None) or getattr(value, "name", None) or value
        path = str(path or "").strip()
        if path:
            runtime_paths.append(path)
    root = Path(output_root)
    auto_input_slot = 0
    order = topological_order(graph, model_defs)
    for index, node_id in enumerate(order, start=1):
        if cancel_check and cancel_check():
            raise InterruptedError("Workflow cancelled.")
        node = nodes[node_id]
        if not node.get("enabled", True):
            if node_callback:
                node_callback(node_id, "skipped", None)
            continue
        node_type = node.get("type")
        inputs = {port: _connected_values(graph, node_id, port, values, model_defs) for port in node_ports(node, model_defs) if node_ports(node, model_defs)[port].get("direction") == "in"}
        if node_callback:
            node_callback(node_id, "running", None)
        if callback:
            callback(f"{index}/{len(nodes)} | {node['title']}", 0.0)
        result: dict[str, Any] = {}
        params = node.get("params") or {}
        if node_type in {"input_image", "input_video", "input_audio", "input_mask_image", "input_mask_video", "load_media"}:
            slot = int(_number(params.get("slot"), auto_input_slot))
            auto_input_slot = max(auto_input_slot + 1, slot + 1)
            if slot >= len(runtime_paths):
                raise ValueError(f"{node['title']} needs runtime input slot {slot}.")
            result["image" if node_type == "input_image" else "video" if node_type == "input_video" else "audio" if node_type == "input_audio" else "mask" if node_type in {"input_mask_image", "input_mask_video"} else "media"] = runtime_paths[slot]
        elif node_type == "text":
            result["text"] = _substitute(params.get("text") or params.get("value") or "", variables)
        elif node_type in GENERATION_NODE_TYPES:
            result["output"] = _submit_generation(plugin, api_session, main_state, node, inputs, variables, graph.get("settings") or {}, root, callback, cancel_check)
        elif node_type == "ai_analyze":
            images = [_first_path([item]) for item in inputs.get("images") or []]
            instruction = _substitute((inputs.get("instruction") or [params.get("instruction") or "Describe the image."])[-1], variables)
            answer = _runtime_ai().run_ai_analysis(getattr(plugin, "_deepy", None), main_state, params.get("model", 3), instruction, [path for path in images if path], params.get("max_tokens", 192), reset_prompt_enhancer=getattr(plugin, "reset_prompt_enhancer", None), reset_prompt_enhancer_if_requested=getattr(plugin, "reset_prompt_enhancer_if_requested", None))
            variables[str(params.get("output_name") or "analysis")] = answer
            result["text"] = answer
        elif node_type in {"last_frame", "extract_frame"}:
            source = _first_path(inputs.get("video") or [])
            if not source:
                raise ValueError(f"{node['title']} needs a video input.")
            frame = int(_number(params.get("frame"), 1))
            result["image"] = media_ops.extract_last_frame(source, _output_path(root, node_id, ".png"), cancel_check) if node_type == "last_frame" else media_ops.extract_frame(source, frame, _output_path(root, node_id, ".png"), cancel_check)
        elif node_type == "video_probe":
            source = _first_path(inputs.get("video") or [])
            result["metadata"] = json.dumps(media_ops.probe(source), ensure_ascii=False)
        elif node_type == "image_sequence":
            result["video"] = media_ops.image_sequence(
                [_path(item) for item in inputs.get("images") or []],
                float(_number(params.get("fps"), 24)),
                _output_path(root, node_id, ".mp4"),
                cancel_check,
            )
        elif node_type == "magic_mask":
            source = _first_path(inputs.get("source") or [])
            mask_port, mask_path = _submit_magic_mask(
                source,
                {**params, "node_id": node_id},
                root,
                main_state,
                cancel_check,
            )
            result[mask_port] = mask_path
        elif node_type == "mask_editor":
            mask_path = _first_path(inputs.get("mask") or []) or _path(params.get("mask_path"))
            if not mask_path or not Path(mask_path).is_file():
                raise ValueError(f"{node['title']} needs an existing mask file. Use Input Mask Image/Video or the native mask editor output.")
            result["mask"] = mask_path
        elif node_type == "postprocess":
            source = _first_path(inputs.get("media") or [])
            if not source:
                raise ValueError(f"{node['title']} needs a media input.")
            result["output"] = _submit_postprocess(api_session, node, source, inputs, callback, cancel_check)
        elif node_type == "ffmpeg_trim":
            result["output"] = media_ops.trim(_first_path(inputs.get("video") or []), float(params.get("start", 0)), float(params.get("duration", 5)), _output_path(root, node_id, ".mp4"), cancel_check)
        elif node_type == "ffmpeg_concat":
            result["output"] = media_ops.concat([_path(item) for item in inputs.get("videos") or []], _output_path(root, node_id, ".mp4"), cancel_check)
        elif node_type == "ffmpeg_mux_audio":
            result["output"] = media_ops.mux_audio(_first_path(inputs.get("video") or []), _first_path(inputs.get("audio") or []), _output_path(root, node_id, ".mp4"), cancel_check)
        elif node_type == "ffmpeg_remove_audio":
            result["output"] = media_ops.remove_audio(_first_path(inputs.get("video") or []), _output_path(root, node_id, ".mp4"), cancel_check)
        elif node_type == "ffmpeg_resize":
            result["output"] = media_ops.resize(_first_path(inputs.get("video") or []), int(_number(params.get("width"), 832)), int(_number(params.get("height"), 480)), _output_path(root, node_id, ".mp4"), cancel_check)
        elif node_type == "ffmpeg_fps":
            result["output"] = media_ops.fps(_first_path(inputs.get("video") or []), float(params.get("fps", 24)), _output_path(root, node_id, ".mp4"), cancel_check)
        elif node_type in {"ffmpeg_normalize", "ffmpeg_transcode", "ffmpeg_export"}:
            source = _first_path(inputs.get("video") or inputs.get("media") or [])
            target = _output_path(root, node_id, Path(source).suffix or ".mp4")
            shutil.copy2(source, target)
            result["output"] = str(target)
        else:
            raise ValueError(f"Unsupported node type: {node_type}")
        values[node_id] = result
        if node_callback:
            node_callback(node_id, "done", result)
        for value in result.values():
            if isinstance(value, str) and Path(value).is_file():
                outputs.append(value)
    return list(dict.fromkeys(outputs)), variables
