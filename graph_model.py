from __future__ import annotations

import copy
import json
import uuid
from collections import defaultdict, deque
from typing import Any


SCHEMA = "wan2gp-wanflow-ui"
LEGACY_SCHEMA = "wan2gp-workflows-v2"
VERSION = 1

IMAGE = "IMAGE"
VIDEO = "VIDEO"
AUDIO = "AUDIO"
MASK_IMAGE = "MASK_IMAGE"
MASK_VIDEO = "MASK_VIDEO"
TEXT = "TEXT"
FRAME_LIST = "FRAME_LIST"
ANY_MEDIA = "MEDIA"


INPUT_NODE_TYPES = {"input_image", "input_video", "input_audio", "input_mask_image", "input_mask_video", "load_media"}
GENERATION_NODE_TYPES = {
    "generate_image",
    "edit_image",
    "inpaint_image",
    "generate_video",
    "edit_video",
    "inpaint_video",
    "generate_audio",
}
PROCESS_NODE_TYPES = {
    "last_frame",
    "extract_frame",
    "image_sequence",
    "video_probe",
    "magic_mask",
    "postprocess",
    "ffmpeg_trim",
    "ffmpeg_concat",
    "ffmpeg_mux_audio",
    "ffmpeg_remove_audio",
    "ffmpeg_transcode",
    "ffmpeg_resize",
    "ffmpeg_fps",
    "ffmpeg_normalize",
    "ffmpeg_export",
}


def new_graph(name: str = "New workflow") -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "id": uuid.uuid4().hex,
        "name": str(name or "New workflow").strip(),
        "settings": {"aspect_ratio": "source", "unload_on_model_change": False},
        "nodes": [],
        "edges": [],
        "groups": [],
        "ui": {},
        "warnings": [],
    }


def new_node(node_type: str, title: str | None = None, *, x: float = 80, y: float = 80, params: dict[str, Any] | None = None) -> dict[str, Any]:
    labels = {
        "input_image": "Input Image",
        "input_video": "Input Video",
        "input_audio": "Input Audio",
        "input_mask_image": "Input Mask Image",
        "input_mask_video": "Input Mask Video",
        "load_media": "Load Media",
        "text": "Text",
        "generate_image": "Generate Image",
        "edit_image": "Edit Image",
        "inpaint_image": "Inpaint Image",
        "generate_video": "Generate Video",
        "edit_video": "Edit Video",
        "inpaint_video": "Inpaint Video",
        "generate_audio": "Generate Audio",
        "ai_analyze": "AI Analyze",
        "mask_editor": "Mask Editor",
        "magic_mask": "Magic Mask",
        "last_frame": "Last Frame",
        "extract_frame": "Extract Frame",
        "image_sequence": "Image Sequence",
        "video_probe": "Video Probe",
        "postprocess": "Postprocess",
        "ffmpeg_trim": "FFmpeg Trim",
        "ffmpeg_concat": "FFmpeg Concat",
        "ffmpeg_mux_audio": "FFmpeg Mux Audio",
        "ffmpeg_remove_audio": "FFmpeg Remove Audio",
        "ffmpeg_transcode": "FFmpeg Transcode",
        "ffmpeg_resize": "FFmpeg Resize",
        "ffmpeg_fps": "FFmpeg FPS",
        "ffmpeg_normalize": "FFmpeg Normalize",
        "ffmpeg_export": "FFmpeg Export",
    }
    return {
        "id": uuid.uuid4().hex,
        "type": str(node_type),
        "title": str(title or labels.get(node_type, node_type.replace("_", " ").title())),
        "position": {"x": float(x), "y": float(y)},
        "params": copy.deepcopy(params or {}),
        "ui": {},
        "enabled": True,
    }


def _text(value: Any) -> str:
    return str(value or "").strip()


def _metadata(model_def: dict[str, Any] | None) -> dict[str, Any]:
    return dict((model_def or {}).get("metadata") or {})


def _media_port_type(value: Any) -> str:
    normalized = _text(value).lower()
    return {"image": IMAGE, "video": VIDEO, "audio": AUDIO, "mask_image": MASK_IMAGE, "mask_video": MASK_VIDEO, "text": TEXT, "frame_list": FRAME_LIST}.get(normalized, _text(value))


def model_capabilities(model_def: dict[str, Any] | None) -> dict[str, Any]:
    metadata = _metadata(model_def)
    capabilities = dict(metadata.get("capabilities") or {})
    media_inputs = dict(metadata.get("media_inputs") or {})
    image_inputs = dict(media_inputs.get("image") or {})
    video_inputs = dict(media_inputs.get("video") or {})
    audio_inputs = dict(media_inputs.get("audio") or {})
    outputs = [_media_port_type(value) for value in list(metadata.get("outputs") or metadata.get("main_output") or [])]
    main_output = [_media_port_type(value) for value in list(metadata.get("main_output") or outputs)]
    return {
        **capabilities,
        "inputs": list(metadata.get("inputs") or ["text"]),
        "outputs": outputs,
        "main_output": main_output,
        "media_inputs": media_inputs,
        "image_inputs": image_inputs,
        "video_inputs": video_inputs,
        "audio_inputs": audio_inputs,
        "text_only": outputs == [IMAGE] and not any(image_inputs.values()) or outputs == [VIDEO] and not any(video_inputs.values()),
    }


def _generation_ports(node: dict[str, Any], model_def: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    caps = model_capabilities(model_def)
    node_type = _text(node.get("type"))
    expected_output = AUDIO if node_type == "generate_audio" else IMAGE if "image" in node_type else VIDEO
    outputs = caps["main_output"] or [expected_output]
    output_type = expected_output if expected_output in outputs else outputs[0]
    result: dict[str, dict[str, Any]] = {"prompt": {"direction": "in", "type": TEXT, "optional": True}}
    image_inputs = caps["image_inputs"]
    video_inputs = caps["video_inputs"]
    audio_inputs = caps["audio_inputs"]
    if image_inputs.get("start") or image_inputs.get("reference") or node_type in {"edit_image", "inpaint_image"}:
        result["image"] = {"direction": "in", "type": IMAGE, "optional": node_type not in {"edit_image", "inpaint_image"}}
    if image_inputs.get("multiple_references"):
        result["references"] = {"direction": "in", "type": f"{IMAGE}[]", "optional": True, "variadic": True}
    if video_inputs.get("continue") or video_inputs.get("reference") or video_inputs.get("control") or node_type in {"edit_video", "inpaint_video"}:
        result["video"] = {"direction": "in", "type": VIDEO, "optional": node_type not in {"edit_video", "inpaint_video"}}
    if video_inputs.get("reference"):
        result["reference_videos"] = {"direction": "in", "type": f"{VIDEO}[]", "optional": True, "variadic": True}
    if audio_inputs.get("prompt") or node_type == "generate_audio":
        result["audio"] = {"direction": "in", "type": AUDIO, "optional": True}
    if caps.get("inpainting") or node_type in {"inpaint_image", "inpaint_video"}:
        result["mask"] = {
            "direction": "in",
            "type": MASK_IMAGE if output_type == IMAGE else MASK_VIDEO,
            # A model may support inpainting without every generation using
            # it.  The dedicated Inpaint nodes require the mask; a normal
            # Generate/Edit node keeps it as an optional capability port.
            "optional": node_type not in {"inpaint_image", "inpaint_video"},
        }
    if image_inputs.get("end"):
        result["end_frame"] = {"direction": "in", "type": IMAGE, "optional": True}
    if image_inputs.get("injected_frames"):
        result["frames"] = {"direction": "in", "type": f"{IMAGE}[]", "optional": True, "variadic": True}
    result.update({"output": {"direction": "out", "type": output_type}})
    return result


def model_supports_node(node_type: str, model_def: dict[str, Any] | None) -> bool:
    caps = model_capabilities(model_def)
    expected = AUDIO if node_type == "generate_audio" else IMAGE if "image" in node_type else VIDEO
    if expected not in (caps.get("main_output") or caps.get("outputs") or []):
        return False
    if node_type.startswith("inpaint_"):
        return bool(caps.get("inpainting"))
    if node_type == "edit_image":
        return bool(caps.get("image_to_image") or caps.get("reference_images") or caps.get("control_image"))
    if node_type == "edit_video":
        return bool(caps.get("video_to_video") or caps.get("video_continuation") or caps.get("reference_videos") or caps.get("control_video"))
    return True


def node_ports(node: dict[str, Any], model_defs: dict[str, dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    node_type = _text(node.get("type"))
    params = node.get("params") if isinstance(node.get("params"), dict) else {}
    model_type = _text(params.get("model_type"))
    model_def = (model_defs or {}).get(model_type) if model_type else None
    if node_type in GENERATION_NODE_TYPES:
        return _generation_ports(node, model_def)
    if node_type == "text":
        return {"text": {"direction": "out", "type": TEXT}}
    if node_type == "mask_editor":
        return {"source": {"direction": "in", "type": IMAGE, "optional": True}, "mask": {"direction": "out", "type": MASK_IMAGE}}
    if node_type == "magic_mask":
        return {
            "source": {"direction": "in", "type": ANY_MEDIA},
            "mask_image": {"direction": "out", "type": MASK_IMAGE},
            "mask_video": {"direction": "out", "type": MASK_VIDEO},
        }
    if node_type == "input_image":
        return {"image": {"direction": "out", "type": IMAGE}}
    if node_type == "input_video":
        return {"video": {"direction": "out", "type": VIDEO}}
    if node_type == "input_audio":
        return {"audio": {"direction": "out", "type": AUDIO}}
    if node_type == "input_mask_image":
        return {"mask": {"direction": "out", "type": MASK_IMAGE}}
    if node_type == "input_mask_video":
        return {"mask": {"direction": "out", "type": MASK_VIDEO}}
    if node_type == "load_media":
        return {"media": {"direction": "out", "type": ANY_MEDIA}}
    if node_type == "ai_analyze":
        return {"images": {"direction": "in", "type": f"{IMAGE}[]", "optional": True, "variadic": True}, "instruction": {"direction": "in", "type": TEXT, "optional": True}, "text": {"direction": "out", "type": TEXT}}
    if node_type in {"last_frame", "extract_frame"}:
        return {"video": {"direction": "in", "type": VIDEO}, "image": {"direction": "out", "type": IMAGE}}
    if node_type == "image_sequence":
        return {"images": {"direction": "in", "type": f"{IMAGE}[]", "variadic": True}, "video": {"direction": "out", "type": VIDEO}}
    if node_type == "video_probe":
        return {"video": {"direction": "in", "type": VIDEO}, "metadata": {"direction": "out", "type": TEXT}}
    if node_type == "postprocess":
        return {
            "media": {"direction": "in", "type": ANY_MEDIA},
            "references": {"direction": "in", "type": f"{IMAGE}[]", "optional": True, "variadic": True},
            "audio": {"direction": "in", "type": AUDIO, "optional": True},
            "output": {"direction": "out", "type": ANY_MEDIA},
        }
    if node_type in {"ffmpeg_trim", "ffmpeg_resize", "ffmpeg_fps", "ffmpeg_normalize", "ffmpeg_export", "ffmpeg_remove_audio"}:
        return {"video": {"direction": "in", "type": VIDEO}, "output": {"direction": "out", "type": VIDEO}}
    if node_type == "ffmpeg_concat":
        return {"videos": {"direction": "in", "type": f"{VIDEO}[]", "variadic": True}, "output": {"direction": "out", "type": VIDEO}}
    if node_type == "ffmpeg_mux_audio":
        return {"video": {"direction": "in", "type": VIDEO}, "audio": {"direction": "in", "type": AUDIO}, "output": {"direction": "out", "type": VIDEO}}
    if node_type == "ffmpeg_transcode":
        return {"media": {"direction": "in", "type": ANY_MEDIA}, "output": {"direction": "out", "type": ANY_MEDIA}}
    return {}


def _normalize_node(node: Any, index: int) -> dict[str, Any] | None:
    if not isinstance(node, dict):
        return None
    normalized = new_node(
        _text(node.get("type")) or "text",
        _text(node.get("title")) or None,
        x=float(((node.get("position") or {}).get("x", 80 + index * 40))),
        y=float(((node.get("position") or {}).get("y", 80 + index * 30))),
        params=node.get("params") if isinstance(node.get("params"), dict) else {},
    )
    normalized["id"] = _text(node.get("id")) or normalized["id"]
    normalized["ui"] = copy.deepcopy(node.get("ui") if isinstance(node.get("ui"), dict) else {})
    normalized["enabled"] = bool(node.get("enabled", True))
    return normalized


def normalize_graph(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return new_graph()
    graph = new_graph(_text(value.get("name")) or "New workflow")
    graph["id"] = _text(value.get("id")) or graph["id"]
    graph["settings"] = copy.deepcopy(value.get("settings") if isinstance(value.get("settings"), dict) else {})
    graph["ui"] = copy.deepcopy(value.get("ui") if isinstance(value.get("ui"), dict) else {})
    graph["warnings"] = [str(item) for item in value.get("warnings", []) if str(item).strip()]
    graph["nodes"] = [node for index, raw in enumerate(value.get("nodes") or []) if (node := _normalize_node(raw, index))]
    node_ids = {node["id"] for node in graph["nodes"]}
    groups = []
    seen_groups = set()
    for raw in value.get("groups") or []:
        if not isinstance(raw, dict):
            continue
        group_id = _text(raw.get("id")) or uuid.uuid4().hex
        if group_id in seen_groups:
            continue
        members = [node_id for node_id in raw.get("node_ids") or [] if _text(node_id) in node_ids]
        if not members:
            continue
        seen_groups.add(group_id)
        groups.append({
            "id": group_id,
            "title": _text(raw.get("title")) or "Group",
            "color": _text(raw.get("color")) or "#395b78",
            "node_ids": list(dict.fromkeys(members)),
            "enabled": bool(raw.get("enabled", True)),
            "collapsed": bool(raw.get("collapsed", False)),
        })
    graph["groups"] = groups
    edges = []
    seen_edges = set()
    for raw in value.get("edges") or []:
        if not isinstance(raw, dict):
            continue
        source = raw.get("source") if isinstance(raw.get("source"), dict) else {}
        target = raw.get("target") if isinstance(raw.get("target"), dict) else {}
        source_node, target_node = _text(source.get("node")), _text(target.get("node"))
        source_port, target_port = _text(source.get("port")), _text(target.get("port"))
        if not source_node or not target_node or source_node not in node_ids or target_node not in node_ids or not source_port or not target_port:
            continue
        key = (source_node, source_port, target_node, target_port)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        edges.append({
            "id": _text(raw.get("id")) or uuid.uuid4().hex,
            "source": {"node": source_node, "port": source_port},
            "target": {"node": target_node, "port": target_port},
            "type": _text(raw.get("type")),
        })
    graph["edges"] = edges
    return graph


def _port_base(port_type: str) -> str:
    return _text(port_type).removesuffix("[]")


def compatible_types(source_type: str, target_type: str) -> bool:
    source_type, target_type = _port_base(source_type), _port_base(target_type)
    return source_type == target_type or source_type == ANY_MEDIA or target_type == ANY_MEDIA


def _disabled_group_nodes(graph: dict[str, Any]) -> set[str]:
    disabled = set()
    for group in graph.get("groups") or []:
        if not group.get("enabled", True):
            disabled.update(group.get("node_ids") or [])
    return disabled


def _edge_type(graph: dict[str, Any], edge: dict[str, Any], model_defs: dict[str, dict[str, Any]] | None = None) -> str:
    if edge.get("type"):
        return str(edge["type"])
    source = edge.get("source") or {}
    node = next((item for item in graph.get("nodes") or [] if item.get("id") == source.get("node")), None)
    return str((node_ports(node or {}, model_defs).get(source.get("port")) or {}).get("type") or "")


def effective_edges(graph: dict[str, Any], model_defs: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Return graph edges plus safe bypasses for disabled groups.

    A disabled visual group is allowed to behave as a bypass only when its
    boundary is unambiguous: one incoming edge and one or more compatible
    outgoing edges. Complex groups remain disabled and surface a missing-input
    error downstream instead of silently guessing a route.
    """
    result = list(graph.get("edges") or [])
    for group in graph.get("groups") or []:
        if group.get("enabled", True):
            continue
        members = set(group.get("node_ids") or [])
        incoming = [edge for edge in graph.get("edges") or [] if edge.get("target", {}).get("node") in members and edge.get("source", {}).get("node") not in members]
        outgoing = [edge for edge in graph.get("edges") or [] if edge.get("source", {}).get("node") in members and edge.get("target", {}).get("node") not in members]
        if len(incoming) != 1:
            continue
        source_edge = incoming[0]
        source_type = _edge_type(graph, source_edge, model_defs)
        for target_edge in outgoing:
            target_node = next((item for item in graph.get("nodes") or [] if item.get("id") == target_edge.get("target", {}).get("node")), None)
            target_port = (node_ports(target_node or {}, model_defs).get(target_edge.get("target", {}).get("port")) or {}).get("type", "")
            if not compatible_types(source_type, target_port):
                continue
            result.append({
                "id": f"bypass:{group.get('id')}:{target_edge.get('id')}",
                "source": copy.deepcopy(source_edge.get("source") or {}),
                "target": copy.deepcopy(target_edge.get("target") or {}),
                "type": source_type,
                "bypass": True,
            })
    return result


def validate_graph(graph: dict[str, Any], model_defs: dict[str, dict[str, Any]] | None = None) -> list[str]:
    graph = normalize_graph(graph)
    errors: list[str] = []
    nodes = {node["id"]: node for node in graph["nodes"]}
    inactive = _disabled_group_nodes(graph)
    if len(nodes) != len(graph["nodes"]):
        errors.append("Node IDs must be unique.")
    edge_keys = set()
    incoming: dict[tuple[str, str], int] = defaultdict(int)
    for edge in effective_edges(graph, model_defs):
        source, target = edge["source"], edge["target"]
        if source["node"] in inactive or target["node"] in inactive:
            continue
        source_node, target_node = nodes.get(source["node"]), nodes.get(target["node"])
        if source_node is None or target_node is None:
            errors.append(f"Edge {edge['id']} references a missing node.")
            continue
        key = (source["node"], source["port"], target["node"], target["port"])
        if key in edge_keys:
            errors.append(f"Duplicate edge {edge['id']}.")
        edge_keys.add(key)
        source_def = node_ports(source_node, model_defs).get(source["port"])
        target_def = node_ports(target_node, model_defs).get(target["port"])
        if not source_def or source_def.get("direction") != "out":
            errors.append(f"Unknown output port {source['port']} on {source_node['title']}.")
            continue
        if not target_def or target_def.get("direction") != "in":
            errors.append(f"Unknown input port {target['port']} on {target_node['title']}.")
            continue
        source_type, target_type = source_def.get("type", ""), target_def.get("type", "")
        if not compatible_types(source_type, target_type):
            errors.append(f"Incompatible connection {source_type} -> {target_type}: {source_node['title']} -> {target_node['title']}.")
        incoming[(target["node"], target["port"])] += 1
    for node in graph["nodes"]:
        if not node.get("enabled", True) or node["id"] in inactive:
            continue
        for port_name, port in node_ports(node, model_defs).items():
            if port.get("direction") != "in" or port.get("optional") or port.get("variadic"):
                continue
            if not incoming.get((node["id"], port_name)):
                params = node.get("params") or {}
                if port_name == "prompt" and params.get("prompt"):
                    continue
                errors.append(f"Required input {port_name} is not connected on {node['title']}.")
        if node["type"] in GENERATION_NODE_TYPES:
            model_type = _text((node.get("params") or {}).get("model_type"))
            if not model_type:
                errors.append(f"{node['title']} needs a model_type.")
            elif model_defs is not None and not model_supports_node(node["type"], (model_defs or {}).get(model_type)):
                errors.append(f"{node['title']} cannot use model '{model_type}' for this node type.")
            if incoming.get((node["id"], "mask")):
                source_port = "video" if "video" in node["type"] else "image"
                if not incoming.get((node["id"], source_port)):
                    errors.append(f"Mask input on {node['title']} requires a connected {source_port} source.")
    try:
        topological_order(graph)
    except ValueError as exc:
        errors.append(str(exc))
    return list(dict.fromkeys(errors))


def topological_order(graph: dict[str, Any], model_defs: dict[str, dict[str, Any]] | None = None) -> list[str]:
    graph = normalize_graph(graph)
    inactive = _disabled_group_nodes(graph)
    node_ids = [node["id"] for node in graph["nodes"] if node.get("enabled", True) and node["id"] not in inactive]
    enabled = set(node_ids)
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    indegree = {node_id: 0 for node_id in node_ids}
    for edge in effective_edges(graph, model_defs):
        source, target = edge["source"]["node"], edge["target"]["node"]
        if source not in enabled or target not in enabled or target in adjacency[source]:
            continue
        adjacency[source].add(target)
        indegree[target] += 1
    queue = deque(node_id for node_id in node_ids if indegree[node_id] == 0)
    order: list[str] = []
    while queue:
        node_id = queue.popleft()
        order.append(node_id)
        for target in sorted(adjacency[node_id]):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if len(order) != len(node_ids):
        raise ValueError("The workflow graph contains a cycle.")
    return order


def graph_for_json(graph: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_graph(graph)
    normalized.pop("warnings", None)
    return normalized


def _legacy_input_node(graph: dict[str, Any], title: str = "Workflow Input") -> dict[str, Any]:
    node = new_node("load_media", title, x=40, y=80, params={"slot": 0})
    graph["nodes"].append(node)
    return node


def migrate_v1_workflow(value: dict[str, Any]) -> dict[str, Any]:
    """Convert the v1 ordered-stage format to an explicit v2 graph."""
    graph = new_graph(_text(value.get("name")) or "Imported workflow")
    graph["settings"] = {
        "aspect_ratio": (value.get("aspect_ratio") or "source"),
        "unload_on_model_change": bool(value.get("unload_on_model_change", False)),
        "unload_around_heavy_processing": bool(value.get("unload_around_heavy_processing", False)),
    }
    stages = [item for item in value.get("stages") or [] if isinstance(item, dict)]
    input_node = _legacy_input_node(graph)
    stage_nodes: dict[str, dict[str, Any]] = {}
    stage_indexes: dict[str, int] = {}
    for index, stage in enumerate(stages):
        stage_type = _text(stage.get("stage_type")) or "generation"
        if stage_type == "generation":
            settings = copy.deepcopy(stage.get("settings") or {})
            output_kind = "image" if settings.get("image_mode") else "video"
            node_type = {
                "image": "inpaint_image" if stage.get("source_mode") == "inpaint" else "generate_image",
                "video": "generate_video",
            }[output_kind]
            params = {"model_type": settings.get("model_type") or settings.get("base_model_type", ""), "settings": settings}
            params.update({key: copy.deepcopy(stage.get(key)) for key in ("prompt", "output_name", "reference_images", "end_frame_images", "injected_frame_images") if key in stage})
        elif stage_type == "ai_analysis":
            node_type, params = "ai_analyze", copy.deepcopy(stage.get("processor_settings") or {})
        elif stage_type == "temporal":
            node_type, params = "postprocess", {"process_type": "temporal_upsampling", **copy.deepcopy(stage.get("processor_settings") or {})}
        elif stage_type == "spatial":
            node_type, params = "postprocess", {"process_type": "spatial_upsampling", **copy.deepcopy(stage.get("processor_settings") or {})}
        elif stage_type == "soundtrack":
            node_type, params = "postprocess", {"process_type": "soundtrack", **copy.deepcopy(stage.get("processor_settings") or {})}
        elif stage_type == "concatenate":
            node_type, params = "ffmpeg_concat", {}
        else:
            node_type, params = "postprocess", copy.deepcopy(stage.get("processor_settings") or {})
        node = new_node(node_type, stage.get("name") or None, x=340 + index * 300, y=80 + (index % 3) * 180, params=params)
        node["id"] = _text(stage.get("id")) or node["id"]
        node["enabled"] = bool(stage.get("enabled", True))
        node["params"]["legacy_stage_type"] = stage_type
        node["params"]["output_name"] = stage.get("output_name") or f"output_{index + 1}"
        known_stage_fields = {
            "id", "name", "stage_type", "output_name", "input_source", "input_transform", "concat_sources",
            "processor_settings", "enabled", "source_mode", "resolution_mode", "resolution_tier", "anchor_mode",
            "reference_images", "reference_sources", "input_references", "end_frame_source", "end_frame_images",
            "injected_frame_source", "injected_frame_images", "injected_frame_positions", "settings",
        }
        unknown_stage_fields = {key: copy.deepcopy(value) for key, value in stage.items() if key not in known_stage_fields}
        if unknown_stage_fields or stage.get("settings"):
            node["params"]["raw_settings"] = {
                "unknown_stage_fields": unknown_stage_fields,
                "legacy_settings": copy.deepcopy(stage.get("settings") or {}),
            }
        if unknown_stage_fields:
            node["params"]["warning"] = "Imported fields were preserved in params.raw_settings."
        graph["nodes"].append(node)
        stage_nodes[node["id"]] = node
        stage_indexes[node["id"]] = index
    def connect(source_node: str, source_port: str, target_node: str, target_port: str, type_name: str = ""):
        graph["edges"].append({"id": uuid.uuid4().hex, "source": {"node": source_node, "port": source_port}, "target": {"node": target_node, "port": target_port}, "type": type_name})
    for index, stage in enumerate(stages):
        node = stage_nodes.get(_text(stage.get("id")))
        if not node:
            continue
        source = _text(stage.get("input_source"))
        source_node = input_node["id"] if source in {"", "original", "previous"} and index == 0 else ""
        if source.startswith("stage:"):
            source_node = source[6:]
        elif source in {"", "original", "previous"}:
            prior = [item for item in stages[:index] if item.get("enabled", True)]
            source_node = _text(prior[-1].get("id")) if prior else input_node["id"]
        if source_node and source_node in stage_nodes:
            connect(source_node, "output", node["id"], "video" if "video" in node["type"] else "image")
        elif source_node == input_node["id"]:
            connect(source_node, "media", node["id"], "video" if "video" in node["type"] else "image", ANY_MEDIA)
        if stage.get("stage_type") == "concatenate":
            for source_value in stage.get("concat_sources") or []:
                source_id = _text(source_value).removeprefix("stage:")
                if source_id in stage_nodes:
                    connect(source_id, "output", node["id"], "videos", VIDEO)
        for source_value in stage.get("reference_sources") or []:
            source_id = _text(source_value).split(":", 1)[-1]
            if source_id in stage_nodes:
                connect(source_id, "output", node["id"], "references", IMAGE)
    graph["warnings"].append("Imported from v1. Review model ports and legacy settings before running.")
    return normalize_graph(graph)
