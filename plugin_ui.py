from __future__ import annotations

import base64
import copy
import html
import json
import mimetypes
import queue
import tempfile
import threading
from pathlib import Path
from typing import Any

import gradio as gr

from postprocessing.catalog import query_processes

from .graph_model import LEGACY_SCHEMA, SCHEMA, graph_for_json, migrate_v1_workflow, new_graph, new_node, normalize_graph
from .graph_runner import run_graph
from .block_store import BlockStore
from .graph_store import GraphStore
from .media_gallery import RuntimeMediaGallery
from . import media_ops
from .node_registry import build_catalog


PLUGIN_ROOT = Path(__file__).resolve().parent
WORKFLOW_ROOT = PLUGIN_ROOT / "workflows"
OUTPUT_ROOT = PLUGIN_ROOT / "workflow_outputs"
BLOCK_ROOT = PLUGIN_ROOT / "blocks"
ASSET_ROOT = PLUGIN_ROOT / "assets"


def _starter_graph() -> dict[str, Any]:
    graph = new_graph()
    text = new_node("text", "Prompt", x=70, y=120, params={"text": "A cinematic portrait, detailed lighting"})
    image = new_node("generate_image", "Generate Image", x=390, y=90, params={"settings": {}, "model_type": ""})
    graph["nodes"] = [text, image]
    graph["edges"] = [{"id": "starter-prompt", "source": {"node": text["id"], "port": "text"}, "target": {"node": image["id"], "port": "prompt"}, "type": "TEXT"}]
    return graph


def _model_defs(plugin) -> dict[str, dict[str, Any]]:
    try:
        records = plugin.list_model_defs() or []
    except Exception:
        records = []
    result = {}
    for record in records:
        if isinstance(record, dict):
            key = str(record.get("model_type") or record.get("architecture") or "").strip()
            if key:
                result[key] = record
    return result


def _processes() -> list[dict[str, Any]]:
    seen = set()
    result = []
    for media_type in ("image", "video", "audio"):
        try:
            values = query_processes(media_type, enabled_only=False)
        except Exception:
            values = []
        for value in values:
            key = (str(value.get("type") or ""), str(value.get("id") or ""))
            if key not in seen:
                seen.add(key)
                result.append(value)
    return result


def _loras(plugin, model_defs: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """Discover the same per-model LoRA files exposed by Wan2GP's UI."""
    result: dict[str, list[str]] = {}
    get_lora_dir = getattr(plugin, "get_lora_dir", None)
    if not callable(get_lora_dir):
        return result
    for model_type, model_def in model_defs.items():
        if model_def.get("no_lora", False):
            continue
        try:
            root = Path(get_lora_dir(model_type))
            if not root.is_dir():
                continue
            result[model_type] = sorted(
                {
                    str(path.relative_to(root)).replace("\\", "/")
                    for suffix in ("*.safetensors", "*.sft")
                    for path in root.rglob(suffix)
                    if path.is_file()
                },
                key=str.casefold,
            )
        except Exception:
            continue
    return result


def _runtime_preview(path: str, slot: int) -> dict[str, Any]:
    """Return a small, transient browser preview without storing it in a graph."""
    value = {"slot": int(slot), "name": Path(path).name, "kind": "file", "src": ""}
    suffix = Path(path).suffix.lower()
    mime = mimetypes.guess_type(path)[0] or ""
    try:
        if mime.startswith("image/") and Path(path).stat().st_size <= 4 * 1024 * 1024:
            payload = base64.b64encode(Path(path).read_bytes()).decode("ascii")
            value.update(kind="image", src=f"data:{mime};base64,{payload}")
        elif mime.startswith("video/"):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
                poster = Path(handle.name)
            try:
                media_ops.extract_frame(path, poster, None)
                payload = base64.b64encode(poster.read_bytes()).decode("ascii")
                value.update(kind="video", src=f"data:image/png;base64,{payload}")
            finally:
                poster.unlink(missing_ok=True)
        elif mime.startswith("audio/") or suffix in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
            value["kind"] = "audio"
    except Exception:
        value["kind"] = "file"
    return value


def _file_path(value) -> str:
    if isinstance(value, (str, Path)):
        return str(value).strip()
    if isinstance(value, dict):
        for key in ("path", "name", "file_name", "orig_name"):
            path = value.get(key)
            if path:
                return str(path).strip()
        return ""
    if isinstance(value, (list, tuple)) and value:
        return _file_path(value[0])
    return str(
        getattr(value, "path", None)
        or getattr(value, "name", None)
        or getattr(value, "file_name", None)
        or ""
    ).strip()


def _runtime_previews(files) -> str:
    values = files if isinstance(files, (list, tuple)) else ([files] if files else [])
    previews = []
    for slot, value in enumerate(values):
        path = _file_path(value)
        if path and Path(path).is_file():
            previews.append(_runtime_preview(path, slot))
    return json.dumps(previews, ensure_ascii=False)


def _gallery_values(files) -> list[str]:
    values = files if isinstance(files, (list, tuple)) else ([files] if files else [])
    return [path for value in values if (path := _file_path(value)) and Path(path).is_file()]


def _runtime_items(value):
    if isinstance(value, dict):
        return value.get("items") or []
    return value or []


def _node_output_preview(result: dict[str, Any]) -> dict[str, Any] | None:
    """Make a small transient preview for one completed node output."""
    for port, value in result.items():
        if not isinstance(value, str):
            continue
        path = Path(value)
        if path.is_file():
            preview = _runtime_preview(str(path), -1)
            preview["port"] = str(port)
            return preview
        if port in {"text", "metadata"} and value:
            return {"kind": "text", "port": str(port), "text": value[:4000]}
    return None


def _iframe(plugin, graph: dict[str, Any], status: str = "Ready") -> str:
    template = (ASSET_ROOT / "editor.html").read_text(encoding="utf-8")
    css = (ASSET_ROOT / "editor.css").read_text(encoding="utf-8")
    js = (ASSET_ROOT / "editor.js").read_text(encoding="utf-8")
    model_defs = _model_defs(plugin)
    catalog = build_catalog(model_defs, _processes(), _loras(plugin, model_defs))
    catalog["blocks"] = BlockStore(BLOCK_ROOT).catalog()
    graph_payload = base64.b64encode(json.dumps(graph_for_json(graph), ensure_ascii=False).encode("utf-8")).decode("ascii")
    catalog_payload = base64.b64encode(json.dumps(catalog, ensure_ascii=False).encode("utf-8")).decode("ascii")
    status_payload = html.escape(str(status or "Ready").replace("'", "\\'"))
    content = template.replace("/* WORKFLOWS_V2_CSS */", css)
    content = content.replace("/* WORKFLOWS_V2_JS */", js)
    content = content.replace("/* WORKFLOWS_V2_GRAPH */", graph_payload)
    content = content.replace("/* WORKFLOWS_V2_CATALOG */", catalog_payload)
    content = content.replace("/* WORKFLOWS_V2_STATUS */", status_payload)
    # A sandboxed data: iframe is blocked by Chromium in the Gradio page.
    # srcdoc keeps the editor self-contained while allowing its scripts to run.
    srcdoc = html.escape(content, quote=True)
    return (
        "<iframe id='wan2gp-wanflow-ui-iframe' title='Wan2GP WanFlow UI' "
        "sandbox='allow-scripts allow-pointer-lock allow-downloads' "
        "style='width:100%;height:760px;border:0;border-radius:10px;display:block' "
        f"srcdoc='{srcdoc}'></iframe>"
    )


def _bridge() -> str:
    return r"""
    (() => {
      const eventName = "WAN2GP_WORKFLOWS_V2";
      function appRoot(){return window.gradioApp ? window.gradioApp() : document.querySelector("gradio-app")?.shadowRoot || document;}
      function field(id){return appRoot().querySelector(`#${id} textarea, #${id} input`);}
      function button(id){return appRoot().querySelector(`#${id} button, #${id}`);}
      function dispatch(id,value){const element=field(id);if(!element)return false;element.value=value;element.dispatchEvent(new Event("input",{bubbles:true}));element.dispatchEvent(new Event("change",{bubbles:true}));return true;}
      function postRuntimePreviews(value){
        const iframe=document.getElementById("wan2gp-wanflow-ui-iframe");
        if(!iframe?.contentWindow)return;
        let previews=[];try{previews=JSON.parse(value||"[]");}catch(_error){}
        iframe.contentWindow.postMessage({type:eventName,action:"runtime_previews",previews},"*");
      }
      function postNodeStatus(value){
        const iframe=document.getElementById("wan2gp-wanflow-ui-iframe");
        if(!iframe?.contentWindow)return;
        let status={};try{status=JSON.parse(value||"{}");}catch(_error){}
        iframe.contentWindow.postMessage({type:eventName,action:"node_status",status},"*");
      }
      function wireRuntimePreviews(){
        const element=field("w2gp_v2_runtime_previews");
        if(!element)return;
        const iframe=document.getElementById("wan2gp-wanflow-ui-iframe");
        if(iframe && element.dataset.w2gpPreviewIframe!==String(iframe)){
          element.dataset.w2gpPreviewIframe=String(iframe);
          iframe.addEventListener("load",()=>postRuntimePreviews(element.value));
          postRuntimePreviews(element.value);
        }
        if(!element.dataset.w2gpPreviewBound){
          element.dataset.w2gpPreviewBound="1";
          element.addEventListener("input",()=>postRuntimePreviews(element.value));
          element.addEventListener("change",()=>postRuntimePreviews(element.value));
        }
        if(element.dataset.w2gpPreviewValue!==element.value){
          element.dataset.w2gpPreviewValue=element.value;
          postRuntimePreviews(element.value);
        }
      }
      function wireNodeStatus(){
        const element=field("w2gp_v2_node_status");
        if(!element)return;
        const iframe=document.getElementById("wan2gp-wanflow-ui-iframe");
        if(iframe && element.dataset.w2gpStatusIframe!==String(iframe)){
          element.dataset.w2gpStatusIframe=String(iframe);
          iframe.addEventListener("load",()=>postNodeStatus(element.value));
          postNodeStatus(element.value);
        }
        if(element.dataset.w2gpNodeStatusValue!==element.value){
          element.dataset.w2gpNodeStatusValue=element.value;
          postNodeStatus(element.value);
        }
      }
      window.addEventListener("message", event => {
        const data=event?.data;
        if(!data || data.type!==eventName)return;
        if(data.action==="runtime_previews")return;
        const payload=JSON.stringify(data.graph || {});
        if(!dispatch("w2gp_v2_graph_payload",payload))return;
        const target={save:"w2gp_v2_save_trigger",save_block:"w2gp_v2_save_block_trigger",delete_block:"w2gp_v2_delete_block_trigger",run:"w2gp_v2_run_trigger",apply:"w2gp_v2_apply_trigger"}[data.action] || "w2gp_v2_apply_trigger";
        // Let Gradio observe the input event before its click handler reads
        // the component state. This matters for Save/Run in queued layouts.
        window.setTimeout(()=>button(target)?.click(),40);
      });
      setInterval(()=>{wireRuntimePreviews();wireNodeStatus();},500);
      wireRuntimePreviews();
      wireNodeStatus();
      console.log("[Wan2GP WanFlow UI] bridge ready");
    })();
    """


def create_config_ui(plugin, api_session):
    store = GraphStore(WORKFLOW_ROOT)
    initial = _starter_graph()

    def parse_payload(payload: str, fallback=None):
        try:
            value = json.loads(payload or "{}")
        except (TypeError, ValueError) as exc:
            raise gr.Error(f"Invalid workflow JSON: {exc}") from exc
        return normalize_graph(value if isinstance(value, dict) else fallback or initial)

    def apply_payload(payload):
        graph = parse_payload(payload)
        return graph, json.dumps(graph_for_json(graph), ensure_ascii=False), _iframe(plugin, graph), "Graph updated"

    def save_payload(payload, name):
        graph = parse_payload(payload)
        saved, path = store.save(graph, name)
        return saved, json.dumps(graph_for_json(saved), ensure_ascii=False), _iframe(plugin, saved, f"Saved {path.name}"), gr.update(choices=store.choices(), value=path.stem), f"Saved {path.name}"

    def save_block_payload(payload):
        graph = parse_payload(payload)
        group_id = str((graph.get("ui") or {}).get("selected_group_id") or "")
        group = next((item for item in graph.get("groups") or [] if item.get("id") == group_id), None)
        if not group:
            raise gr.Error("Select a group before saving it as a reusable block.")
        block_store = BlockStore(BLOCK_ROOT)
        path = block_store.save_group(graph, group_id, group.get("title"))
        graph.setdefault("ui", {}).pop("selected_group_id", None)
        return graph, json.dumps(graph_for_json(graph), ensure_ascii=False), _iframe(plugin, graph, f"Saved block {path.name}"), f"Saved block {path.name}"

    def delete_block_payload(payload):
        graph = parse_payload(payload)
        block_id = str((graph.get("ui") or {}).get("selected_block_id") or "").strip()
        if not block_id:
            raise gr.Error("Select a reusable block before deleting it.")
        try:
            path = BlockStore(BLOCK_ROOT).delete(block_id)
        except FileNotFoundError as exc:
            raise gr.Error(f"Reusable block not found: {block_id}") from exc
        graph.setdefault("ui", {}).pop("selected_block_id", None)
        return graph, json.dumps(graph_for_json(graph), ensure_ascii=False), _iframe(plugin, graph, f"Deleted block {path.stem}"), f"Deleted block {path.stem}"

    def load_payload(slug):
        if not slug:
            raise gr.Error("Select a workflow to load.")
        graph = store.load(slug)
        return graph, json.dumps(graph_for_json(graph), ensure_ascii=False), _iframe(plugin, graph, f"Loaded {slug}"), graph.get("name", slug), f"Loaded {slug}"

    def delete_payload(slug):
        if slug:
            store.delete(slug)
        graph = new_graph()
        return graph, json.dumps(graph_for_json(graph), ensure_ascii=False), _iframe(plugin, graph), gr.update(choices=store.choices(), value=None), "Deleted workflow"

    def import_payload(file_path):
        if not file_path:
            raise gr.Error("Choose a WanFlow workflow JSON file.")
        path = Path(str(file_path))
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise gr.Error(f"Could not read workflow JSON: {exc}") from exc
        if payload.get("schema") in {SCHEMA, LEGACY_SCHEMA}:
            graph = normalize_graph(payload)
        else:
            graph = migrate_v1_workflow(payload)
        graph["name"] = f"{graph.get('name') or path.stem} v2"
        return graph, json.dumps(graph_for_json(graph), ensure_ascii=False), _iframe(plugin, graph, f"Imported {path.name}"), graph["name"], f"Imported {path.name}"

    def run_payload(main_state, payload, runtime_media):
        graph = parse_payload(payload)
        events = queue.Queue()
        states = {
            node["id"]: {"state": "queued" if node.get("enabled", True) else "skipped"}
            for node in graph.get("nodes") or []
        }
        disabled_groups = {
            node_id
            for group in graph.get("groups") or []
            if not group.get("enabled", True)
            for node_id in group.get("node_ids") or []
        }
        for node_id in disabled_groups:
            states.setdefault(node_id, {})["state"] = "skipped"
        current_node = {"id": None}

        def progress(status, value):
            if status:
                events.put(("progress", f"{status} ({float(value or 0):.0f}%)"))

        def node_event(node_id, state, result):
            events.put(("node", node_id, state, result))

        # GradioWanGPSession keeps the live request and session hash in a
        # thread-local. The event worker must inherit that binding before it
        # calls Wan2GP's WebUI API.
        api_local = getattr(api_session, "_ui_local", None)
        api_context = {
            name: getattr(api_local, name, None)
            for name in ("call_id", "defer_load_queue_trigger", "bound_state", "bound_gradio_context")
        }

        def worker():
            if api_local is not None:
                for name, value in api_context.items():
                    setattr(api_local, name, value)
            try:
                outputs, variables = run_graph(
                    plugin,
                    api_session,
                    main_state if isinstance(main_state, dict) else {},
                    graph,
                    [_file_path(value) for value in _runtime_items(runtime_media) if _file_path(value)],
                    output_root=OUTPUT_ROOT,
                    callback=progress,
                    node_callback=node_event,
                )
                events.put(("complete", outputs, variables))
            except Exception as exc:
                events.put(("error", exc))

        threading.Thread(target=worker, name="wan2gp-v2-workflow", daemon=True).start()
        yield "Running workflow...", gr.update(), json.dumps({"nodes": states}, ensure_ascii=False)

        titles = {node["id"]: node.get("title") or node["id"] for node in graph.get("nodes") or []}
        while True:
            event = events.get()
            kind = event[0]
            if kind == "node":
                _, node_id, state, result = event
                states.setdefault(node_id, {})["state"] = state
                if state == "running":
                    current_node["id"] = node_id
                elif state == "done" and result:
                    preview = _node_output_preview(result)
                    if preview:
                        states[node_id]["output"] = preview
                label = titles.get(node_id, node_id)
                message = f"Running: {label}" if state == "running" else f"Finished: {label}"
                yield message, gr.update(), json.dumps({"nodes": states}, ensure_ascii=False)
            elif kind == "progress":
                yield event[1], gr.update(), json.dumps({"nodes": states}, ensure_ascii=False)
            elif kind == "complete":
                _, outputs, variables = event
                detail = f"Complete: {len(outputs)} file(s)"
                if variables:
                    detail += f", {len(variables)} text variable(s)"
                yield detail, gr.update(value=_gallery_values(outputs)), json.dumps({"nodes": states}, ensure_ascii=False)
                return
            else:
                exc = event[1]
                if current_node["id"]:
                    states.setdefault(current_node["id"], {})["state"] = "error"
                detail = f"Stopped: {exc}" if isinstance(exc, InterruptedError) else f"Error: {exc}"
                yield detail, gr.update(value=[]), json.dumps({"nodes": states}, ensure_ascii=False)
                return

    def update_runtime_previews(runtime_files):
        return _runtime_previews(_runtime_items(runtime_files))

    with gr.Column(elem_id="w2gp-workflows-v2"):
        gr.HTML("<style>#w2gp-workflows-v2{padding:8px}#w2gp-v2-graph{padding:0!important}#w2gp-v2-hidden{display:none!important}</style>")
        graph_state = gr.State(initial)
        graph_payload = gr.Textbox(value=json.dumps(graph_for_json(initial)), visible=False, elem_id="w2gp_v2_graph_payload")
        with gr.Row():
            workflow_name = gr.Textbox(label="Workflow name", value=initial["name"], scale=3)
            saved = gr.Dropdown(label="Saved v2 workflows", choices=store.choices(), filterable=True, scale=3)
            load_btn = gr.Button("Load", scale=1)
            delete_btn = gr.Button("Delete", scale=1)
        with gr.Row():
            with gr.Column(scale=4):
                runtime_gallery = RuntimeMediaGallery().mount()
            with gr.Column(scale=3):
                import_file = gr.File(label="Import workflow JSON", file_types=[".json"], type="filepath")
                import_btn = gr.Button("Import")
        runtime_preview_payload = gr.Textbox(value="[]", visible=False, elem_id="w2gp_v2_runtime_previews")
        node_status_payload = gr.Textbox(value='{"nodes":{}}', visible=False, elem_id="w2gp_v2_node_status")
        graph_html = gr.HTML(_iframe(plugin, initial), elem_id="w2gp-v2-graph")
        status = gr.Markdown("Ready")
        output_gallery = gr.Gallery(label="Outputs", columns=5, height=280, object_fit="contain", preview=True, show_fullscreen_button=True, interactive=False)
        with gr.Row(elem_id="w2gp-v2-hidden"):
            apply_trigger = gr.Button("apply", elem_id="w2gp_v2_apply_trigger")
            save_trigger = gr.Button("save", elem_id="w2gp_v2_save_trigger")
            save_block_trigger = gr.Button("save block", elem_id="w2gp_v2_save_block_trigger")
            delete_block_trigger = gr.Button("delete block", elem_id="w2gp_v2_delete_block_trigger")
            run_trigger = gr.Button("run", elem_id="w2gp_v2_run_trigger")

        editor_outputs = [graph_state, graph_payload, graph_html, status]
        apply_trigger.click(apply_payload, inputs=graph_payload, outputs=editor_outputs, queue=False)
        save_trigger.click(save_payload, inputs=[graph_payload, workflow_name], outputs=[graph_state, graph_payload, graph_html, saved, status], queue=False)
        save_block_trigger.click(save_block_payload, inputs=graph_payload, outputs=[graph_state, graph_payload, graph_html, status], queue=False)
        delete_block_trigger.click(delete_block_payload, inputs=graph_payload, outputs=[graph_state, graph_payload, graph_html, status], queue=False)
        load_btn.click(load_payload, inputs=saved, outputs=[graph_state, graph_payload, graph_html, workflow_name, status], queue=False)
        delete_btn.click(delete_payload, inputs=saved, outputs=[graph_state, graph_payload, graph_html, saved, status], queue=False)
        import_btn.click(import_payload, inputs=import_file, outputs=[graph_state, graph_payload, graph_html, workflow_name, status], queue=False)
        run_trigger.click(run_payload, inputs=[plugin.state, graph_payload, runtime_gallery.state], outputs=[status, output_gallery, node_status_payload], queue=True)
        runtime_gallery.state.change(update_runtime_previews, inputs=runtime_gallery.state, outputs=runtime_preview_payload, queue=False)
    return graph_state
