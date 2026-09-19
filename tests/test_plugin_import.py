from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import gradio as gr


PLUGINS_ROOT = str(Path(__file__).resolve().parents[2])
if PLUGINS_ROOT not in sys.path:
    sys.path.insert(0, PLUGINS_ROOT)


class PluginImportTests(unittest.TestCase):
    def test_plugin_module_imports_with_hyphenated_directory_name(self):
        module = importlib.import_module("wan2gp-wanflow-ui.plugin")
        self.assertEqual(module.PlugIn_Id, "wan2gp_wanflow_ui")
        plugin = module.ConfigTabPlugin()
        plugin.setup_ui()
        self.assertIn("wan2gp_wanflow_ui", plugin.tabs)

    def test_ui_constructor_builds_inside_gradio(self):
        module = importlib.import_module("wan2gp-wanflow-ui.plugin")
        plugin = module.ConfigTabPlugin()
        plugin.list_model_defs = lambda: []
        plugin.state = gr.State({})
        with gr.Blocks():
            plugin.create_config_ui(None)

    def test_graph_store_round_trip(self):
        model = importlib.import_module("wan2gp-wanflow-ui.graph_model")
        store_module = importlib.import_module("wan2gp-wanflow-ui.graph_store")
        graph = model.new_graph("Round trip")
        graph["nodes"] = [model.new_node("text", params={"text": "hello"})]
        with tempfile.TemporaryDirectory() as directory:
            store = store_module.GraphStore(directory)
            _, path = store.save(graph, "Round trip")
            loaded = store.load(path.stem)
            self.assertEqual(loaded["name"], "Round trip")
            self.assertEqual(loaded["nodes"][0]["params"]["text"], "hello")

    def test_block_store_saves_selected_group_as_own_schema(self):
        model = importlib.import_module("wan2gp-wanflow-ui.graph_model")
        block_module = importlib.import_module("wan2gp-wanflow-ui.block_store")
        graph = model.new_graph("Blocks")
        first = model.new_node("text", "Prompt", params={"text": "hello"})
        second = model.new_node("text", "Prompt 2", params={"text": "world"})
        graph["nodes"] = [first, second]
        graph["edges"] = [{"id": "edge", "source": {"node": first["id"], "port": "text"}, "target": {"node": second["id"], "port": "instruction"}, "type": "TEXT"}]
        graph["groups"] = [{"id": "group", "title": "Reusable prompts", "node_ids": [first["id"], second["id"]]}]
        with tempfile.TemporaryDirectory() as directory:
            store = block_module.BlockStore(directory)
            path = store.save_group(graph, "group")
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], block_module.SCHEMA)
            self.assertEqual(len(payload["nodes"]), 2)
            self.assertEqual(len(store.catalog()), 1)
            store.delete(path.stem)
            self.assertFalse(path.exists())

    def test_runtime_gallery_removes_only_selected_item(self):
        module = importlib.import_module("wan2gp-wanflow-ui.media_gallery")
        gallery = module.RuntimeMediaGallery()
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.jpg"
            second = Path(directory) / "second.jpg"
            first.touch()
            second.touch()
            values = [{"path": str(first)}, (str(second), "caption")]
            self.assertEqual(gallery._items(values), [str(first), str(second)])
            _update, state = gallery._remove(
                {"items": [str(first), str(second)], "selected": 0},
                values,
            )
            self.assertEqual(state["items"], [str(second)])
            self.assertEqual(state["selected"], 0)

    def test_catalog_exposes_model_native_video_controls(self):
        module = importlib.import_module("wan2gp-wanflow-ui.node_registry")
        catalog = module.build_catalog({
            "test_video": {
                "name": "Test Video",
                "flow_shift": True,
                "sliding_window": True,
                "sliding_window_defaults": {"window_min": 17, "window_max": 129, "window_step": 4},
                "first_block_cache": True,
                "sample_solvers": [("Euler", "euler"), ("Test", "test")],
                "custom_attention_modes": {"test_attn": {"label": "Test attention", "supports_sparsity": True}},
                "attention_sparsity": {"start": 0.0, "end": 2.0, "inc": 0.1},
                "metadata": {
                    "main_output": ["video"],
                    "outputs": ["video"],
                    "inputs": ["text", "video"],
                    "media_inputs": {"video": {"continue": True}},
                },
            }
        })
        model = next(item for item in catalog["models"] if item["model_type"] == "test_video")
        settings = {item["key"]: item for item in model["native_settings"]}
        self.assertIn("sliding_window_size", settings)
        self.assertIn("sliding_window_overlap", settings)
        self.assertIn("flow_shift", settings)
        self.assertIn("sample_solver", settings)
        self.assertIn("skip_steps_cache_type", settings)
        self.assertIn("override_attention", settings)
        self.assertEqual(settings["sample_solver"]["choices"][1]["value"], "test")

    def test_catalog_keeps_loras_scoped_to_the_selected_model(self):
        module = importlib.import_module("wan2gp-wanflow-ui.node_registry")
        catalog = module.build_catalog(
            {
                "minimax_test": {"name": "MiniMax", "metadata": {"main_output": ["video"]}},
                "flux_test": {"name": "Flux", "metadata": {"main_output": ["image"]}},
            },
            loras={"minimax_test": ["minimax/style.safetensors"], "flux_test": ["flux/style.safetensors"]},
        )
        models = {item["model_type"]: item for item in catalog["models"]}
        self.assertEqual(models["minimax_test"]["loras"], ["minimax/style.safetensors"])
        self.assertEqual(models["flux_test"]["loras"], ["flux/style.safetensors"])

    def test_generation_settings_preserve_remote_lora_identifiers(self):
        model = importlib.import_module("wan2gp-wanflow-ui.graph_model")
        graph = model.new_graph("Remote LoRA")
        node = model.new_node(
            "generate_image",
            "Remote",
            params={
                "model_type": "test_model",
                "activated_loras": ["https://huggingface.co/example/style.safetensors"],
                "loras_multipliers": "0.75",
            },
        )
        graph["nodes"] = [node]
        normalized = model.normalize_graph(graph)
        params = normalized["nodes"][0]["params"]
        self.assertEqual(params["activated_loras"], ["https://huggingface.co/example/style.safetensors"])
        self.assertEqual(params["loras_multipliers"], "0.75")

    def test_injected_frame_positions_match_wangp_tokens(self):
        runtime = importlib.import_module("wan2gp-wanflow-ui.runtime_engine")
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.png"
            second = Path(directory) / "second.png"
            first.touch()
            second.touch()
            settings = runtime.bind_injected_frames(
                {"video_prompt_type": "I"},
                [str(first), str(second)],
                "1 X",
                {"image_ref_choices": {"choices": [("Inject", "F")]}},
            )
            self.assertEqual(settings["frames_positions"], "1 X")
            self.assertIn("F", settings["video_prompt_type"])


if __name__ == "__main__":
    unittest.main()
