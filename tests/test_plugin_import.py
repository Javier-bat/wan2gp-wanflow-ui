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

    def test_prompt_enhancer_resets_before_and_after_model_switch(self):
        runner = importlib.import_module("wan2gp-wanflow-ui.graph_runner")
        calls = []

        class Plugin:
            def reset_prompt_enhancer(self):
                calls.append("reset")

            def reset_prompt_enhancer_if_requested(self):
                calls.append("reset_if_requested")

            def release_model(self):
                calls.append("release_model")

            def get_model_def(self, _model_type):
                return {"metadata": {"main_output": ["video"]}}

            def get_prompt_enhancer_choices(self, *_args):
                return [("Text", "T")], "T", {}

            def exec_prompt_enhancer_engine(self, *_args, **_kwargs):
                calls.append("enhancer")
                return ["enhanced prompt"]

        node = {
            "title": "Prompt Enhancer",
            "params": {"model_type": "test_video", "mode": "T", "target_media": "video"},
        }
        variables = {}
        answer = runner._submit_prompt_enhancer(
            Plugin(), {}, node, {"prompt": ["hello"], "images": []}, variables, None, None
        )
        self.assertEqual(answer, "enhanced prompt")
        self.assertEqual(variables["enhanced_prompt"], "enhanced prompt")
        self.assertEqual(
            calls,
            ["reset", "reset_if_requested", "release_model", "enhancer", "reset", "reset_if_requested"],
        )

    def test_prompt_enhancer_cleans_up_after_cuda_failure(self):
        runner = importlib.import_module("wan2gp-wanflow-ui.graph_runner")
        calls = []

        class Plugin:
            def reset_prompt_enhancer(self):
                calls.append("reset")

            def reset_prompt_enhancer_if_requested(self):
                calls.append("reset_if_requested")

            def release_model(self):
                calls.append("release_model")

            def get_model_def(self, _model_type):
                return {"metadata": {"main_output": ["video"]}}

            def get_prompt_enhancer_choices(self, *_args):
                return [("Text", "T")], "T", {}

            def exec_prompt_enhancer_engine(self, *_args, **_kwargs):
                raise RuntimeError("CUDA error: an illegal memory access was encountered")

        node = {
            "title": "Prompt Enhancer",
            "params": {"model_type": "test_video", "mode": "T", "target_media": "video"},
        }
        with self.assertRaisesRegex(RuntimeError, "Restart Wan2GP"):
            runner._submit_prompt_enhancer(
                Plugin(), {}, node, {"prompt": ["hello"], "images": []}, {}, None, None
            )
        self.assertEqual(calls[-2:], ["reset", "reset_if_requested"])

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

    def test_catalog_does_not_expose_control_definitions_as_defaults(self):
        module = importlib.import_module("wan2gp-wanflow-ui.node_registry")
        catalog = module.build_catalog({
            "test_video": {
                "flow_shift": True,
                "attention_sparsity": {"label": "Start Tau", "start": 0, "end": 4},
                "image_refs_relative_size": {"min": 50, "max": 400},
                "metadata": {"main_output": ["video"]},
            }
        })
        model = catalog["models"][0]
        self.assertNotIn("flow_shift", model["defaults"])
        self.assertNotIn("attention_sparsity", model["defaults"])
        self.assertNotIn("image_refs_relative_size", model["defaults"])

    def test_runner_discards_stale_control_metadata_and_applies_model_defaults(self):
        runner = importlib.import_module("wan2gp-wanflow-ui.graph_runner")
        settings = {
            "flow_shift": True,
            "attention_sparsity": {"label": "Start Tau"},
            "image_refs_relative_size": {"min": 50},
        }
        runner._sanitize_native_settings(settings)
        self.assertEqual(settings, {})

        class Defaults:
            @staticmethod
            def get_default_settings(_model_type):
                return {"flow_shift": 12, "num_inference_steps": 6, "prompt": "ignored"}

        runner._apply_model_defaults(settings, Defaults(), "test_video")
        self.assertEqual(settings["flow_shift"], 12)
        self.assertEqual(settings["num_inference_steps"], 6)
        self.assertNotIn("prompt", settings)

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

    def test_catalog_groups_configuration_nodes_and_lora_stack(self):
        module = importlib.import_module("wan2gp-wanflow-ui.node_registry")
        catalog = module.build_catalog({"test": {"name": "Test", "metadata": {"main_output": ["image"]}}}, loras={"test": ["style.safetensors"]})
        nodes = {item["type"]: item for item in catalog["nodes"]}
        self.assertEqual(nodes["resolution_config"]["category"], "Configuration")
        self.assertEqual(nodes["lora_stack"]["category"], "Configuration")
        self.assertEqual(nodes["ffmpeg_concat"]["category"], "FFmpeg")
        self.assertIn("lora_stack", nodes)

    def test_catalog_exposes_model_specific_prompt_enhancer_modes(self):
        module = importlib.import_module("wan2gp-wanflow-ui.node_registry")
        catalog = module.build_catalog({
            "minimax": {
                "name": "MiniMax Ref2VA",
                "prompt_enhancer_def": {
                    "selection": ["T", "TI"],
                    "labels": {"TV": "Text prompt", "TIV": "Text + reference image"},
                    "default": "",
                },
                "metadata": {"main_output": ["video"]},
            },
        })
        model = catalog["models"][0]
        self.assertEqual([item["value"] for item in model["prompt_enhancer"]["video"]["choices"]], ["T", "TI"])
        self.assertEqual(model["prompt_enhancer"]["image"]["choices"], [])
        self.assertEqual(
            [item["value"] for item in model["prompt_enhancer"]["audio"]["choices"]],
            ["T", "TI"],
        )

    def test_catalog_resolution_tiers_come_from_runtime(self):
        module = importlib.import_module("wan2gp-wanflow-ui.node_registry")
        catalog = module.build_catalog({})
        tiers = [item["value"] for item in catalog["resolution_tiers"]]
        self.assertIn("540p", tiers)
        self.assertNotIn("576p", tiers)
        self.assertNotIn("640p", tiers)

    def test_catalog_exposes_base_models_and_finetune_defaults(self):
        module = importlib.import_module("wan2gp-wanflow-ui.node_registry")
        catalog = module.build_catalog({
            "wan_base": {"name": "Wan Base", "architecture": "wan_base", "metadata": {"main_output": ["video"]}},
            "wan_fast": {"name": "Wan Fast Finetune", "architecture": "wan_base", "base_model_type": "wan_base", "num_inference_steps": 8, "sample_solvers": ["euler"], "metadata": {"main_output": ["video"]}},
        }, loras={"wan_fast": ["style.safetensors"]})
        models = {item["model_type"]: item for item in catalog["models"]}
        self.assertEqual(models["wan_fast"]["base_model_type"], "wan_base")
        self.assertEqual(models["wan_fast"]["defaults"]["num_inference_steps"], 8)
        self.assertEqual(catalog["base_models"][0]["model_type"], "wan_base")
        self.assertEqual(catalog["base_models"][0]["loras"], ["style.safetensors"])

    def test_catalog_aggregates_loras_by_wan2gp_family(self):
        module = importlib.import_module("wan2gp-wanflow-ui.node_registry")
        catalog = module.build_catalog(
            {
                "ltx_dev": {"name": "LTX Dev", "architecture": "ltx2_22B", "lora_family": "ltx", "lora_family_name": "LTX Video", "metadata": {"main_output": ["video"]}},
                "ltx_edit": {"name": "LTX Edit", "architecture": "ltx2_22B_edit_anything", "lora_family": "ltx", "lora_family_name": "LTX Video", "metadata": {"main_output": ["video"]}},
            },
            loras={"ltx_dev": ["dev/style.safetensors"], "ltx_edit": ["edit/style.safetensors"]},
        )
        self.assertEqual(len(catalog["base_models"]), 1)
        self.assertEqual(catalog["base_models"][0]["name"], "LTX Video")
        self.assertEqual(catalog["base_models"][0]["loras"], ["dev/style.safetensors", "edit/style.safetensors"])

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

    def test_connected_configuration_overrides_generation_values(self):
        runner = importlib.import_module("wan2gp-wanflow-ui.graph_runner")
        settings = {}
        params = {"resolution_tier": "720p", "cfg_scale": 1}
        runner._apply_connected_configuration(settings, params, {
            "resolution_config": {"resolution_tier": "480p", "aspect_ratio": "16:9"},
            "sampling_config": {"cfg_scale": 5, "sample_solver": "euler"},
            "lora_stack": {"model_type": "test_model", "activated_loras": ["style.safetensors"], "loras_multipliers": "0.8"},
        }, "test_model")
        self.assertEqual(params["resolution_tier"], "480p")
        self.assertEqual(params["aspect_ratio"], "16:9")
        self.assertEqual(settings["cfg_scale"], 5)
        self.assertEqual(settings["sample_solver"], "euler")
        self.assertEqual(settings["activated_loras"], ["style.safetensors"])

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
