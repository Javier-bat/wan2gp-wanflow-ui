from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("wan2gp_wanflow_ui_graph_model", ROOT / "graph_model.py")
GRAPH = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(GRAPH)


class GraphModelTests(unittest.TestCase):
    def test_topological_order_supports_branches_and_convergence(self):
        source = GRAPH.new_node("input_image", "Source")
        left = GRAPH.new_node("last_frame", "Left")
        right = GRAPH.new_node("last_frame", "Right")
        join = GRAPH.new_node("ffmpeg_concat", "Join")
        graph = GRAPH.new_graph()
        graph["nodes"] = [source, left, right, join]
        graph["edges"] = [
            {"id": "1", "source": {"node": source["id"], "port": "image"}, "target": {"node": left["id"], "port": "video"}, "type": "IMAGE"},
            {"id": "2", "source": {"node": source["id"], "port": "image"}, "target": {"node": right["id"], "port": "video"}, "type": "IMAGE"},
            {"id": "3", "source": {"node": left["id"], "port": "image"}, "target": {"node": join["id"], "port": "videos"}, "type": "IMAGE"},
            {"id": "4", "source": {"node": right["id"], "port": "image"}, "target": {"node": join["id"], "port": "videos"}, "type": "IMAGE"},
        ]
        order = GRAPH.topological_order(graph)
        self.assertEqual(order[0], source["id"])
        self.assertEqual(order[-1], join["id"])

    def test_cycle_is_rejected(self):
        one = GRAPH.new_node("text", "One")
        two = GRAPH.new_node("text", "Two")
        graph = GRAPH.new_graph()
        graph["nodes"] = [one, two]
        graph["edges"] = [
            {"id": "1", "source": {"node": one["id"], "port": "text"}, "target": {"node": two["id"], "port": "instruction"}, "type": "TEXT"},
            {"id": "2", "source": {"node": two["id"], "port": "text"}, "target": {"node": one["id"], "port": "instruction"}, "type": "TEXT"},
        ]
        with self.assertRaises(ValueError):
            GRAPH.topological_order(graph)

    def test_normalize_drops_edges_to_missing_nodes(self):
        graph = GRAPH.normalize_graph({
            "schema": GRAPH.SCHEMA,
            "nodes": [{"id": "one", "type": "text", "title": "One"}],
            "edges": [{"id": "bad", "source": {"node": "one", "port": "text"}, "target": {"node": "missing", "port": "text"}}],
        })
        self.assertEqual(graph["edges"], [])

    def test_v1_import_creates_explicit_nodes_and_edges(self):
        graph = GRAPH.migrate_v1_workflow({
            "name": "Legacy",
            "stages": [
                {"id": "first", "name": "First", "stage_type": "generation", "settings": {"model_type": "test", "image_mode": 0}, "input_source": "original"},
                {"id": "second", "name": "Second", "stage_type": "temporal", "processor_settings": {"temporal_upsampling": "rife4"}, "input_source": "stage:first"},
            ],
        })
        self.assertEqual(graph["schema"], GRAPH.SCHEMA)
        self.assertEqual(len(graph["nodes"]), 3)
        self.assertTrue(any(edge["source"]["node"] == "first" and edge["target"]["node"] == "second" for edge in graph["edges"]))

    def test_model_metadata_controls_generation_ports(self):
        node = GRAPH.new_node("generate_image", params={"model_type": "text_only"})
        model_defs = {
            "text_only": {"metadata": {"inputs": ["text"], "outputs": ["image"], "main_output": ["image"], "media_inputs": {}}}
        }
        ports = GRAPH.node_ports(node, model_defs)
        self.assertIn("prompt", ports)
        self.assertNotIn("image", ports)
        self.assertEqual(ports["output"]["type"], GRAPH.IMAGE)

    def test_multimodal_model_uses_node_output_type(self):
        model_defs = {
            "multi": {"metadata": {"outputs": ["image", "video"], "main_output": ["image", "video"], "media_inputs": {}}}
        }
        image_ports = GRAPH.node_ports(GRAPH.new_node("generate_image", params={"model_type": "multi"}), model_defs)
        video_ports = GRAPH.node_ports(GRAPH.new_node("generate_video", params={"model_type": "multi"}), model_defs)
        self.assertEqual(image_ports["output"]["type"], GRAPH.IMAGE)
        self.assertEqual(video_ports["output"]["type"], GRAPH.VIDEO)
        self.assertTrue(GRAPH.model_supports_node("generate_image", model_defs["multi"]))
        self.assertTrue(GRAPH.model_supports_node("generate_video", model_defs["multi"]))

    def test_postprocess_exposes_reference_and_audio_inputs(self):
        ports = GRAPH.node_ports(GRAPH.new_node("postprocess"))
        self.assertEqual(ports["references"]["type"], "IMAGE[]")
        self.assertEqual(ports["audio"]["type"], GRAPH.AUDIO)

    def test_prompt_enhancer_ports_follow_reference_mode(self):
        node = GRAPH.new_node("prompt_enhancer", params={"mode": "T"})
        self.assertIn("prompt", GRAPH.node_ports(node))
        self.assertNotIn("images", GRAPH.node_ports(node))
        node["params"]["mode"] = "TI"
        self.assertEqual(GRAPH.node_ports(node)["images"]["type"], "IMAGE[]")
        self.assertEqual(GRAPH.node_ports(node)["text"]["type"], GRAPH.TEXT)

    def test_configuration_nodes_have_typed_outputs_and_generation_overrides(self):
        generation = GRAPH.node_ports(GRAPH.new_node("generate_video", params={"model_type": "video"}), {
            "video": {"metadata": {"outputs": ["video"], "main_output": ["video"], "media_inputs": {}}}
        })
        self.assertEqual(generation["resolution_config"]["type"], GRAPH.RESOLUTION_SETTINGS)
        self.assertEqual(generation["lora_stack"]["type"], GRAPH.LORA_STACK)
        self.assertEqual(generation["sampling_config"]["type"], GRAPH.SAMPLING_SETTINGS)
        self.assertEqual(GRAPH.node_ports(GRAPH.new_node("lora_stack"))["stack"]["type"], GRAPH.LORA_STACK)
        self.assertEqual(GRAPH.node_ports(GRAPH.new_node("attention_config"))["settings"]["type"], GRAPH.ATTENTION_SETTINGS)

    def test_magic_mask_accepts_media_and_exposes_typed_mask_outputs(self):
        ports = GRAPH.node_ports(GRAPH.new_node("magic_mask"))
        self.assertEqual(ports["source"]["type"], GRAPH.ANY_MEDIA)
        self.assertEqual(ports["mask_image"]["type"], GRAPH.MASK_IMAGE)
        self.assertEqual(ports["mask_video"]["type"], GRAPH.MASK_VIDEO)
        self.assertTrue(GRAPH.validate_graph({**GRAPH.new_graph(), "nodes": [GRAPH.new_node("magic_mask")]}, None))

    def test_disabled_group_bypasses_a_simple_media_chain(self):
        source = GRAPH.new_node("input_video", "Source")
        process = GRAPH.new_node("ffmpeg_fps", "Preview-only FPS")
        output = GRAPH.new_node("ffmpeg_export", "Output")
        graph = GRAPH.new_graph()
        graph["nodes"] = [source, process, output]
        graph["groups"] = [{"id": "post", "title": "Postprocess", "node_ids": [process["id"]], "enabled": False}]
        graph["edges"] = [
            {"id": "in", "source": {"node": source["id"], "port": "video"}, "target": {"node": process["id"], "port": "video"}, "type": "VIDEO"},
            {"id": "out", "source": {"node": process["id"], "port": "output"}, "target": {"node": output["id"], "port": "video"}, "type": "VIDEO"},
        ]
        order = GRAPH.topological_order(graph)
        self.assertEqual(order, [source["id"], output["id"]])
        bypass = [edge for edge in GRAPH.effective_edges(graph) if edge.get("bypass")]
        self.assertEqual(len(bypass), 1)
        self.assertEqual(bypass[0]["source"]["node"], source["id"])
        self.assertEqual(bypass[0]["target"]["node"], output["id"])
        self.assertEqual(GRAPH.validate_graph(graph), [])

    def test_inpainting_requires_a_mask_and_keeps_unknown_import_fields(self):
        graph = GRAPH.migrate_v1_workflow({
            "name": "Legacy",
            "stages": [{
                "id": "first", "stage_type": "generation", "settings": {"model_type": "test"},
                "future_option": {"enabled": True},
            }],
        })
        node = next(item for item in graph["nodes"] if item["id"] == "first")
        self.assertEqual(node["params"]["raw_settings"]["unknown_stage_fields"]["future_option"], {"enabled": True})
        self.assertIn("warning", node["params"])
        inpaint = GRAPH.new_node("inpaint_image", params={"model_type": "test"})
        errors = GRAPH.validate_graph({**GRAPH.new_graph(), "nodes": [inpaint], "edges": []}, {"test": {"metadata": {"outputs": ["image"], "main_output": ["image"], "media_inputs": {}}}})
        self.assertTrue(any("Required input mask" in error for error in errors))

        capable = {"metadata": {"outputs": ["image"], "main_output": ["image"], "capabilities": {"inpainting": True}, "media_inputs": {}}}
        generate_ports = GRAPH.node_ports(GRAPH.new_node("generate_image", params={"model_type": "capable"}), {"capable": capable})
        self.assertIn("mask", generate_ports)
        self.assertTrue(generate_ports["mask"]["optional"])

        image = GRAPH.new_node("input_image", "Image")
        mask = GRAPH.new_node("input_mask_image", "Mask")
        generate = GRAPH.new_node("generate_image", "Generate", params={"model_type": "capable"})
        graph = {**GRAPH.new_graph(), "nodes": [image, mask, generate], "edges": [
            {"id": "mask", "source": {"node": mask["id"], "port": "mask"}, "target": {"node": generate["id"], "port": "mask"}, "type": "MASK_IMAGE"},
        ]}
        self.assertTrue(any("requires a connected image source" in error for error in GRAPH.validate_graph(graph, {"capable": capable})))

    def test_all_public_example_workflows_normalize(self):
        workflows = ROOT / "examples"
        files = sorted(workflows.glob("*.json"))
        self.assertTrue(files)
        for path in files:
            with self.subTest(path=path.name):
                graph = GRAPH.normalize_graph(json.loads(path.read_text(encoding="utf-8")))
                self.assertEqual(graph["schema"], GRAPH.SCHEMA)
                self.assertTrue(graph["nodes"])


if __name__ == "__main__":
    unittest.main()
