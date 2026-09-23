from __future__ import annotations

from shared.utils.plugins import WAN2GPPlugin

from .plugin_ui import create_config_ui
from .plugin_ui import _bridge


PlugIn_Name = "WanFlow UI"
PlugIn_Id = "wan2gp_wanflow_ui"


class ConfigTabPlugin(WAN2GPPlugin):
    def setup_ui(self):
        # The plugin manager collects custom JS during setup_ui.  Registering
        # this from create_config_ui is too late, so iframe Save/Run messages
        # would never reach the Gradio components.
        self.add_custom_js(_bridge())
        self.request_global("get_model_def")
        self.request_global("list_model_defs")
        self.request_global("get_default_settings")
        self.request_global("get_lora_dir")
        self.request_global("get_resolution_choices")
        self.request_global("get_base_model_type")
        self.request_global("get_parent_model_type")
        self.request_global("get_model_family")
        self.request_global("get_prompt_enhancer_choices")
        self.request_global("exec_prompt_enhancer_engine")
        self.request_global("release_extension_offloadobjs")
        self.request_global("release_model")
        self.request_global("any_GPU_process_running")
        self.request_global("release_deepy_vram")
        self.request_global("reset_prompt_enhancer")
        self.request_global("reset_prompt_enhancer_if_requested")
        self.request_global("_deepy")
        self.request_component("state")
        self.add_tab(
            tab_id=PlugIn_Id,
            label=PlugIn_Name,
            component_constructor=self.create_config_ui,
        )

    def create_config_ui(self, api_session):
        return create_config_ui(self, api_session)
