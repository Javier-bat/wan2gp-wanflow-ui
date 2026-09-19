from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PIL import Image

from .runtime_engine import ai_model_supports_multiple_images, normalize_ai_model


AI_SYSTEM_PROMPT = (
    "Follow the user's request accurately. When images are provided, base the answer only on visible "
    "information from those images. Return only the requested result, without an introduction, labels, "
    "analysis, or quotation marks."
)


def _load_images(paths: list[str]) -> list[Image.Image]:
    images = []
    for path in paths:
        source = Path(path)
        if not source.is_file():
            raise ValueError(f"AI image input no longer exists: {path}")
        try:
            with Image.open(source) as image:
                images.append(image.convert("RGB"))
        except Exception as exc:
            raise ValueError(f"AI image input is not a readable image: {path}") from exc
    return images


def _run_qwen_multimodal(
    caption_model: Any,
    processor: Any,
    text_model: Any,
    instruction: str,
    images: list[Image.Image],
    max_tokens: int,
) -> str:
    from shared.prompt_enhancer import qwen35_text
    from shared.prompt_enhancer.qwen35_assistant_runtime import Qwen35AssistantRuntime
    from shared.prompt_enhancer.qwen35_vl import _generate_and_decode, _prepare_multimodal_vllm_prompt

    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": instruction})
    messages = [
        {"role": "system", "content": AI_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    model_inputs = processor(
        text=[text],
        images=images,
        return_tensors="pt",
        padding=True,
        return_mm_token_type_ids=True,
    )
    if not (
        qwen35_text._use_vllm_prompt_enhancer(text_model)
        or qwen35_text._use_legacy_cuda_runner_prompt_enhancer(text_model)
    ):
        outputs = _generate_and_decode(
            caption_model,
            model_inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            seed=0,
            progress_desc="Workflow AI analysis tokens",
        )
        return str(outputs[0] if outputs else "").strip()
    prompt_token_ids, prompt_embeds, prompt_position_ids, position_offset = _prepare_multimodal_vllm_prompt(
        caption_model,
        model_inputs,
    )
    runtime = Qwen35AssistantRuntime(text_model)
    return runtime.generate_embedded_answer(
        prompt_token_ids,
        prompt_embeds,
        prompt_position_ids,
        position_offset,
        max_new_tokens=max_tokens,
        seed=0,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )


def _run_caption_then_text(
    controller: Any,
    text_model: Any,
    tokenizer: Any,
    instruction: str,
    images: list[Image.Image],
    max_tokens: int,
) -> str:
    from shared.prompt_enhancer.prompt_enhance_utils import generate_cinematic_prompt

    deps = controller._deps
    outputs = generate_cinematic_prompt(
        deps.get_image_caption_model(),
        deps.get_image_caption_processor(),
        text_model,
        tokenizer,
        [instruction],
        images or None,
        video_prompt=False,
        text_prompt=not images,
        max_new_tokens=max_tokens,
        prompt_enhancer_instructions=AI_SYSTEM_PROMPT,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        seed=0,
        thinking_enabled=False,
    )
    return str(outputs[0] if outputs else "").strip()


def run_ai_analysis(
    controller: Any,
    main_state: dict[str, Any],
    model_choice: Any,
    instruction: str,
    image_paths: list[str],
    max_tokens: Any,
    *,
    reset_prompt_enhancer: Callable[[], Any],
    reset_prompt_enhancer_if_requested: Callable[[], Any],
) -> str:
    model_choice = normalize_ai_model(model_choice)
    instruction = str(instruction or "").strip()
    if not instruction:
        raise ValueError("AI instruction is empty.")
    image_paths = [str(path) for path in image_paths if str(path)]
    if len(image_paths) > 1 and not ai_model_supports_multiple_images(model_choice):
        raise ValueError("The selected AI model accepts only one image per request.")
    max_tokens = max(32, min(1024, int(max_tokens or 192)))

    if controller is None or not hasattr(controller, "_deps"):
        raise RuntimeError("Wan2GP AI runtime is unavailable.")
    deps = controller._deps
    server_config = deps.get_server_config()
    previous_model_choice = int(server_config.get("enhancer_enabled", 0) or 0)
    acquired = False
    images: list[Image.Image] = []
    try:
        reset_prompt_enhancer()
        reset_prompt_enhancer_if_requested()
        server_config["enhancer_enabled"] = model_choice
        deps.acquire_gpu(main_state)
        acquired = True
        text_model, tokenizer = deps.ensure_prompt_enhancer_loaded(override_profile=-1)
        images = _load_images(image_paths)
        if images and ai_model_supports_multiple_images(model_choice):
            caption_model, processor = controller._ensure_vision_loaded(override_profile=-1)
            answer = _run_qwen_multimodal(
                caption_model,
                processor,
                text_model,
                instruction,
                images,
                max_tokens,
            )
        else:
            answer = _run_caption_then_text(controller, text_model, tokenizer, instruction, images, max_tokens)
        answer = str(answer or "").strip()
        if not answer:
            raise RuntimeError("The AI model returned an empty response.")
        return answer
    finally:
        for image in images:
            image.close()
        try:
            deps.unload_prompt_enhancer_runtime()
        finally:
            try:
                controller._unload_weights()
            finally:
                try:
                    if acquired:
                        deps.release_gpu(main_state)
                finally:
                    server_config["enhancer_enabled"] = previous_model_choice
                    reset_prompt_enhancer()
                    reset_prompt_enhancer_if_requested()
