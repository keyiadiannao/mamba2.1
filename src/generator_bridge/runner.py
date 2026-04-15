from __future__ import annotations

from typing import Any

from .prompting import build_generator_prompt

_GENERATOR_CACHE: dict[tuple[str, str, str], tuple[object, object]] = {}


def build_generator_result(
    config: dict[str, Any],
    question: str,
    evidence_texts: list[str],
) -> tuple[str | None, str, str | None]:
    prompt = build_generator_prompt(question, evidence_texts)
    if not bool(config.get("run_generator", False)):
        return None, prompt, None

    try:
        answer = generate_answer(config, prompt, evidence_texts)
    except Exception as exc:  # pragma: no cover - exercised through runner tests
        return None, prompt, str(exc)
    return answer, prompt, None


def generate_answer(
    config: dict[str, Any],
    prompt: str,
    evidence_texts: list[str],
) -> str:
    inference_mode = str(
        config.get("generator_inference_mode")
        or ("extractive_first_evidence" if str(config.get("generator_type", "qwen")) == "mock" else "hf_causal_lm")
    )

    if inference_mode == "extractive_first_evidence":
        return evidence_texts[0].strip() if evidence_texts else ""
    if inference_mode != "hf_causal_lm":
        raise ValueError(f"Unsupported generator_inference_mode: {inference_mode}")

    model_name = str(
        config.get("generator_hf_model_name")
        or config.get("generator_model_path")
        or config.get("generator_model_name")
        or ""
    ).strip()
    if not model_name or model_name == "qwen":
        raise ValueError(
            "Generator inference requires `generator_hf_model_name` or `generator_model_path`; "
            "the placeholder `generator_model_name = qwen` is not a runnable model id."
        )

    tokenizer, model = _load_hf_generator(
        model_name=model_name,
        device=str(config.get("generator_device", "cpu")),
        dtype=str(config.get("generator_dtype", "float16")),
    )

    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    model_device = next(model.parameters()).device
    inputs = {name: tensor.to(model_device) for name, tensor in inputs.items()}

    generate_kwargs = {
        "max_new_tokens": int(config.get("generator_max_new_tokens", 64)),
        "do_sample": bool(config.get("generator_do_sample", False)),
        "temperature": float(config.get("generator_temperature", 1.0)),
        "top_p": float(config.get("generator_top_p", 1.0)),
        "pad_token_id": tokenizer.pad_token_id,
    }
    if not generate_kwargs["do_sample"]:
        generate_kwargs.pop("temperature", None)
        generate_kwargs.pop("top_p", None)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)

    prompt_token_count = int(inputs["input_ids"].shape[-1])
    generated_ids = output_ids[0][prompt_token_count:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if not answer:
        return ""
    return answer.splitlines()[0].strip()


def _load_hf_generator(model_name: str, device: str, dtype: str) -> tuple[object, object]:
    cache_key = (model_name, device, dtype)
    cached = _GENERATOR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_kwargs: dict[str, Any] = {}
    torch_dtype = _resolve_torch_dtype(dtype)
    if torch_dtype is not None:
        model_kwargs["dtype"] = torch_dtype
    if device == "cuda":
        model_kwargs["device_map"] = "auto"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, **model_kwargs)
    if device != "cuda":
        model.to(torch.device(device))

    _GENERATOR_CACHE[cache_key] = (tokenizer, model)
    return tokenizer, model


def _resolve_torch_dtype(dtype: str) -> Any:
    normalized = dtype.lower().strip()
    if not normalized:
        return None

    import torch

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(normalized)
