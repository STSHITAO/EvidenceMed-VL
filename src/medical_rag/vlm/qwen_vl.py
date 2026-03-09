from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import List

import requests
import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from medical_rag.config import VLMConfig
from medical_rag.prompts import SYSTEM_PROMPT, build_user_prompt

try:
    from qwen_vl_utils import process_vision_info
except Exception:  # pragma: no cover - optional dependency fallback
    process_vision_info = None


class QwenVLMReasoner:
    def __init__(self, config: VLMConfig) -> None:
        self.config = config
        self.backend = (config.backend or "transformers").lower()

        if self.backend == "vllm_openai":
            self.api_base_url = config.api_base_url.rstrip("/")
            self.api_key = config.api_key or os.getenv(config.api_key_env, "")
            if not self.api_base_url:
                raise ValueError("vlm.api_base_url is required when vlm.backend=vllm_openai")
            self.model_name = config.model_name or Path(config.base_model_path).name
            return

        if self.backend != "transformers":
            raise ValueError(f"Unsupported vlm.backend: {config.backend}")

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(config.dtype, torch.bfloat16)

        self.processor = AutoProcessor.from_pretrained(config.base_model_path, trust_remote_code=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.base_model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map="auto" if config.device.startswith("cuda") else None,
        )

        adapter_path = (config.lora_adapter_path or "").strip()
        if adapter_path and Path(adapter_path).exists():
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        if not config.device.startswith("cuda") or not torch.cuda.is_available():
            self.model.to("cpu")
            self.torch_device = torch.device("cpu")
        else:
            self.torch_device = torch.device("cuda")

        self.model.eval()

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _image_to_data_url(image_path: str) -> str:
        mime, _ = mimetypes.guess_type(image_path)
        if not mime:
            mime = "image/png"
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{encoded}"

    def _build_messages(self, image_path: str, question: str, evidence_blocks: List[str]) -> list:
        user_prompt = build_user_prompt(question, evidence_blocks)
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

    @staticmethod
    def _extract_openai_text(body: dict) -> str:
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"Invalid VLM response format: keys={list(body.keys())}")

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
            if parts:
                return "\n".join(parts)
        return str(content).strip()

    def _generate_vllm(self, image_path: str, question: str, evidence_blocks: List[str]) -> str:
        user_prompt = build_user_prompt(question, evidence_blocks)
        image_data_url = self._image_to_data_url(image_path)
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                },
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_new_tokens,
            "stream": False,
        }

        url = f"{self.api_base_url}{self.config.api_chat_path}"
        resp = requests.post(url, headers=self._headers(), json=payload, timeout=self.config.request_timeout_sec)
        resp.raise_for_status()
        body = resp.json()
        return self._extract_openai_text(body)

    def generate(self, image_path: str, question: str, evidence_blocks: List[str]) -> str:
        if self.backend == "vllm_openai":
            return self._generate_vllm(image_path=image_path, question=question, evidence_blocks=evidence_blocks)

        messages = self._build_messages(image_path=image_path, question=question, evidence_blocks=evidence_blocks)
        prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if process_vision_info is not None:
            image_inputs, video_inputs = process_vision_info(messages)
        else:
            image_inputs = [Image.open(image_path).convert("RGB")]
            video_inputs = None

        inputs = self.processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {
            k: (v.to(self.torch_device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.temperature > 0,
            )

        input_len = inputs["input_ids"].shape[1]
        new_tokens = generated[:, input_len:]
        output = self.processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output[0].strip()
