from typing import List, Optional
import os

from openai import OpenAI


class VLLMServerClient:
    """
    Thin client for a persistent vLLM OpenAI-compatible server.

    Exposes the same generate/generate_batch interface as VLLMWrapper to
    minimize code changes elsewhere.
    """

    def __init__(
        self,
        server_url: str,
        served_model_name: str,
        stop_list: Optional[List[str]] = None,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.model = served_model_name
        self.stop_list = stop_list or ["\nHuman:", "\n```\n", "\nQuestion:", "<|endoftext|>", "\n"]

        # The vLLM OpenAI server can be configured to not validate the key.
        # We accept any provided key or fall back to a benign default.
        api_key = os.getenv("OPENAI_API_KEY") or "EMPTY"
        self.client = OpenAI(base_url=self.server_url, api_key=api_key)

    def _create(self, prompts: List[str], **kwargs) -> List[str]:
        # Map supported params; ignore unsupported to keep compatibility.
        request = {
            "model": self.model,
            "prompt": prompts,
        }
        if "max_new_tokens" in kwargs and kwargs["max_new_tokens"] is not None:
            request["max_tokens"] = int(kwargs["max_new_tokens"])
        if "temperature" in kwargs and kwargs["temperature"] is not None:
            request["temperature"] = float(kwargs["temperature"])
        if "top_p" in kwargs and kwargs["top_p"] is not None:
            request["top_p"] = float(kwargs["top_p"])
        if "presence_penalty" in kwargs and kwargs["presence_penalty"] is not None:
            request["presence_penalty"] = float(kwargs["presence_penalty"])
        # Repetition penalty and top_k are not standard OpenAI params; omit here.
        if self.stop_list:
            request["stop"] = self.stop_list

        resp = self.client.completions.create(**request)
        # OpenAI completions batches return choices aligned to input order.
        # vLLM mirrors this behavior.
        return [c.text for c in resp.choices]

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 15,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> List[str]:
        return self._create(
            [prompt],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 15,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 500,
        repetition_penalty: float = 1.1,
        presence_penalty: float = 0.0,
    ) -> List[str]:
        return self._create(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
        )

