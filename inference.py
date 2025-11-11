"""Minimal Ollama HTTP wrapper: health, models, generate, embeddings and streaming.

This module provides a small, explicit wrapper around the local Ollama HTTP
API. It intentionally keeps the surface small while exposing commonly used
endpoints: status, models, show, ps (running processes), generate, embeddings
and a streaming generator.
"""
from __future__ import annotations

import argparse
import json
from typing import List, Optional, Iterator, Any

import requests


class OllamaClient:
    """HTTP client for a locally hosted Ollama server exposing useful endpoints.

    Methods implemented:
      - is_healthy(): try /api/status then /api/version
      - list_models(): GET /api/models (fallback /api/tags)
      - list_running_models(): GET /api/ps
      - show_model(model, verbose=False): POST /api/show
      - generate(...): POST /api/generate
      - embeddings(...): POST /api/embeddings
      - stream_generate(...): POST /api/generate with stream=True and yield chunks
    """

    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_healthy(self) -> bool:
        # prefer /api/status if available
        r = requests.get(f"{self.base_url}/api/version", timeout=2)
        if r.status_code == 200:
            return True
        return False

    def list_models(self) -> List[Any]:
        # try the documented /api/models, fall back to /api/tags
        for path in ("/api/models", "/api/tags"):
            try:
                r = requests.get(f"{self.base_url}{path}", timeout=self.timeout)
                r.raise_for_status()
                return r.json()
            except Exception:
                continue
        raise RuntimeError("Unable to list models from Ollama server")

    def list_running_models(self) -> List[Any]:
        r = requests.get(f"{self.base_url}/api/ps", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def show_model(self, model: str, verbose: bool = False) -> dict:
        payload = {"model": model, "verbose": bool(verbose)}
        r = requests.post(f"{self.base_url}/api/show", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()
    
    def delete_model(self, model: str) -> dict:
        payload = {"model": model}
        r = requests.post(f"{self.base_url}/api/delete", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()
    
    def download_model(self, model: str) -> dict:
        payload = {"model": model}
        r = requests.post(f"{self.base_url}/api/pull", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def warmup_model(self, model: str) -> dict:
        payload = {
            "model": model,
        }
        r = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
        r.raise_for_status()
        if r.status_code == 200:
            print(f"Model {model} warmed up successfully.")
    
    def generate(self, model: str, prompt: str, temperature: float = 0.0, max_tokens: int = 256) -> str:#, thinking=True) -> str:
        # if not thinking and "qwen" in model.lower():
        #     # Add a clear separator for the model
        #     prompt = f"{prompt} /no_think"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": float(temperature), "num_predict": int(max_tokens)},
            #"think": thinking
        }
        r = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json().get("response", "")

    def embeddings(self, model: str, inputs: List[str]) -> dict:
        """Call the embeddings endpoint and return raw JSON.

        Depending on Ollama version the request/response shape may vary. This
        returns the parsed JSON so the caller can interpret it.
        """
        payload = {"model": model, "input": inputs}
        r = requests.post(f"{self.base_url}/api/embeddings", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()


def _cli():
    ap = argparse.ArgumentParser(description="Minimal Ollama client: list models, generate text")
    ap.add_argument("--base_url", default="http://localhost:11434")
    ap.add_argument("--list", action="store_true", help="List models on Ollama server")
    ap.add_argument("--generate", action="store_true", help="Run generation")
    ap.add_argument("--model", help="Model name for generation (required with --generate)")
    ap.add_argument("--prompt", help="Prompt text for generation (required with --generate)")
    ap.add_argument("--temp", type=float, default=0.0, help="Sampling temperature")
    ap.add_argument("--max_tokens", type=int, default=256, help="Max tokens to generate")
    args = ap.parse_args()

    client = OllamaClient(base_url=args.base_url)
    if args.list:
        models = client.list_models()
        print(models)
        return
    if args.generate:
        if not args.model or not args.prompt:
            raise SystemExit("--model and --prompt are required with --generate")
        out = client.generate(args.model, args.prompt, temperature=args.temp, max_tokens=args.max_tokens)
        print(out)


if __name__ == "__main__":
    _cli()
