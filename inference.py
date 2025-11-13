"""Minimal Ollama HTTP wrapper: health, models, generate, embeddings and streaming.

This module provides a small, explicit wrapper around the local Ollama HTTP
API. It intentionally keeps the surface small while exposing commonly used
endpoints: status, models, show, ps (running processes), generate, embeddings
and a streaming generator.
"""
from __future__ import annotations

import argparse
import json
from typing import List, Optional, Iterator, Any, Tuple
import torch
from trl import AutoModelForCausalLMWithValueHead
from copy import deepcopy
from transformers import AutoTokenizer
import os


import requests

class LLMClient:
    """Abstract base class for LLM clients. Defines the interface for health check,
    model listing, generation, and embeddings.
    """
    def __init__(self):
        pass

    def is_healthy(self) -> bool:
        """Check if the LLM service is healthy."""
        raise NotImplementedError

    def list_models(self) -> List[Any]:
        """List available models."""
        raise NotImplementedError

    def generate(self, model: str, prompt: str) -> str:
        """Generate text from a prompt using the specified model."""
        raise NotImplementedError

    def embeddings(self, model: str, inputs: List[str]) -> dict:
        """Get embeddings for a list of input strings."""
        raise NotImplementedError

class OllamaClient(LLMClient):
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

    def _parse_models(self, data: Any) -> List[Any]:
        models = []
        for item in data.get("models", []):
            if isinstance(item, dict) and "name" in item:
                size = item.get("details").get("parameter_size") if item.get("details") else None
                models.append((item["name"], size))
        return models

    def list_models(self) -> List[Any]:
        # try the documented /api/models, fall back to /api/tags
        for path in ("/api/models", "/api/tags"):
            try:
                r = requests.get(f"{self.base_url}{path}", timeout=self.timeout)
                r.raise_for_status()
                return self._parse_models(r.json())
            except Exception:
                continue
        raise RuntimeError("Unable to list models from Ollama server")

    def list_running_models(self) -> List[Any]:
        r = requests.get(f"{self.base_url}/api/ps", timeout=self.timeout)
        r.raise_for_status()
        return self._parse_models(r.json())

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
        if r.status_code == 200:
            print(f"Model {model} downloaded successfully.")
        else:
            raise RuntimeError(f"Failed to download model {model}: {r.status_code} {r.text}")
    #this loads the model into memory, so call it before training
    def warmup_model(self, model: str) -> dict:
        payload = {
            "model": model,
        }
        r = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
        r.raise_for_status()
        if r.status_code == 200:
            print(f"Model {model} warmed up successfully.")
        else:
            raise RuntimeError(f"Failed to warm up model {model}: {r.status_code} {r.text}")
    #this is the main method for inference
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
    #might use this?
    def embeddings(self, model: str, inputs: List[str]) -> dict:
        """Call the embeddings endpoint and return raw JSON.

        Depending on Ollama version the request/response shape may vary. This
        returns the parsed JSON so the caller can interpret it.
        """
        payload = {"model": model, "input": inputs}
        r = requests.post(f"{self.base_url}/api/embeddings", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()


class HFClient(LLMClient):
    """ This client uses local Hugging Face models for inference. It expects the model to be
    downloaded and available locally. You initialize it with the model name and it will load the tokenizer and model."""
    def __init__(self, compile_model: bool = False, load_in_8bit: bool = False, torch_dtype: Optional[str] = None):
        super().__init__()
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Configuration for faster inference
        self.compile_model = bool(compile_model)
        self.load_in_8bit = bool(load_in_8bit)
        # torch_dtype may be 'fp16', 'bf16', or 'fp32'
        self.torch_dtype = torch_dtype

    def is_healthy(self):
        return True
    def list_models(self):
        #get the list of models in the local directory
        models = []
        for root, dirs, files in os.walk("models"):
            if "config.json" in files:
                # root like models/<owner>/<model>
                rel = os.path.relpath(root, "models")
                models.append(rel.replace("\\", "/"))
        return models
    
    def warmup_model(self, model_id: str):
        local_dir = f"models/{model_id}"
        if model_id not in self.list_models():
            download_model(model_id, local_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(local_dir, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Build kwargs for from_pretrained depending on precision/quantization settings
        model_kwargs = {}
        if self.load_in_8bit:
            # bitsandbytes / 8-bit path; device_map auto will place weights
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"
        elif isinstance(self.torch_dtype, str):
            dt = self.torch_dtype.lower()
            dtype_map = {
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "fp32": torch.float32,
            }
            model_kwargs["torch_dtype"] = dtype_map.get(dt, None)

        # Load model with chosen options
        try:
            load_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
            if load_kwargs:
                self.model = AutoModelForCausalLMWithValueHead.from_pretrained(local_dir, **load_kwargs)
            else:
                self.model = AutoModelForCausalLMWithValueHead.from_pretrained(local_dir)
        except Exception as e:
            # Fallback to default load if specialized load fails
            print(f"Model load with kwargs {model_kwargs} failed: {e}. Falling back to default load.")
            self.model = AutoModelForCausalLMWithValueHead.from_pretrained(local_dir)

        # If model wasn't placed on device_map auto (i.e., load_in_8bit False), move to desired device
        try:
            first_param = next(self.model.parameters())
            if first_param.device.type == "cpu" and self.device.type == "cuda":
                try:
                    self.model.to(self.device)
                except Exception:
                    pass
        except StopIteration:
            pass

        # Prepare model for inference
        try:
            self.model.eval()
        except Exception:
            pass

        # Optionally compile the model for faster inference (PyTorch 2.x)
        if self.compile_model:
            try:
                self.model = torch.compile(self.model)
                print("Model compiled with torch.compile()")
            except Exception as e:
                print(f"torch.compile failed or not available: {e}")

    def generate(
        self,
        model_id: str,
        prompt: str,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 1.0,
        use_cache: bool = True,
    ) -> str:
        """Generate text with tunable generation params. Defaults favor greedy decoding for speed."""
        if self.model is None or self.tokenizer is None:
            self.warmup_model(model_id)
        toks = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        toks = {k: v.to(self.device) for k, v in toks.items()}
        # Use inference_mode for faster inference when gradients aren't needed
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=toks.get("input_ids"),
                attention_mask=toks.get("attention_mask"),
                max_new_tokens=int(max_new_tokens),
                do_sample=bool(do_sample),
                top_k=int(top_k),
                top_p=float(top_p),
                use_cache=bool(use_cache),
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def embeddings(self, model_id: str, inputs: List[str]) -> dict:
        if self.model is None or self.tokenizer is None:
            self.warmup_model(model_id)
        toks = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
        toks = {k: v.to(self.device) for k, v in toks.items()}
        with torch.no_grad():
            emb_layer = self.model.get_input_embeddings()
            token_embs = emb_layer(toks["input_ids"])  # [B, T, D]
            # If attention_mask is present, mean-pool over valid tokens
            mask = toks.get("attention_mask")
            if mask is not None:
                mask = mask.unsqueeze(-1).to(dtype=token_embs.dtype)
                summed = (token_embs * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1)
                pooled = (summed / counts).cpu().numpy().tolist()
                return {"embeddings": pooled}
            return {"embeddings": token_embs.cpu().numpy().tolist()}

        

# ---------------- HF loading helper -----------------
def download_model(model_id: str, local_dir: str) -> str:
    """Download model from Hugging Face hub to local directory.

    Returns the local directory path where the model is saved.
    """
    from huggingface_hub import snapshot_download
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(repo_id=model_id, local_dir=local_dir)
    return local_dir

# This loads model_id from Hugging face and saves it locally.
def load_from_hf(model_id: str) -> Tuple[Any, Any, Any]:
    """Load model & tokenizer from Hugging Face hub or local path; optionally persist locally.

    Returns (tokenizer, model, ref_model) where model is an AutoModelForCausalLMWithValueHead instance.
    Ref_model is a frozen copy of the base model without value head.
    This is useful for baseline SFT loading prior to PPO wrapping.
    """
    #check if the model is downloaded locally already, recall models are saved as models/{owner}/{model_name}
    local_dir = "models/" + model_id
    if not os.path.exists(local_dir):
        print(f"Downloading model {model_id} from Hugging Face hub...")
        owner, model_name = model_id.split("/")
        local_dir = download_model(model_id, f"models/{owner}/{model_name}")
    print(f"Loading model from local directory: {local_dir}")
    tokenizer = AutoTokenizer.from_pretrained(local_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    #this needs to be a model with value head, otherwise you need a separate value model for PPOTrainer
    model = AutoModelForCausalLMWithValueHead.from_pretrained(local_dir)
    ref_model = deepcopy(model)
    #freeze the reference model
    for param in ref_model.parameters():
        param.requires_grad = False
    return tokenizer, model.to("cuda"), ref_model.to("cuda")