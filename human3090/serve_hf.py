#!/usr/bin/env python3
"""OpenAI-compatible server for HuggingFace models.

Drop-in replacement for llama-server when benchmarking non-GGUF models.
Loads any transformers model with optional 4-bit quantization.

Usage:
    python -m human3090.serve_hf openbmb/MiniCPM-SALA
    python -m human3090.serve_hf meta-llama/Llama-3-8B-Instruct --no-quantize
    python -m human3090.serve_hf ./local-model --port 8084

    # Then benchmark as usual:
    python -m human3090.run_eval --model MiniCPM-SALA --temperature 0.9
"""

import argparse
import json
import os
import time
import uuid
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from huggingface_hub import snapshot_download
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "/mnt/raid5/models"

app = FastAPI()
_model = None
_tokenizer = None
_model_name = None


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = ""
    messages: list[Message]
    temperature: float = 0.9
    top_p: float = 0.9
    max_tokens: int = 1000
    stream: bool = False
    min_p: Optional[float] = None
    top_k: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None


@app.get("/health")
def health():
    return {"status": "ok" if _model is not None else "loading model"}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    prompt = _tokenizer.apply_chat_template(
        [{"role": m.role, "content": m.content} for m in req.messages],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)

    gen_kwargs = {
        "max_new_tokens": req.max_tokens,
        "do_sample": req.temperature > 0,
    }
    if req.temperature > 0:
        gen_kwargs["temperature"] = req.temperature
        gen_kwargs["top_p"] = req.top_p
        if req.min_p is not None:
            gen_kwargs["min_p"] = req.min_p
        if req.top_k is not None:
            gen_kwargs["top_k"] = req.top_k

    with torch.no_grad():
        outputs = _model.generate(**inputs, **gen_kwargs)

    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    text = _tokenizer.decode(new_tokens, skip_special_tokens=True)

    resp_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    model_name = req.model or _model_name

    if req.stream:
        def stream_response():
            chunk = {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream_response(), media_type="text/event-stream")

    return {
        "id": resp_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": inputs.input_ids.shape[1],
            "completion_tokens": len(new_tokens),
            "total_tokens": inputs.input_ids.shape[1] + len(new_tokens),
        },
    }


def main():
    global _model, _tokenizer, _model_name

    parser = argparse.ArgumentParser(description="Serve a HuggingFace model with OpenAI-compatible API")
    parser.add_argument("model", help="HuggingFace model ID or local path")
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument("--no-quantize", action="store_true", help="Load in full precision (default: 4-bit)")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    _model_name = args.model.split("/")[-1]

    # Resolve model path: use local dir if it exists, otherwise download
    if os.path.isdir(args.model):
        model_path = args.model
    else:
        model_path = os.path.join(MODEL_DIR, _model_name)
        if not os.path.isdir(model_path):
            print(f"Downloading {args.model} to {model_path}...")
            snapshot_download(args.model, local_dir=model_path)
        else:
            print(f"Using existing model at {model_path}")

    print(f"Loading {model_path}{'  (full precision)' if args.no_quantize else ' (4-bit)'}...")
    t0 = time.time()

    load_kwargs = {"device_map": "auto"}
    if args.trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    if not args.no_quantize:
        load_kwargs["load_in_4bit"] = True

    tok_kwargs = {}
    if args.trust_remote_code:
        tok_kwargs["trust_remote_code"] = True
    _tokenizer = AutoTokenizer.from_pretrained(model_path, **tok_kwargs)
    _model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    _model.eval()
    print(f"Ready in {time.time() - t0:.1f}s — http://localhost:{args.port}")

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
