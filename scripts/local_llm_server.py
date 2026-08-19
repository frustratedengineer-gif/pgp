#!/usr/bin/env python
"""
Zero-cost local LLM server for Mem0's indexing calls, so the Mem0
baseline (reviewer gap #1) doesn't have to spend real OpenRouter money on
every one of Mem0's own extraction calls per conversation turn -- the
same Qwen2.5-7B-Instruct model this project already used as a local,
zero-marginal-cost baseline in Week 3 (`baselines/llm_prompted_ttl.py`),
just served differently: that used an Ollama server (unavailable on this
VM -- ollama's install script couldn't reach GitHub's release CDN from
here, a network restriction, not a code problem), this uses a minimal
hand-rolled OpenAI-compatible HTTP server (stdlib `http.server` only, no
new heavy dependency, since neither `fastapi` nor `flask` were already
installed) that Mem0's BUILT-IN "vllm" provider can talk to (it's just an
OpenAI-client pointed at a configurable base_url -- confirmed by reading
`mem0/llms/vllm.py` directly, not assumed from docs).

Not a production inference server: single-threaded, greedy decoding, no
batching. Fine for this use case since the Mem0 indexing driver
(scripts/eval_mem0_baseline.py) calls it one turn at a time anyway.

    python scripts/local_llm_server.py --port 8420 --model Qwen/Qwen2.5-7B-Instruct --device cuda:6
"""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_model = None
_tokenizer = None
_device = None


def load_model(model_name: str, device: str):
    global _model, _tokenizer, _device
    print(f"Loading {model_name} on {device} ...")
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    _model.eval()
    _device = device
    print("Model loaded.")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # keep stdout clean; the driver script logs progress itself

    def do_POST(self):
        if not self.path.startswith("/v1/chat/completions"):
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        messages = body.get("messages", [])
        response_format = body.get("response_format")
        max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or 512

        if response_format and response_format.get("type") == "json_object":
            messages = messages + [{"role": "system",
                                     "content": "Respond with ONLY a single valid JSON object. No markdown, no code fences, no explanation."}]

        prompt = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = _tokenizer(prompt, return_tensors="pt").to(_device)
        with torch.no_grad():
            out = _model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                   pad_token_id=_tokenizer.eos_token_id)
        completion_ids = out[0][inputs["input_ids"].shape[1]:]
        text = _tokenizer.decode(completion_ids, skip_special_tokens=True)

        response = {
            "id": "local-qwen",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "local-qwen"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": int(inputs["input_ids"].shape[1]), "completion_tokens": len(completion_ids),
                      "total_tokens": int(inputs["input_ids"].shape[1]) + len(completion_ids)},
        }
        payload = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8420)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    load_model(args.model, args.device)
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Serving on http://127.0.0.1:{args.port}/v1/chat/completions")
    server.serve_forever()


if __name__ == "__main__":
    main()
