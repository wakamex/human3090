# Preferred Sampling Parameters by Model

Recommended params for coding/LCB benchmarks, sourced from each model's HuggingFace page.

## Qwen3.5 (27B, 35B-A3B, 9B)

Source: https://huggingface.co/Qwen/Qwen3.5-27B, https://huggingface.co/Qwen/Qwen3.5-35B-A3B

Thinking mode (default) for precise coding tasks:
- `temperature: 0.6`
- `top_p: 0.95`
- `top_k: 20`
- `min_p: 0.0`
- `enable_thinking: true`
- `context_size: 32768` (minimum for thinking; native support up to 262K)
- `max_tokens: 10000` (HF recommends 32768 general, 81920 for competitions)

## QwQ-32B

Source: https://huggingface.co/Qwen/QwQ-32B

Thinks by default — no `enable_thinking` flag needed.
- `temperature: 0.6`
- `top_p: 0.95`
- `top_k: 20`
- `min_p: 0.0`
- `context_size: 32768`
- `max_tokens: 10000`

Note: Avoid greedy decoding (temp=0) to prevent endless repetitions.

## IQuest-Coder-V1-14B-Thinking

Source: https://huggingface.co/IQuestLab/IQuest-Coder-V1-14B-Thinking

- `temperature: 1.0`
- `top_p: 0.95`
- `top_k: 20`
- `backend: hf` (no GGUF available, requires transformers 4.50-4.53)

## Jan-code-4b (Qwen3-4B fine-tune)

Source: https://huggingface.co/janhq/Jan-code-4b

- `temperature: 0.7`
- `top_p: 0.8`
- `top_k: 20`
- `min_p: 0.0`
- `context_size: 32768`
- `enable_thinking: true` (Qwen3-based)
