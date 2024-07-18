Unless otherwise specified:
- maximum number of layers offloaded to GPU
- local models run with llama.cpp server and .gguf formats
- parameter changes carried over into following tests (temperature, penalties, etc.)
* denotes non-local for comparison

| Model               | Configuration                                                  | Human Eval |
|---------------------|----------------------------------------------------------------|------------|
| GPT-4*              | Instruction-style, `temperature=0.2`, `presence_penalty=0`     |     63.4%  |
| GPT-4*              | Completion-style                                               |     84.1%  |
| Mixtral8x7b         | mixtral-8x7b-v0.1.Q5_K_M.gguf                                  |     45.7%  |
| Mistral-medium*     |                                                                |     62.2%  |
| Llama2*             | HF API, CodeLlama-34b-Instruct-hf                              |     42.1%  |
| Mistral-large*      |                                                                |     73.2%  |
| WizardLM2           | WizardLM-2-8x22B.IQ3_XS-00001-of-00005.gguf                    |     56.7%  |
| Wizardcoder         | wizardcoder-33b-v1.1.Q4_K_M.gguf, `temperature=0.0`            |     73.8%  |
| Wizardcoder-Python  | Q4_K_M. quant, Modified prompt                                 |     57.3%  |
| CodeFuse-Deepseek   | CodeFuse-DeepSeek-33B-Q4_K_M.gguf                              |     68.7%  |
| Deepseek            | deepseek-coder-33b-instruct.Q4_K_M.gguf                        |     79.9%  |
| OpenCodeInterpreter | ggml-opencodeinterpreter-ds-33b-q8_0.gguf, -ngl 40             |    Failed  |
| Deepseek            | ggml-deepseek-coder-33b-instruct-q4_k_m.gguf                   |     78.7%  |
| Deepseek            | deepseek-coder-33b-instruct.Q5_K_M.gguf, -ngl 60, a bit slow   |     79.3%  |
| Llama3*             | together.ai API, Llama-3-70b-chat-hf                           |     75.6%  |
| DBRX*               | together.ai API, dbrx-instruct                                 |     48.8%  |
| CodeQwen            | codeqwen-1_5-7b-chat-q8_0.gguf                                 |     83.5%  |
| Llama3-8B           | bartowski/Meta-Llama-3-8B-Instruct-GGUF                        |     52.4%  |
| Phi-3-mini          | 4k context, 4bit quantized                                     |     60.4%  |
| Phi-3-mini          | 4k context, fp16 quantized                                     |     62.2%  |
| Hermes-Llama        | Hermes-2-Pro-Llama-3-Instruct-Merged-DPO-F16                   |     53.7%  |
