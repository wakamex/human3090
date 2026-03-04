# Job examples

Copy and modify these when creating new benchmark jobs.

## Thinking model defaults

Thinking models (Qwen3.5, QwQ, DeepSeek-R1, DeepCoder, Nemotron, IQuest-Thinking)
MUST use these settings:

    max_tokens: 32000
    context_size: 32768
    save_raw: true

Without these, the model spends all tokens on reasoning and never produces code.
The job loader will warn if a thinking model has max_tokens < 32000.
