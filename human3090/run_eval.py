import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from human3090.human_eval.data import read_problems
from openai import OpenAI
from pydantic_core import ValidationError
from together import Together

from .bench_constants import DEFAULT_VALUES

# pylint: disable=redefined-outer-name, line-too-long, missing-module-docstring, invalid-name, import-outside-toplevel
# ruff: noqa: E501

load_dotenv(".env")
OPENAI_KEY = os.getenv("OPENAI_KEY")
MISTRAL_KEY = os.getenv("MISTRAL_KEY")
HUGGINGFACE_KEY = os.getenv("HUGGINGFACE_KEY")
TOGETHER_KEY = os.getenv("TOGETHER_KEY")
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_KEY")


def clear_output_interactive():
    """Clear output in an interactive environment."""
    from IPython.display import clear_output

    clear_output(wait=True)


def clear_output_non_interactive():
    """Clear output in a non-interactive environment."""
    if sys.platform == "win32":  # For Windows
        os.system("cls")
    else:  # For macOS and Linux
        os.system("clear")


def clear_output_robust():
    """Clear output irrespective of OS, and whether running interactively or non-interactively."""
    if "ipykernel" in sys.modules:
        clear_output_interactive()
    else:
        clear_output_non_interactive()


def sanitize_answer(raw_answer):
    """Sanitize the answer to extract code from LLM outputs."""
    # Handle thinking LLMs
    if "<think>" in raw_answer and "</think>" in raw_answer:
        raw_answer = raw_answer.split("</think>", 1)[1].strip()
    elif "<think>" in raw_answer:
        # Unclosed think tag - only salvage if there's a proper code block
        # (loose def/class in thinking text are just reasoning fragments, not runnable code)
        after_think = raw_answer.rsplit("<think>", 1)[1]
        if "```python" in after_think:
            raw_answer = after_think
        else:
            return ""

    # Try to extract code from markdown code blocks
    if "```python" in raw_answer and "```" in raw_answer.split("```python", 1)[1]:
        return raw_answer.split("```python", 1)[1].split("```", 1)[0].strip()

    # Also try bare ``` code blocks
    if "```\n" in raw_answer and raw_answer.count("```") >= 2:
        blocks = raw_answer.split("```")
        for i in range(1, len(blocks), 2):
            block = blocks[i].strip()
            if block and ("def " in block or "class " in block):
                # Strip optional language identifier on first line
                lines = block.split('\n')
                if lines[0].strip() in ('python', 'py', ''):
                    block = '\n'.join(lines[1:])
                return block.strip()

    # Line-by-line extraction
    code_lines, in_code = [], False

    for line in raw_answer.splitlines():
        stripped = line.strip()

        # Code indicators
        if line.lstrip().startswith(("def ", "import ", "from ", "class ")):
            in_code = True
            code_lines.append(line)
        # Code blocks
        elif "```python" in line: in_code = True
        elif line.startswith("```"): in_code = False
        # Indented lines or blank lines within code
        elif in_code and (line.startswith(" ") or line.startswith("\t") or stripped == ""):
            code_lines.append(line)
        # End code mode on non-indented text
        elif stripped and not (line.startswith(" ") or line.startswith("\t")):
            in_code = False

    return "\n".join(code_lines).strip()


def _detect_loop(text, window=30, min_unique=10):
    """Detect loops: if fewer than `min_unique` unique lines in the last `window` lines."""
    lines = text.split('\n')
    if len(lines) < window:
        return False
    recent = [l.strip() for l in lines[-window:] if l.strip()]
    if len(recent) < window // 2:
        return False
    return len(set(recent)) < min_unique


def parse_completion_stream(completion_stream, prompt, task_id, end_after_n_codeblocks=None, framework="ai"):
    """Parse a completion stream and return the response."""
    response = []
    in_reasoning = False
    finished = False
    loop_check_counter = 0
    while not finished:
        try:
            if framework == "ai":
                chunk = next(completion_stream)
                delta = chunk.choices[0].delta
                # Handle reasoning_content (thinking models like Qwen3.5)
                reasoning = getattr(delta, 'reasoning_content', None)
                if reasoning:
                    if not in_reasoning:
                        response.append("<think>")
                        in_reasoning = True
                    response.append(reasoning)
                    # Periodically check for loops in reasoning (every ~500 tokens)
                    loop_check_counter += 1
                    if loop_check_counter >= 500:
                        loop_check_counter = 0
                        if _detect_loop("".join(response)):
                            print(f"\n[loop detected, stopping early]", flush=True)
                            response.append("</think>")
                            in_reasoning = False
                            finished = True
                            continue
                text = delta.content
                if text and in_reasoning:
                    response.append("</think>")
                    in_reasoning = False
            else:
                text = next(completion_stream)
            if text:
                response.append(text)
                # Check for loops in content too
                loop_check_counter += 1
                if loop_check_counter >= 500:
                    loop_check_counter = 0
                    if _detect_loop("".join(response)):
                        print(f"\n[loop detected in content, stopping early]", flush=True)
                        finished = True
                        continue
                clear_output_robust()
                if task_id:
                    print(f"{task_id}\n")
                print(prompt, flush=True)
                print("".join(response), flush=True)
                if end_after_n_codeblocks:
                    num_code_blocks = 0
                    for line in "".join(response).splitlines():
                        if line.lstrip(" ").startswith("```"):
                            num_code_blocks += 1
                            if num_code_blocks == end_after_n_codeblocks:
                                finished = True
                        if "</|im_end|>" in line:
                            finished = True
        except (StopIteration, ValidationError):
            finished = True
    return "".join(response)


def hf(prompt, model, temperature=0.8, task_id=None, end_after_n_codeblocks=None):
    client = InferenceClient(model=model, token=HUGGINGFACE_KEY)
    completion_stream = client.text_generation(prompt=prompt, stream=True, max_new_tokens=1_000, temperature=temperature)
    return parse_completion_stream(completion_stream=completion_stream, prompt=prompt, task_id=task_id, end_after_n_codeblocks=end_after_n_codeblocks, framework="hf")

def make_kwargs(temperature=0.8, frequency_penalty=None, presence_penalty=None, top_p=None, min_p=None):
    kwargs = {"temperature": temperature}
    if presence_penalty is not None:
        kwargs["presence_penalty"] = presence_penalty
    if frequency_penalty is not None:
        kwargs["frequency_penalty"] = frequency_penalty
    if top_p is not None:
        kwargs["top_p"] = top_p
    # min_p is intentionally not added to kwargs as it's not supported by the OpenAI client's create method
    return kwargs

def make_kwargs_o3(frequency_penalty=None, presence_penalty=None):
    kwargs = {}
    if presence_penalty is not None:
        kwargs["presence_penalty"] = presence_penalty
    if frequency_penalty is not None:
        kwargs["frequency_penalty"] = frequency_penalty
    return kwargs

def together(prompt, model, system=None, task_id=None, temperature=0.8, frequency_penalty=None, presence_penalty=None):
    client = Together(api_key=TOGETHER_KEY)
    messages = [{"role": "user", "content": prompt}]
    if system:
        messages = [{"role": "system", "content": system}] + messages
    kwargs = make_kwargs(temperature=temperature, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)
    # response = client.chat.completions.create(model=model, messages=messages, stream=True, max_tokens=1_000, **kwargs)
    # print(response.choices[0].message.content)
    completion_stream = client.chat.completions.create(model=model, messages=messages, stream=True, max_tokens=1_000, **kwargs)
    return parse_completion_stream(completion_stream=completion_stream, prompt=prompt, task_id=task_id)

def ai(prompt, system=None, url="http://127.0.0.1:8083/v1", model="llama!", key="na", temperature=DEFAULT_VALUES["--temperature"], top_p=DEFAULT_VALUES["--top-p"], min_p=DEFAULT_VALUES["--min-p"], max_tokens=DEFAULT_VALUES["--max-tokens"], frequency_penalty=None, presence_penalty=None, task_id=None):
    client = OpenAI(base_url=url, api_key=key)
    messages = [{"role": "user", "content": prompt}]
    if system:
        messages = [{"role": "system", "content": system}] + messages
    kwargs = make_kwargs(temperature=temperature, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty, top_p=top_p, min_p=min_p)
    completion_stream = client.chat.completions.create(model=model, messages=messages, stream=True, max_tokens=max_tokens, **kwargs)
    return parse_completion_stream(completion_stream=completion_stream, prompt=prompt, task_id=task_id)

def ai_o3(prompt, system=None, url="http://127.0.0.1:8083/v1", model="llama!", key="na", frequency_penalty=None, presence_penalty=None, task_id=None):
    client = OpenAI(base_url=url, api_key=key)
    messages = [{"role": "user", "content": prompt}]
    if system:
        messages = [{"role": "system", "content": system}] + messages
    kwargs = make_kwargs_o3(frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)
    completion_stream = client.chat.completions.create(model=model, messages=messages, stream=True, **kwargs)
    return parse_completion_stream(completion_stream=completion_stream, prompt=prompt, task_id=task_id)

def main(model, temperature, top_p, min_p, preamble = "Please continue to complete the function.\n```python\n", max_tokens = 1_000, start_problem = 1, end_problem = None):
    # Get model shortname (e.g. 'smollm2-1.7b-instruct-q4_k_m' from 'smollm2-1.7b-instruct-q4_k_m.gguf')
    model_shortname = os.path.splitext(os.path.basename(model))[0]
    problems = read_problems()
    keys = list(problems.keys())
    subset = {key: problems[key] for key in keys[start_problem-1:end_problem]}
    start_time = time.time()
    for task_id in subset:
        raw_prompt = problems[task_id]["prompt"]
        prompt = preamble + raw_prompt
        raw_answer = ai(prompt=prompt,model=model,temperature=temperature,top_p=top_p,min_p=min_p,max_tokens=max_tokens,task_id=task_id)

        # sanitize answer, and append it to the jsonl file
        with open(f"{model_shortname}_human_eval.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(task_id=task_id, completion=sanitize_answer(raw_answer))))
            f.write("\n")
    print(f"finished in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HumanEval benchmark with specified parameters")
    parser.add_argument("--model", required=True, help="Model file path")
    parser.add_argument("--temperature", type=float, default=float(DEFAULT_VALUES["--temperature"]), help="Temperature for sampling")
    parser.add_argument("--top-p", type=float, default=float(DEFAULT_VALUES["--top-p"]), help="Top-p for sampling")
    parser.add_argument("--min-p", type=float, default=float(DEFAULT_VALUES["--min-p"]), help="Minimum p for sampling")
    parser.add_argument("--preamble", default="Please continue to complete the function.\n```python\n", help="Optional preamble text")
    parser.add_argument("--max-tokens", type=int, default=int(DEFAULT_VALUES["--max-tokens"]), help="Maximum tokens per completion")
    parser.add_argument("--start-problem", type=int, default=int(DEFAULT_VALUES["--start-problem"]), help="Problem index to start from (1-based)")
    parser.add_argument("--end-problem", type=int, default=None, help="Problem index to end at (1-based)")
    parser.add_argument("--stream", action="store_true", help="Use streaming output for sglang server")

    args = parser.parse_args()
    main(
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        min_p=args.min_p,
        preamble=args.preamble,
        max_tokens=args.max_tokens,
        start_problem=args.start_problem,
        end_problem=args.end_problem
    )
