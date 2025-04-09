import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from human_eval.data import read_problems
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
        return ""  # Unclosed think tag

    # Try to extract code from markdown code blocks
    if "```python" in raw_answer and "```" in raw_answer.split("```python", 1)[1]:
        return raw_answer.split("```python", 1)[1].split("```", 1)[0].strip()

    # Line-by-line extraction
    code_lines, in_code, found_def = [], False, False

    for line in raw_answer.splitlines():
        stripped = line.strip()

        # Code indicators
        if line.lstrip().startswith(("def ", "import ", "from ", "class ")):
            if line.lstrip().startswith("def "):
                if found_def: continue  # Skip duplicate functions
                found_def = True
            in_code = True
            code_lines.append(line)
        # Code blocks
        elif "```python" in line: in_code = True
        elif line.startswith("```"): in_code = False
        # Indented lines
        elif in_code and (line.startswith(" ") or line.startswith("\t")):
            code_lines.append(line)
        # End code mode on non-indented text
        elif stripped and not (line.startswith(" ") or line.startswith("\t")):
            in_code = False

    return "\n".join(code_lines)


def parse_completion_stream(completion_stream, prompt, task_id, end_after_n_codeblocks=None, framework="ai"):
    """Parse a completion stream and return the response."""
    response = []
    finished = False
    while not finished:
        try:
            text = next(completion_stream).choices[0].delta.content if framework == "ai" else next(completion_stream)
            # if text=='':
            #     finished = True
            if text:
                response.append(text)
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

def make_kwargs(temperature=0.8, frequency_penalty=None, presence_penalty=None):
    kwargs = {"temperature": temperature}
    if presence_penalty is not None:
        kwargs["presence_penalty"] = presence_penalty
    if frequency_penalty is not None:
        kwargs["frequency_penalty"] = frequency_penalty
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
    kwargs = make_kwargs(temperature=temperature, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)
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
