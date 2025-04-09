#!/usr/bin/env python
"""Test a single HumanEval problem with a running LLM server."""

import argparse
import json
import sys
import time

from human_eval.data import read_problems
from openai import OpenAI

from .run_eval import sanitize_answer


def ai(prompt, system=None, url="http://127.0.0.1:8083/v1", model="llama!", key="na", 
       temperature=0.8, max_tokens=1_000, frequency_penalty=None, presence_penalty=None):
    """Call the AI with the given prompt and return the raw response."""
    client = OpenAI(base_url=url, api_key=key)
    messages = [{"role": "user", "content": prompt}]
    if system:
        messages = [{"role": "system", "content": system}] + messages

    kwargs = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if frequency_penalty is not None:
        kwargs["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        kwargs["presence_penalty"] = presence_penalty

    response = client.chat.completions.create(
        model=model, 
        messages=messages, 
        stream=False,  # No streaming for easier debugging
        max_tokens=max_tokens, 
        **kwargs
    )

    return response.choices[0].message.content

# Custom wrapper for sanitize_answer to add debug printing
def debug_sanitize_answer(raw_answer):
    """Wrap sanitize_answer to add debug printing."""
    print("=== RAW ANSWER ===")
    print(raw_answer)
    print("=== END RAW ANSWER ===")

    sanitized = sanitize_answer(raw_answer)

    print("=== SANITIZED ANSWER ===")
    print(sanitized)
    print("=== END SANITIZED ANSWER ===")

    return sanitized

def main():
    parser = argparse.ArgumentParser(description="Test a single HumanEval problem")
    parser.add_argument("--problem", required=True, help="Problem ID (e.g., '0' for HumanEval/0)")
    parser.add_argument("--model", default="llama!", help="Model name for the server")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling")
    parser.add_argument("--preamble", default="Please continue to complete the function.\n```python\n", 
                       help="Optional preamble text")
    parser.add_argument("--max-tokens", type=int, default=1000, help="Maximum tokens per completion")
    parser.add_argument("--url", default="http://127.0.0.1:8083/v1", help="URL of the LLM server")

    args = parser.parse_args()

    # Read problems
    problems = read_problems()

    # Find the requested problem
    problem_id = None
    for task_id in problems:
        if task_id.split("/")[-1] == args.problem:
            problem_id = task_id
            break

    if not problem_id:
        print(f"Problem {args.problem} not found")
        sys.exit(1)

    # Get the problem prompt
    raw_prompt = problems[problem_id]["prompt"]
    prompt = args.preamble + raw_prompt

    print(f"Testing problem: {problem_id}")
    print("=== PROMPT ===")
    print(prompt)
    print("=== END PROMPT ===")

    # Get the model's response
    start_time = time.time()
    raw_answer = ai(
        prompt=prompt,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        url=args.url
    )
    elapsed = time.time() - start_time

    # Process the answer
    sanitized = debug_sanitize_answer(raw_answer)

    # Save the result
    result = {
        "task_id": problem_id,
        "completion": sanitized,
        "raw_completion": raw_answer,
        "time_taken": elapsed
    }

    output_file = f"test_problem_{args.problem}_result.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Result saved to {output_file}")
    print(f"Completed in {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
