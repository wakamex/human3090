#!/usr/bin/env python3
import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict

from dotenv import load_dotenv

from human3090.run_eval import ai, sanitize_answer

# Load environment variables
load_dotenv(".env")
OPENAI_KEY = os.getenv("OPENAI_KEY")

def read_lcb_problems(filename: str) -> Dict:
    """Read LCB problems from a JSONL file."""
    problems = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            problem = json.loads(line)
            task_id = f"{problem['platform']}/{problem['contest_id']}/{problem['question_id']}"
            problems[task_id] = problem
    return problems

def create_stdin_prompt(problem: Dict) -> str:
    """Create a prompt for stdin/stdout problems (AtCoder)."""
    tc = json.loads(problem['public_test_cases'])

    prompt = f"# {problem['question_title']}\n\n"
    prompt += problem['question_content'] + "\n\n"

    # Show examples
    prompt += "## Examples\n\n"
    for i, t in enumerate(tc[:3]):
        prompt += f"Input:\n{t['input']}\nOutput:\n{t['output']}\n"

    prompt += "Write a Python function that reads from stdin and prints to stdout.\n"
    prompt += "The function signature must be exactly:\n\n"
    prompt += "```python\n"
    prompt += "def solve(input_text: str) -> str:\n"
    prompt += "```\n\n"
    prompt += "Where `input_text` is the full stdin as a single string, and the return value is the full stdout as a string.\n"
    prompt += "Return ONLY the function definition in a ```python code block.\n"
    return prompt


def create_functional_prompt(problem: Dict) -> str:
    """Create a prompt for functional problems (LeetCode)."""
    tc = json.loads(problem['public_test_cases'])

    prompt = f"# {problem['question_title']}\n\n"
    prompt += problem['question_content'] + "\n\n"

    # Show examples
    prompt += "## Examples\n\n"
    for i, t in enumerate(tc[:3]):
        prompt += f"Input:\n{t['input']}\nOutput:\n{t['output']}\n"

    prompt += "Complete the following Python code:\n\n"
    prompt += "```python\n"
    prompt += problem['starter_code'] + "\n"
    prompt += "```\n\n"
    prompt += "Return the complete class definition in a ```python code block.\n"
    return prompt


def create_prompt(problem: Dict) -> str:
    """Create a platform-aware prompt for the model from an LCB problem."""
    tc = json.loads(problem['public_test_cases'])
    testtype = tc[0].get('testtype', 'stdin')

    if testtype == 'stdin':
        return create_stdin_prompt(problem)
    else:
        return create_functional_prompt(problem)

def main(model: str,
         temperature: float = 0.0,
         max_tokens: int = 1_000,
         top_p: float = 0.9,
         min_p: float = 0.1,
         start_problem: int = 1,
         end_problem: int = None,
         problems_file: str = "test5.jsonl",
         start_date: str = None,
         end_date: str = None,
         save_raw: bool = False):
    """Run LCB benchmark with specified parameters."""

    # Read all problems
    problems = read_lcb_problems(problems_file)
    filtered_problems = {}

    # Filter by date range if dates are provided
    if start_date or end_date:
        start = datetime.strptime(start_date, '%Y-%m-%d').date() if start_date else datetime.min.date()
        end = datetime.strptime(end_date, '%Y-%m-%d').date() if end_date else datetime.max.date()

        for task_id, problem in problems.items():
            problem_date = datetime.strptime(problem['contest_date'].split('T')[0], '%Y-%m-%d').date()
            if start <= problem_date <= end:
                filtered_problems[task_id] = problem

        date_range = f" between {start} and {end}"
    else:
        filtered_problems = problems
        date_range = ""

    if not filtered_problems:
        print(f"No problems found{date_range}")
        return

    # Sort and apply start_problem and end_problem
    keys = sorted(filtered_problems.keys())
    subset = {key: filtered_problems[key] for key in keys[start_problem-1:end_problem]}

    print(f"Found {len(filtered_problems)} problems{date_range}")
    print(f"Processing {len(subset)} problems starting from index {start_problem} and ending at index {end_problem}")

    start_time = time.time()
    model_shortname = os.path.splitext(os.path.basename(model))[0]

    # Process problems
    for task_id, problem in subset.items():
        problem_start = time.time()
        prompt = create_prompt(problem)
        print(f"{task_id=}")

        url = "http://127.0.0.1:8083/v1"
        try:
            raw_answer = ai(prompt=prompt, url=url, model=model, temperature=temperature, max_tokens=max_tokens, top_p=top_p, min_p=min_p, task_id=task_id)
        except Exception as e:
            print(f"Error on {task_id}: {e}")
            raw_answer = ""
        sanitized_answer = sanitize_answer(raw_answer)
        problem_duration = time.time() - problem_start

        result = {
            'task_id': task_id,
            'completion': sanitized_answer,
            'difficulty': problem['difficulty'],
            'time_taken': round(problem_duration, 2)
        }
        with open(f"{model_shortname}_lcb.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(result))
            f.write("\n")

        if save_raw:
            raw_result = {'task_id': task_id, 'raw_response': raw_answer, 'time_taken': round(problem_duration, 2)}
            with open(f"{model_shortname}_lcb_raw.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(raw_result))
                f.write("\n")

    total_time = time.time() - start_time

    # Append metadata summary
    metadata = {
        '_metadata': True,
        'total_time': round(total_time, 2),
        'total_problems': len(subset),
        'model': model,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'top_p': top_p,
        'min_p': min_p,
        'problems_file': problems_file,
        'timestamp': datetime.now().isoformat()
    }
    with open(f"{model_shortname}_lcb.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(metadata))
        f.write("\n")

    print(f"finished in {total_time:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LCB benchmark with specified parameters")
    parser.add_argument("--model", required=True, help="Model file path")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling")
    parser.add_argument("--max-tokens", type=int, default=1_000, help="Maximum tokens per completion")
    parser.add_argument("--start-problem", type=int, default=1, help="Problem index to start from (1-based)")
    parser.add_argument("--end-problem", type=int, default=None, help="Problem index to end at (1-based)")
    parser.add_argument("--problems-file", default="test5.jsonl", help="Problems file (e.g., test5.jsonl)")
    parser.add_argument("--start-date", help="Start date for LCB problems (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date for LCB problems (YYYY-MM-DD)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p for sampling")
    parser.add_argument("--min-p", type=float, default=0.1, help="Min-p for sampling")
    parser.add_argument("--save-raw", action="store_true", help="Save raw model responses for debugging")

    args = parser.parse_args()

    main(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        start_problem=args.start_problem,
        end_problem=args.end_problem,
        problems_file=args.problems_file,
        start_date=args.start_date,
        end_date=args.end_date,
        top_p=args.top_p,
        min_p=args.min_p,
        save_raw=args.save_raw
    )
