#!/usr/bin/env python3
import json
import os
import time
from typing import Dict
import argparse
from datetime import datetime

from dotenv import load_dotenv

from run_eval import ai, sanitize_answer

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

def create_prompt(problem: Dict) -> str:
    """Create a prompt for the model from an LCB problem."""
    # Format problem description and test cases
    description = f"# {problem['question_title']}\n\n"
    content = problem['question_content']

    # Extract parameter names from first example input
    param_names = []
    for line in content.split('\n'):
        if line.startswith('Input:'):
            # Extract variable names from "Input: var1 = val1, var2 = val2"
            params = line.split(':', 1)[1].strip()
            if not param_names:
                param_names = [p.split('=')[0].strip() for p in params.split(', ')]
            description += line.replace('Input: ', 'solve(') + ')\n'
        else:
            description += f"{line.replace('Output: ', '')}\n"

    # Function template
    description += "\nComplete this function:\n\n"
    description += f"def solve({', '.join(param_names)}) -> str:\n"
    description += '    """Solve the problem.\n'
    description += "    Args:\n"
    description += "        input_text: problem input as a single string with newlines\n"
    description += "    Returns:\n"
    description += "        solution output as a string with newlines\n"
    description += '    """\n'
    return description

def main(model: str,
         temperature: float = 0.0,
         max_tokens: int = 1_000,
         start_problem: int = 1,
         end_problem: int = None,
         problems_file: str = "test5.jsonl",
         start_date: str = None,
         end_date: str = None,
         stream: bool = False):
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

    # Process problems
    for task_id, problem in subset.items():
        prompt = create_prompt(problem)
        print(f"{task_id=}")

        url = "http://127.0.0.1:8083/v1"
        raw_answer = ai(prompt=prompt, url=url, model=model, temperature=temperature, max_tokens=max_tokens, task_id=task_id, stream=stream)
        sanitized_answer = sanitize_answer(raw_answer)

        result = {
            'task_id': task_id,
            'completion': sanitized_answer,
            'difficulty': problem['difficulty']
        }
        # Get model shortname (e.g. 'smollm2-1.7b-instruct-q4_k_m' from 'smollm2-1.7b-instruct-q4_k_m.gguf')
        model_shortname = os.path.splitext(os.path.basename(model))[0]
        with open(f"{model_shortname}_lcb.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(result))
            f.write("\n")

    print(f"finished in {time.time() - start_time:.2f}s")

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
    parser.add_argument("--stream", action="store_true", help="Enable streaming mode")

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
        stream=args.stream
    )
