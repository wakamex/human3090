#!/usr/bin/env python3
import json
import os
import sys
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

def main():
    parser = argparse.ArgumentParser(description='Run LCB problems')
    parser.add_argument('problems_file', type=str, help='Problems file (e.g., test5.jsonl)')
    parser.add_argument('--start-date', type=str, help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end-date', type=str, help='End date in YYYY-MM-DD format')
    parser.add_argument('--start-problem', type=int, default=1, help='Starting problem')
    args = parser.parse_args()

    # Read all problems
    problems = read_lcb_problems(args.problems_file)
    filtered_problems = {}

    # Filter by date range if dates are provided
    if args.start_date or args.end_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date() if args.start_date else datetime.min.date()
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date() if args.end_date else datetime.max.date()

        for task_id, problem in problems.items():
            problem_date = datetime.strptime(problem['contest_date'].split('T')[0], '%Y-%m-%d').date()
            if start_date <= problem_date <= end_date:
                filtered_problems[task_id] = problem

        date_range = f" between {start_date} and {end_date}"
    else:
        filtered_problems = problems
        date_range = ""

    if not filtered_problems:
        print(f"No problems found{date_range}")
        return

    # Sort and apply start_problem
    keys = sorted(filtered_problems.keys())
    subset = {key: filtered_problems[key] for key in keys[args.start_problem-1:]}

    print(f"Found {len(filtered_problems)} problems{date_range}")
    print(f"Processing {len(subset)} problems starting from index {args.start_problem}")

    start_time = time.time()

    # Process problems
    for task_id, problem in subset.items():
        prompt = create_prompt(problem)
        print(f"{task_id=}")

        # ./build/bin/llama-server -m /seagate/models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf --port 8084 -ngl 80
        url = "http://127.0.0.1:8084/v1"
        model = "test_results"
        temperature = 0.6
        max_tokens = 20_000
        raw_answer = ai(prompt=prompt, url=url, model=model, temperature=temperature, max_tokens=max_tokens, task_id=task_id)
        sanitized_answer = sanitize_answer(raw_answer)

        result = {
            'task_id': task_id,
            'completion': sanitized_answer,
            'difficulty': problem['difficulty']
        }
        with open(f"{model}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(result))
            f.write("\n")

    print(f"finished in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
