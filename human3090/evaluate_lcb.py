#!/usr/bin/env python
import argparse
import ast
import contextlib
import io
import itertools
import json
import os
import signal
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Manager, Process
from typing import Any, Dict, List, Union

import numpy as np
import tqdm


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield

@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname

class TimeoutException(Exception):
    pass

class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from."""

    def read(self, *args, **kwargs):
        raise IOError
    def readline(self, *args, **kwargs):
        raise IOError
    def readlines(self, *args, **kwargs):
        raise IOError
    def readable(self, *args, **kwargs):
        return False

class redirect_stdin(contextlib._RedirectStream):
    _stream = 'stdin'

@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(cwd)

def extract_function_name(code: str) -> str:
    """Extract the name of any suitable function defined in the code."""
    if not code or not code.strip():
        return None
    try:
        # Remove any leading empty lines and normalize line endings
        clean_code = '\n'.join(line.rstrip() for line in code.split('\n')).strip()
        tree = ast.parse(clean_code)
        # Find all function definitions
        functions = [
            node.name 
            for node in ast.walk(tree) 
            if isinstance(node, ast.FunctionDef)
        ]
        # Skip main() function if there are other options
        if len(functions) > 1:
            non_main = [f for f in functions if f != "main"]
            if non_main:
                return non_main[-1]  # Return the last non-main function
        # Return last function found, or None if no functions
        return functions[-1] if functions else None
    except:
        return None  # Failed to parse code

def extract_code_from_text(text: str) -> str:
    """Extract valid Python code from text that may contain explanations."""
    if not text or not text.strip():
        return ""

    lines = text.split('\n')
    # Find first line that starts with def, import, from, or class
    for i, line in enumerate(lines):
        if line.startswith(('def ', 'import ', 'from ', 'class ')):
            code = '\n'.join(lines[i:])
            # Add missing imports
            imports = []
            if 'defaultdict' in code and 'from collections import defaultdict' not in code:
                imports.append('from collections import defaultdict')
            if 'bisect_left' in code and 'from bisect import bisect_left' not in code:
                imports.append('from bisect import bisect_left')
            return '\n'.join(imports + [''] + [code] if imports else [code])
    return ""

def unsafe_execute(completion: str, test_case: Dict[str, Any], timeout: float, result: List):
    """Execute untrusted code safely."""
    if not completion or not completion.strip():
        result.append("failed: Empty solution")
        return

    # Try to extract just the code if there's explanatory text
    completion = extract_code_from_text(completion)
    if not completion:
        result.append("failed: No valid code found")
        return

    testtype = test_case.get('testtype', 'stdin')

    with create_tempdir():
        try:
            # Create namespace and exec the completion
            namespace = {}
            with swallow_io():
                with time_limit(timeout):
                    try:
                        exec(completion, namespace)
                    except IndentationError as e:
                        result.append(f"failed: Code indentation error - {str(e)}")
                        return
                    except SyntaxError as e:
                        result.append(f"failed: Code syntax error - {str(e)}")
                        return

                    if testtype == 'stdin':
                        # AtCoder: call solve(input_text) -> str
                        func_name = extract_function_name(completion)
                        if not func_name:
                            result.append("failed: No function found in solution")
                            return
                        if func_name not in namespace:
                            result.append(f"failed: Function {func_name} not found")
                            return
                        func = namespace[func_name]

                        actual = str(func(test_case['input']))
                        expected = test_case['output'].strip()

                        if actual.strip() == expected:
                            result.append("passed")
                        else:
                            result.append(f"failed: Expected {expected}, got {actual.strip()}")
                    else:
                        # LeetCode: handle class Solution or standalone function
                        if 'Solution' in namespace:
                            sol = namespace['Solution']()
                            method_name = test_case.get('method_name')
                            if method_name and hasattr(sol, method_name):
                                func = getattr(sol, method_name)
                            else:
                                # Fallback: find user-defined method
                                methods = [m for m in dir(sol)
                                           if not m.startswith('_') and callable(getattr(sol, m))]
                                if not methods:
                                    result.append("failed: No method found in Solution class")
                                    return
                                func = getattr(sol, methods[0])
                        else:
                            func_name = extract_function_name(completion)
                            if not func_name or func_name not in namespace:
                                result.append("failed: No function found in solution")
                                return
                            func = namespace[func_name]

                        # Parse JSON args from input lines
                        input_lines = test_case['input'].strip().split('\n')
                        args = []
                        for line in input_lines:
                            try:
                                arg = json.loads(line)
                                args.append(arg)
                            except json.JSONDecodeError:
                                args.append(line)

                        actual = func(*args)
                        expected_str = test_case['output'].strip()
                        try:
                            expected = json.loads(expected_str)
                        except json.JSONDecodeError:
                            expected = expected_str

                        if actual == expected:
                            result.append("passed")
                        elif str(actual) == str(expected):
                            result.append("passed")
                        else:
                            result.append(f"failed: Expected {expected}, got {actual}")

        except TimeoutException:
            result.append("timed out")
        except json.JSONDecodeError as e:
            result.append(f"failed: Invalid input format - {str(e)}")
        except TypeError as e:
            result.append(f"failed: Wrong number of arguments - {str(e)}")
        except BaseException as e:
            result.append(f"failed: {e}")

def check_correctness(completion: str, test_case: Dict[str, Any], timeout: float, completion_id: int = None) -> Dict:
    """Evaluate the functional correctness of a completion by running the test case.

    Uses multiprocessing for safety.
    """
    with Manager() as manager:
        result = manager.list()

        p = Process(target=unsafe_execute, args=(completion, test_case, timeout, result))
        p.start()
        p.join(timeout=timeout + 1)
        if p.is_alive():
            p.kill()

        if not result:
            result.append("timed out")

        passed = result[0] == "passed"
        return dict(
            task_id=test_case.get("task_id", "unknown"),
            difficulty=test_case.get("difficulty", "unknown"),
            passed=passed,
            completion_id=completion_id,
            completion=completion,  # Store completion for output
        )

def evaluate_functional_correctness(
    solutions: str,
    test_file: str,
    k: List[int] = [1, 10, 100],
    n_workers: int = 4,
    timeout: float = 3.0,
    debug: bool = False,
):
    """Evaluate the functional correctness of generated samples.

    Returns pass@k metrics and writes detailed results to f"{solutions}_results.jsonl"
    """
    # Load all test cases first
    print("Reading test cases...")
    test_cases_by_id = {}
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            prob = json.loads(line)
            test_cases = json.loads(prob["public_test_cases"])
            # Extract method name from starter_code for LeetCode problems
            method_name = None
            starter = prob.get("starter_code", "")
            if starter:
                for sc_line in starter.split('\n'):
                    sc_stripped = sc_line.strip()
                    if sc_stripped.startswith("def ") and sc_stripped != "def __init__":
                        method_name = sc_stripped.split("(")[0].replace("def ", "")
                        break
            for tc in test_cases:
                if method_name:
                    tc["method_name"] = method_name
            test_cases_by_id[prob["question_id"]] = {
                "test_cases": test_cases,
                "difficulty": prob.get("difficulty", "unknown")
            }

    # Load and process all samples
    print("Reading samples...")
    all_task_ids = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        with open(solutions, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                task_id = data["task_id"]
                all_task_ids.append(task_id)
                completion = data["completion"]
                question_id = task_id.split('/')[-1]

                if question_id in test_cases_by_id:
                    test_info = test_cases_by_id[question_id]
                    if debug:
                        print(f"\nDebug - Task {task_id}:")
                        print(f"  Number of test cases: {len(test_info['test_cases'])}")
                    for i, test_case in enumerate(test_info['test_cases']):
                        if debug:
                            print(f"  Test case {i}: {test_case}")
                        test_case["task_id"] = task_id
                        test_case["difficulty"] = test_info["difficulty"]
                        args = (completion, test_case, timeout, completion_id[task_id])
                        future = executor.submit(check_correctness, *args)
                        futures.append(future)
                        completion_id[task_id] += 1
                        n_samples += 1

        print("Running test suites...")
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    # Process results in task ID order
    total, correct = [], []
    unique_task_ids = list(dict.fromkeys(all_task_ids))  # preserve order, remove duplicates
    for task_id in unique_task_ids:
        if task_id in results:
            result = results[task_id]
            result.sort()
            passed = [r[1]["passed"] for r in result]
            if debug:
                print(f"\nDebug - Task {task_id}:")
                print(f"  Result: {result}")
                print(f"  Passed array: {passed}")
                print(f"  Total test cases: {len(passed)}")
                print(f"  All test cases passed: {all(passed)}")
            total.append(1)  # Each task counts as 1 attempt
            correct.append(1 if all(passed) else 0)  # Task is correct only if all test cases pass
        else:
            if debug:
                print(f"\nDebug - Task {task_id}: No results")
            # Empty solution or failed to parse
            total.append(1)  # One attempt
            correct.append(0)  # Failed attempt

    total = np.array(total)
    correct = np.array(correct)

    if debug:
        print(f"\nDebug - Unique tasks: {len(total)}")
        print(f"Debug - Total tasks: {sum(total)}")
        print(f"Debug - Tasks with all test cases passing: {sum(correct)}")
        print(f"Debug - Raw correct array: {correct}")
        print(f"Debug - Raw total array: {total}")

    ks = k
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
                 for k in ks if (total >= k).all()}

    # Save results in same format as HumanEval
    def combine_results():
        seen_task_ids = set()
        for task_id in all_task_ids:
            if task_id in results and task_id not in seen_task_ids:
                seen_task_ids.add(task_id)
                for completion_id, result in sorted(results[task_id]):
                    # Match HumanEval format while preserving multiple completions
                    yield {
                        "task_id": task_id,
                        "completion": result["completion"],
                        "passed": result["passed"],
                        "difficulty": result["difficulty"]
                    }

    # Save detailed results
    out_file = f"{solutions}_results.jsonl"
    print(f"Writing results to {out_file}...")
    with open(out_file, "w", encoding="utf-8") as f:
        for sample in tqdm.tqdm(combine_results(), total=n_samples):
            f.write(json.dumps(sample) + "\n")

    return pass_at_k

def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """Estimate pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculate 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("solutions", help="Path to solutions")
    parser.add_argument("test_file", help="Path to test cases")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 10, 100], help="Values of k for pass@k")
    parser.add_argument("--n-workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--timeout", type=float, default=3.0, help="Timeout in seconds")
    parser.add_argument("--debug", action="store_true", help="Enable debug prints")
    args = parser.parse_args()

    results = evaluate_functional_correctness(
        args.solutions,
        args.test_file,
        k=args.k,
        n_workers=args.n_workers,
        timeout=args.timeout,
        debug=args.debug,
    )

    print(f"Results: {results}")

if __name__ == "__main__":
    main()
