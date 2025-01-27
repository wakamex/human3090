#!/usr/bin/env python
import json
import sys
import ast
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Union
import contextlib
import io
import os
import signal
import tempfile
from multiprocessing import Manager, Process
import tqdm
import numpy as np
import itertools

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
    """StringIO that throws an exception when it's read from"""
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
    """Extract the name of the first function defined in the code."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return node.name
    except:
        return "solve"  # Default fallback
    return "solve"  # Default fallback

def unsafe_execute(completion: str, test_case: Dict[str, Any], timeout: float, result: List):
    """Execute untrusted code safely."""
    with create_tempdir():
        try:
            # Create namespace and exec the completion
            namespace = {}
            with swallow_io():
                with time_limit(timeout):
                    # First execute the completion to define the function
                    exec(completion, namespace)

                    # Get the function name from the completion
                    func_name = extract_function_name(completion)
                    if func_name not in namespace:
                        result.append(f"failed: Function {func_name} not found")
                        return

                    func = namespace[func_name]

                    # Parse inputs and run function
                    input_lines = test_case['input'].strip().split('\n')
                    if len(input_lines) != 2:
                        result.append(f"failed: Expected 2 input lines, got {len(input_lines)}")
                        return

                    # Parse inputs as Python literals
                    coins = json.loads(input_lines[0])
                    k = json.loads(input_lines[1])

                    # Run function and compare output
                    actual = str(func(coins, k))
                    expected = test_case['output'].strip()

                    if actual == expected:
                        result.append("passed")
                    else:
                        result.append(f"failed: Expected {expected}, got {actual}")

        except TimeoutException:
            result.append("timed out")
        except BaseException as e:
            result.append(f"failed: {e}")

def check_correctness(completion: str, test_case: Dict[str, Any], timeout: float, completion_id: int = None) -> Dict:
    """
    Evaluates the functional correctness of a completion by running the test case.
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

        return dict(
            task_id=test_case.get("task_id", "unknown"),
            difficulty=test_case.get("difficulty", "unknown"),
            passed=result[0] == "passed",
            result=result[0],
            completion_id=completion_id,
        )

def evaluate_functional_correctness(
    sample_file: str,
    k: List[int] = [1, 10, 100],
    n_workers: int = 4,
    timeout: float = 3.0,
):
    """
    Evaluates the functional correctness of generated samples.
    Returns pass@k metrics and writes detailed results to f"{sample_file}_results.jsonl"
    """
    # Check the generated samples against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        print("Reading samples...")
        with open(sample_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                task_id = data["task_id"]
                completion = data["completion"]

                # Load the original problem
                with open("test5.jsonl", "r", encoding="utf-8") as f2:
                    for line2 in f2:
                        prob = json.loads(line2)
                        if prob["question_id"] == task_id.split('/')[-1]:
                            test_cases = json.loads(prob["public_test_cases"])
                            for test_case in test_cases:
                                test_case["task_id"] = task_id
                                test_case["difficulty"] = prob["difficulty"]
                                args = (completion, test_case, timeout, completion_id[task_id])
                                future = executor.submit(check_correctness, *args)
                                futures.append(future)
                                completion_id[task_id] += 1
                                n_samples += 1
                            break

        print("Running test suites...")
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    # Calculate pass@k.
    total, correct = [], []
    for result in results.values():
        result.sort()
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    ks = k
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
                 for k in ks if (total >= k).all()}

    # Save results in same format as original samples
    def combine_results():
        with open(sample_file, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line)
                task_id = sample["task_id"]
                if task_id in results:
                    result = results[task_id].pop(0)
                    sample["result"] = result[1]["result"]
                    sample["passed"] = result[1]["passed"]
                    sample["difficulty"] = result[1]["difficulty"]
                yield sample

    # Save detailed results
    out_file = f"{sample_file}_results.jsonl"
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
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
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
    if len(sys.argv) != 2:
        print("Usage: python evaluate_lcb.py <completion_file>")
        sys.exit(1)

    results = evaluate_functional_correctness(sys.argv[1])
    print(f"Results: {results}")

if __name__ == "__main__":
    main()
