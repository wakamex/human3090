#!/usr/bin/env python
"""Generic benchmark runner for LLM evaluation tasks."""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.error import URLError

import requests
from requests.exceptions import ConnectionError

import parse_results

class ReadmeUpdater:
    """Updates README.md with benchmark results."""

    def __init__(self, readme_path: str = "README.md"):
        self.readme_path = readme_path

    def update_table(self, results: Dict[str, Any]):
        """Update the results table in README.md."""
        if not os.path.exists(self.readme_path):
            raise FileNotFoundError(f"README not found: {self.readme_path}")

        with open(self.readme_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Find the appropriate table based on benchmark type
        benchmark = results["benchmark"]
        table_start = None
        for i, line in enumerate(lines):
            if "|" in line:
                if benchmark == "human_eval" and "Human Eval" in line:
                    table_start = i
                    break
                elif benchmark == "lcb" and "test5.jsonl" in line:
                    table_start = i
                    break

        if table_start is None:
            raise ValueError(f"Could not find {benchmark} results table in README.md")

        # Create new row
        model_name = results["model"]
        score = results["results"]["score"]
        time_taken = results["results"]["time_taken"]
        new_row = f"| {model_name} |  | {score:.1f}% | {time_taken:.2f}s |\n"

        # Insert the row after the header
        lines.insert(table_start + 2, new_row)

        # Write back to file
        with open(self.readme_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

class BenchmarkRunner:
    """Manages LLM benchmarks, handling server lifecycle and results storage."""

    def __init__(
        self,
        model_path: str,
        gpu_layers: int,
        server_type: str = "llama",  # "llama" or "sglang"
        results_file: str = "benchmark_results.json",
        server_path: str = "/code/llama.cpp/build/bin/llama-server",
        readme_path: str = "README.md",
        sglang_python_path: Optional[str] = None
    ):
        self.model_path = model_path
        self.gpu_layers = gpu_layers
        self.server_type = server_type
        self.results_file = results_file
        self.server_path = server_path
        self.sglang_python_path = sglang_python_path or "/code/sglang/.venv/bin/python"
        self.readme_updater = ReadmeUpdater(readme_path)

        # Create results file if it doesn't exist
        if not os.path.exists(results_file):
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump({"runs": []}, f, indent=2)

    @contextmanager
    def llama_server(self, port: int = 8083, stream: bool = True):
        """Start llama.cpp server and manage its lifecycle."""
        if not os.path.exists(self.server_path):
            raise FileNotFoundError(f"Server not found at {self.server_path}")

        process = subprocess.Popen([
            self.server_path,
            "-m", self.model_path,
            "--port", str(port),
            "-ngl", str(self.gpu_layers)
        ])

        try:
            # Wait for server to be ready
            start_time = time.time()
            while time.time() - start_time < 60:  # 60 second timeout
                try:
                    # Check health endpoint
                    with urllib.request.urlopen(f'http://localhost:{port}/health') as response:
                        if response.status != 200:
                            time.sleep(1)
                            continue

                    # Try a simple completion to ensure model is loaded
                    data = json.dumps({
                        "messages": [{"role": "user", "content": "Hi"}],
                        "temperature": 0,
                        "max_tokens": 1
                    }).encode('utf-8')
                    req = urllib.request.Request(
                        f'http://localhost:{port}/v1/chat/completions',
                        data=data,
                        headers={'Content-Type': 'application/json'}
                    )
                    with urllib.request.urlopen(req) as response:
                        if response.status == 200:
                            break
                except URLError:
                    # Server not ready yet
                    time.sleep(1)
                    continue
            else:
                raise TimeoutError("Server failed to start within 60 seconds")
            yield port
        finally:
            process.terminate()
            process.wait()

    def _load_results(self) -> Dict[str, Any]:
        """Load existing results from file."""
        with open(self.results_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    @contextmanager
    def sglang_server(self, port: int = 8083, stream = True):
        """Start sglang server and manage its lifecycle."""
        if not os.path.exists(self.sglang_python_path):
            raise FileNotFoundError(f"Python not found at {self.sglang_python_path}")

        kwargs = [
            "-m", "sglang.launch_server",
            "--model-path", self.model_path,
            "--host", "0.0.0.0",
            "--port", str(port)
        ]
        if stream:  # "--stream-output" if stream else nothing
            kwargs.append("--stream-output")
        print(f"running SGlang with {kwargs}")

        server_process = subprocess.Popen([self.sglang_python_path] + kwargs)

        try:
            # Wait for server to be ready
            start_time = time.time()
            while time.time() - start_time < 60:  # 60 second timeout
                try:
                    response = requests.get(f"http://localhost:{port}/get_model_info")
                    if response.status_code == 200:
                        break
                except ConnectionError:
                    time.sleep(1)
                    continue
            else:
                raise TimeoutError("Server failed to start within 60 seconds")
            yield port
        finally:
            server_process.terminate()
            server_process.wait()

    def _save_results(self, results: Dict[str, Any]):
        """Save results to file."""
        class CompactJSONEncoder(json.JSONEncoder):
            def encode(self, obj):
                if isinstance(obj, dict):
                    # Keep failed_tasks list on one line
                    if 'failed_tasks' in obj:
                        failed = obj['failed_tasks']
                        obj = {**obj}
                        del obj['failed_tasks']
                        result = super().encode(obj)[:-1]  # Remove closing brace
                        return result + ', "failed_tasks": ' + self.encode(failed) + '}'
                    return super().encode(obj)
                if isinstance(obj, list):
                    return '[' + ', '.join(self.encode(item) for item in obj) + ']'
                return super().encode(obj)

        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, cls=CompactJSONEncoder)

    def _process_results(self, 
                        benchmark: str,
                        output_file: str,
                        duration: float,
                        cli_args: Dict[str, Any]) -> Dict[str, Any]:
        """Process benchmark results and format for storage."""
        score, details = parse_results.parse_results(output_file)

        # Reconstruct CLI command for reproducibility
        cli_cmd = [".venv/bin/python", "bench_runner.py"]
        for k, v in cli_args.items():
            if v is not None:  # Only include non-None arguments
                k = k.replace('_', '-')  # Convert snake_case to kebab-case
                cli_cmd.extend([f"--{k}", str(v)])

        return {
            "timestamp": datetime.now().isoformat(),
            "benchmark": benchmark,
            "model": Path(self.model_path).stem,
            "command": ' '.join(cli_cmd),  # Store exact command for reproduction
            "results": {
                "score": score,
                "time_taken": duration,
                **details
            }
        }

    def run_benchmark(self, cli_args: Dict[str, Any]) -> Dict[str, Any]:
        """Run a benchmark and store its results."""
        script_path = cli_args['script']
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Benchmark script not found: {script_path}")

        # Run the benchmark
        server_ctx = self.llama_server if self.server_type == "llama" else self.sglang_server
        with server_ctx(stream=cli_args.get('stream', False)) as port:
            start_time = time.time()
            # First run the benchmark script
            cmd = [".venv/bin/python", script_path,
                   "--model", self.model_path,
                   "--temperature", str(cli_args['temperature']),
                   "--max-tokens", str(cli_args['max_tokens'])]
            if cli_args.get('preamble'):
                cmd.extend(["--preamble", cli_args['preamble']])
            if cli_args.get('start_problem', 1) > 1:
                cmd.extend(["--start-problem", str(cli_args['start_problem'])])
            if cli_args.get('end_problem'):
                cmd.extend(["--end-problem", str(cli_args['end_problem'])])

            # Add LCB-specific arguments
            if cli_args['benchmark'] == "lcb":
                if not cli_args.get('problems_file'):
                    raise ValueError("problems_file is required for LCB benchmark")
                cmd.extend(["--problems-file", cli_args['problems_file']])
                if cli_args.get('start_date'):
                    cmd.extend(["--start-date", cli_args['start_date']])
                if cli_args.get('end_date'):
                    cmd.extend(["--end-date", cli_args['end_date']])
            subprocess.run(cmd, check=True)

            # Get model shortname (e.g. 'smollm2-1.7b-instruct-q4_k_m' from 'smollm2-1.7b-instruct-q4_k_m.gguf')
            model_shortname = os.path.splitext(os.path.basename(self.model_path))[0]
            output_file = f"{model_shortname}_{cli_args['benchmark']}.jsonl"

            # Then run the evaluation
            if cli_args['benchmark'] == "human_eval":
                subprocess.run([
                    ".venv/bin/python", "-m",
                    "human_eval.evaluate_functional_correctness",
                    output_file
                ], check=True)
            elif cli_args['benchmark'] == "lcb":
                from evaluate_lcb import evaluate_functional_correctness
                evaluate_functional_correctness(
                    output_file,
                    cli_args['problems_file'],
                    k=[1],  # We only need pass@1
                    debug=False
                )

            duration = time.time() - start_time

        # Process and store results
        run_results = self._process_results(cli_args['benchmark'], output_file, duration, cli_args)
        all_results = self._load_results()
        all_results["runs"].append(run_results)
        self._save_results(all_results)

        # Update README if requested
        if cli_args['update_readme']:
            self.readme_updater.update_table(run_results)

        return run_results

def main():
    parser = argparse.ArgumentParser(description="Run LLM benchmarks")
    parser.add_argument("--model", required=True, help="Model file path")
    parser.add_argument("--gpu-layers", type=int, required=True, help="Number of GPU layers")
    parser.add_argument("--server-type", choices=["llama", "sglang"], default="llama",
                      help="Server type to use (llama.cpp or sglang)")
    parser.add_argument("--benchmark", required=True, help="Benchmark type (e.g., human_eval, lcb)")
    parser.add_argument("--script", required=True, help="Path to benchmark script")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling")
    parser.add_argument("--preamble", help="Optional preamble text")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens per completion")
    parser.add_argument("--no-readme", action="store_true", help="Don't update README.md")
    parser.add_argument("--start-problem", type=int, default=1, help="Problem index to start from (1-based)")
    parser.add_argument("--end-problem", type=int, default=None, help="Problem index to end at (1-based)")
    parser.add_argument("--stream", action="store_true", help="Use streaming output for sglang server")

    # LCB-specific arguments
    parser.add_argument("--problems-file", help="Problems file for LCB benchmark (e.g., test5.jsonl)")
    parser.add_argument("--start-date", help="Start date for LCB problems (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date for LCB problems (YYYY-MM-DD)")

    args = parser.parse_args()

    runner = BenchmarkRunner(args.model, args.gpu_layers, server_type=args.server_type)
    cli_args = vars(args)
    cli_args['update_readme'] = not args.no_readme  # Convert no_readme to update_readme
    try:
        results = runner.run_benchmark(cli_args)
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"Error running benchmark: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
