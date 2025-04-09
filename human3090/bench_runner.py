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
from typing import Any, Dict
from urllib.error import URLError

from . import parse_results
from .bench_constants import DEFAULT_VALUES


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

        # Find the end of the table
        table_end = table_start
        for i in range(table_start + 1, len(lines)):
            if "|" not in lines[i]:
                table_end = i
                break

        # Create new row with non-default parameters
        model_name = results["model"]
        score = results["results"]["score"]
        time_taken = results["results"]["time_taken"]
        model_short_name = model_name.split('-')[0]

        # Extract command line arguments for non-default parameters
        command = results["command"]
        command_parts = command.split()

        # Extract non-default parameters
        non_default_params = []
        i = 0
        while i < len(command_parts):
            if command_parts[i].startswith("--") and i + 1 < len(command_parts):
                param = command_parts[i]
                value = command_parts[i + 1]

                # Check if this is a non-default value and a parameter we want to include
                if param in DEFAULT_VALUES and value != DEFAULT_VALUES[param]:
                    param_name = param[2:]  # Remove -- prefix
                    non_default_params.append(f"`{param_name}={value}`")
                i += 2
            else:
                i += 1

        # Create configuration string
        configuration = model_name
        if non_default_params:
            configuration += ", " + ", ".join(non_default_params)

        # Format the new row
        new_row = f"| {model_short_name:<19} | {configuration:<62} | {score:>9.1f}%  | {time_taken:>9.2f}s |\n"

        # Insert the row at the end of the table
        lines.insert(table_end, new_row)

        # Write back to file
        with open(self.readme_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

class BenchmarkRunner:
    """Manages LLM benchmarks, handling server lifecycle and results storage."""

    def __init__(
        self,
        model_path: str,
        gpu_layers: int,
        results_file: str = "benchmark_results.json",
        server_path: str = "/code/llama.cpp/build/bin/llama-server",
        readme_path: str = "README.md"
    ):
        self.model_path = model_path
        self.gpu_layers = gpu_layers
        self.results_file = results_file
        self.server_path = server_path
        self.readme_updater = ReadmeUpdater(readme_path)

        # Create results file if it doesn't exist
        if not os.path.exists(results_file):
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump({"runs": []}, f, indent=2)

    @contextmanager
    def llama_server(self, port: int = 8083):
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
            while time.time() - start_time < 600:  # 600 second timeout
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

    def _save_results(self, results: Dict[str, Any]):
        """Save results to file with failed_tasks as a JSON list on a single line."""
        # Step 1: Process each run to replace 'failed_tasks' with placeholders
        compact_list = []
        for i, run in enumerate(results['runs']):
            if 'results' in run and 'failed_tasks' in run['results']:
                failed_tasks = run['results']['failed_tasks']
                # Serialize failed_tasks compactly
                compact = json.dumps(failed_tasks, separators=(',', ':'))
                compact_list.append(compact)
                # Set a unique placeholder
                run['results']['failed_tasks'] = f"__FAILED_TASKS_PLACEHOLDER_{i}__"

        # Step 2: Serialize the entire results with indent=2
        json_str = json.dumps(results, indent=2)

        # Step 3: Replace each placeholder with the corresponding compact version
        for i, compact in enumerate(compact_list):
            placeholder = f'"__FAILED_TASKS_PLACEHOLDER_{i}__"'
            json_str = json_str.replace(placeholder, compact)

        # Step 4: Write to file
        with open(self.results_file, 'w', encoding='utf-8') as f:
            f.write(json_str)

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
        # Get the directory where this script (bench_runner.py) is located
        package_dir = Path(__file__).parent.resolve()
        script_filename = cli_args['script']
        # Construct the full path to the target script within the package
        script_path = package_dir / script_filename

        if not script_path.exists():
            # Look also in CWD for backward compatibility or other use cases? Optional.
            script_path_cwd = Path(script_filename).resolve()
            if not script_path_cwd.exists():
                raise FileNotFoundError(f"Benchmark script not found in package ({package_dir}) or CWD: {script_filename}")
            else:
                script_path = script_path_cwd # Found in CWD

        # Run the benchmark using the resolved script path
        with self.llama_server() as port:
            start_time = time.time()
            # First run the benchmark script
            module_name = f"human3090.{script_path.stem}"
            cmd = [sys.executable, "-m", module_name, # Use sys.executable and the full path
                   "--model", self.model_path,
                   "--temperature", str(cli_args['temperature']),
                   "--max-tokens", str(cli_args['max_tokens'])]
            if cli_args.get('preamble'):
                cmd.extend(["--preamble", cli_args['preamble']])
            if cli_args.get('start_problem', 1) > 1:
                cmd.extend(["--start-problem", str(cli_args['start_problem'])])
            if cli_args.get('end_problem'):
                cmd.extend(["--end-problem", str(cli_args['end_problem'])])
            if cli_args.get('top_p'):
                cmd.extend(["--top-p", str(cli_args['top_p'])])
            if cli_args.get('min_p'):
                cmd.extend(["--min-p", str(cli_args['min_p'])])

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
        if not cli_args['no_readme']:
            self.readme_updater.update_table(run_results)

        return run_results

def main():
    parser = argparse.ArgumentParser(description="Run LLM benchmarks")
    parser.add_argument("--model", required=True, help="Model file path")
    parser.add_argument("--gpu-layers", type=int, default=int(DEFAULT_VALUES["--gpu-layers"]), help="Number of GPU layers")
    parser.add_argument("--benchmark", required=True, help="Benchmark type (e.g., human_eval, lcb)")
    parser.add_argument("--script", required=True, help="Path to benchmark script")
    parser.add_argument("--temperature", type=float, default=float(DEFAULT_VALUES["--temperature"]),help="Temperature for sampling")
    parser.add_argument("--top-p", type=float, default=float(DEFAULT_VALUES["--top-p"]),help="Top-p for sampling")
    parser.add_argument("--min-p", type=float, default=float(DEFAULT_VALUES["--min-p"]),help="Minimum p for sampling")
    parser.add_argument("--preamble", help="Optional preamble text")
    parser.add_argument("--max-tokens", type=int, default=int(DEFAULT_VALUES["--max-tokens"]),help="Maximum tokens per completion")
    parser.add_argument("--no-readme", action="store_true", help="Don't update README.md")
    parser.add_argument("--start-problem", type=int, default=int(DEFAULT_VALUES["--start-problem"]),help="Problem index to start from (1-based)")
    parser.add_argument("--end-problem", type=int, help="Problem index to end at (1-based)")

    # LCB-specific arguments
    parser.add_argument("--problems-file", help="Problems file for LCB benchmark (e.g., test5.jsonl)")
    parser.add_argument("--start-date", help="Start date for LCB problems (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date for LCB problems (YYYY-MM-DD)")

    args = parser.parse_args()

    runner = BenchmarkRunner(args.model, args.gpu_layers)
    try:
        # Pass args for command reconstruction and execution
        runner.run_benchmark(vars(args))
    except Exception as exc:
        print(f"Error running benchmark: {exc}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
