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

from parsers import PARSERS

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

        # Find the table
        table_start = None
        for i, line in enumerate(lines):
            if "|" in line and "Human Eval" in line:
                table_start = i
                break

        if table_start is None:
            raise ValueError("Could not find results table in README.md")

        # Create new row
        model_name = results["model"]
        score = results["results"]["score"]
        time_taken = results["results"]["time_taken"]
        new_row = f"| {model_name} | {results['config']['gpu_layers']} layers | {score:.1f}% | {time_taken:.2f}s |\n"

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

    def _save_results(self, results: Dict[str, Any]):
        """Save results to file."""
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

    def _process_results(self, 
                        benchmark: str,
                        output_file: str,
                        duration: float) -> Dict[str, Any]:
        """Process benchmark results and format for storage."""
        parser = PARSERS.get(benchmark)
        if not parser:
            raise ValueError(f"No parser available for benchmark: {benchmark}")

        score, details = parser.parse_results(output_file)

        return {
            "timestamp": datetime.now().isoformat(),
            "benchmark": benchmark,
            "model": Path(self.model_path).stem,
            "config": {
                "path": self.model_path,
                "gpu_layers": self.gpu_layers
            },
            "results": {
                "score": score,
                "time_taken": duration,
                **details
            }
        }

    def run_benchmark(self, 
                     benchmark: str,
                     script_path: str,
                     temperature: float = 0.0,
                     preamble: Optional[str] = None,
                     max_tokens: int = 512,
                     update_readme: bool = True) -> Dict[str, Any]:
        """Run a benchmark and store its results."""
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Benchmark script not found: {script_path}")

        # Run the benchmark
        with self.llama_server() as port:
            start_time = time.time()
            # First run the benchmark script
            cmd = [".venv/bin/python", script_path,
                   "--model", self.model_path,
                   "--temperature", str(temperature),
                   "--max-tokens", str(max_tokens)]
            if preamble:
                cmd.extend(["--preamble", preamble])
            subprocess.run(cmd, check=True)

            # Get model shortname (e.g. 'smollm2-1.7b-instruct-q4_k_m' from 'smollm2-1.7b-instruct-q4_k_m.gguf')
            model_shortname = os.path.splitext(os.path.basename(self.model_path))[0]
            output_file = f"{model_shortname}.jsonl"

            # Then run the evaluation
            if benchmark == "human_eval":
                subprocess.run([
                    ".venv/bin/python", "-m",
                    "human_eval.evaluate_functional_correctness",
                    output_file
                ], check=True)

            duration = time.time() - start_time

        # Process and store results
        run_results = self._process_results(benchmark, output_file, duration)
        all_results = self._load_results()
        all_results["runs"].append(run_results)
        self._save_results(all_results)

        # Update README if requested
        if update_readme:
            self.readme_updater.update_table(run_results)

        return run_results

def main():
    parser = argparse.ArgumentParser(description="Run LLM benchmarks")
    parser.add_argument("--model", required=True, help="Model file path")
    parser.add_argument("--gpu-layers", type=int, required=True, help="Number of GPU layers")
    parser.add_argument("--benchmark", required=True, help="Benchmark type (e.g., human_eval, lcb)")
    parser.add_argument("--script", required=True, help="Path to benchmark script")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling")
    parser.add_argument("--preamble", help="Optional preamble text")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens per completion")
    parser.add_argument("--no-readme", action="store_true", help="Don't update README.md")

    args = parser.parse_args()

    runner = BenchmarkRunner(args.model, args.gpu_layers)
    try:
        results = runner.run_benchmark(
            args.benchmark,
            args.script,
            temperature=args.temperature,
            preamble=args.preamble,
            max_tokens=args.max_tokens,
            update_readme=not args.no_readme
        )
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"Error running benchmark: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
