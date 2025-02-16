#!/usr/bin/env python
"""Generic benchmark runner for LLM evaluation tasks."""

import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from parsers import PARSERS

class BenchmarkRunner:
    """Manages LLM benchmarks, handling server lifecycle and results storage."""

    def __init__(
        self,
        model_path: str,
        gpu_layers: int,
        results_file: str = "benchmark_results.json",
        server_path: str = "./build/bin/llama-server"
    ):
        self.model_path = model_path
        self.gpu_layers = gpu_layers
        self.results_file = results_file
        self.server_path = server_path

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
            # Wait for server to start
            time.sleep(5)
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
                     output_file: str,
                     script_args: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Run a benchmark and store its results."""
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Benchmark script not found: {script_path}")

        # Run the benchmark
        with self.llama_server() as port:
            start_time = time.time()
            # First run the benchmark script
            cmd = [".venv/bin/python", script_path]
            if script_args:
                for k, v in script_args.items():
                    cmd.extend([k, v])
            subprocess.run(cmd, check=True)

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

        return run_results

def main():
    parser = argparse.ArgumentParser(description="Run LLM benchmarks")
    parser.add_argument("--model", required=True, help="Path to model file")
    parser.add_argument("--gpu-layers", type=int, required=True, help="Number of GPU layers")
    parser.add_argument("--benchmark", required=True, help="Benchmark type (e.g., human_eval, lcb)")
    parser.add_argument("--script", required=True, help="Path to benchmark script")
    parser.add_argument("--output", required=True, help="Output file for benchmark results")

    args = parser.parse_args()

    runner = BenchmarkRunner(args.model, args.gpu_layers)
    try:
        results = runner.run_benchmark(
            args.benchmark,
            args.script,
            args.output
        )
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"Error running benchmark: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
