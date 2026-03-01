#!/usr/bin/env python
"""Benchmark runner with job queue support.

Usage:
    # Queue mode: process all YAML files in jobs/queued/
    bench_runner --queue [--dry-run]

    # Single-run mode (backward-compatible)
    bench_runner --model model.gguf --benchmark human_eval [--context-size 32768]

    # Disable post-job steps
    bench_runner --queue --no-readme --no-plot --no-commit
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from human3090.bench_constants import DEFAULT_VALUES
from human3090.job import Job, job_from_cli, load_queue_dir, move_job_file, next_queued_file
from human3090.parse_results import parse_results
from human3090.server import ServerManager

BENCHMARK_MODULES = {
    "human_eval": "human3090.run_eval",
    "lcb": "human3090.run_lcb",
}

RESULTS_FILE = "benchmark_results.json"
TRACKED_FILES = ["README.md", "benchmark_results.json", "human_eval_scatter.png"]


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

        benchmark = results["benchmark"]
        table_start = None
        for i, line in enumerate(lines):
            if "|" in line:
                if benchmark == "human_eval" and "Human Eval" in line:
                    table_start = i
                    break
                elif benchmark == "lcb" and "Version" in line and "Score" in line:
                    table_start = i
                    break

        if table_start is None:
            raise ValueError(f"Could not find {benchmark} results table in README.md")

        table_end = table_start
        for i in range(table_start + 1, len(lines)):
            if "|" not in lines[i]:
                table_end = i
                break

        model_name = results["model"]
        score = results["results"]["score"]
        time_taken = results["results"]["time_taken"]
        model_short_name = model_name.split('-')[0]

        command = results["command"]
        command_parts = command.split()

        non_default_params = []
        i = 0
        while i < len(command_parts):
            if command_parts[i].startswith("--") and i + 1 < len(command_parts):
                param = command_parts[i]
                value = command_parts[i + 1]
                if param in DEFAULT_VALUES and value != DEFAULT_VALUES[param]:
                    param_name = param[2:]
                    non_default_params.append(f"`{param_name}={value}`")
                i += 2
            else:
                i += 1

        configuration = model_name
        if non_default_params:
            configuration += ", " + ", ".join(non_default_params)

        if benchmark == "lcb":
            problems_file = results.get("problems_file") or "test5.jsonl"
            version = problems_file.replace("test", "LCBv").replace(".jsonl", "")
            new_row = f"| {model_short_name:<19} | {configuration:<62} | {version:<7} | {score:>9.1f}%  | {time_taken:>9.2f}s |\n"
        else:
            new_row = f"| {model_short_name:<19} | {configuration:<62} | {score:>9.1f}%  | {time_taken:>9.2f}s |\n"
        lines.insert(table_end, new_row)

        with open(self.readme_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)


# --- Result storage ---

def _load_results() -> Dict[str, Any]:
    if not os.path.exists(RESULTS_FILE):
        return {"runs": []}
    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def _save_results(results: Dict[str, Any]):
    """Save results with failed_tasks compacted to single lines."""
    compact_list = []
    for i, run in enumerate(results['runs']):
        if 'results' in run and 'failed_tasks' in run['results']:
            failed_tasks = run['results']['failed_tasks']
            compact = json.dumps(failed_tasks, separators=(',', ':'))
            compact_list.append(compact)
            run['results']['failed_tasks'] = f"__FAILED_TASKS_PLACEHOLDER_{i}__"

    json_str = json.dumps(results, indent=2)
    for i, compact in enumerate(compact_list):
        placeholder = f'"__FAILED_TASKS_PLACEHOLDER_{i}__"'
        json_str = json_str.replace(placeholder, compact)

    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        f.write(json_str)


# --- Command building ---

def _build_benchmark_cmd(job: Job) -> list[str]:
    """Build the subprocess command for a benchmark run."""
    module = BENCHMARK_MODULES[job.benchmark]
    cmd = [sys.executable, "-m", module]

    params = {
        "--model": job.model,
        "--temperature": job.temperature,
        "--max-tokens": job.max_tokens,
        "--top-p": job.top_p,
        "--min-p": job.min_p,
        "--start-problem": job.start_problem,
    }

    optional = {
        "--preamble": job.preamble,
        "--end-problem": job.end_problem,
    }

    if job.benchmark == "lcb":
        params["--problems-file"] = job.problems_file
        optional["--start-date"] = job.start_date
        optional["--end-date"] = job.end_date

    for flag, value in params.items():
        cmd.extend([flag, str(value)])
    for flag, value in optional.items():
        if value is not None:
            cmd.extend([flag, str(value)])

    if job.save_raw and job.benchmark == "lcb":
        cmd.append("--save-raw")

    return cmd


def _reconstruct_command(job: Job) -> str:
    """Reconstruct a CLI command string for reproducibility."""
    parts = [sys.executable, "-m", "human3090.bench_runner",
             "--model", job.model, "--benchmark", job.benchmark]
    for param, attr, default in [
        ("--temperature", "temperature", float(DEFAULT_VALUES["--temperature"])),
        ("--max-tokens", "max_tokens", int(DEFAULT_VALUES["--max-tokens"])),
        ("--top-p", "top_p", float(DEFAULT_VALUES["--top-p"])),
        ("--min-p", "min_p", float(DEFAULT_VALUES["--min-p"])),
        ("--context-size", "context_size", int(DEFAULT_VALUES["--context-size"])),
        ("--start-problem", "start_problem", int(DEFAULT_VALUES["--start-problem"])),
    ]:
        value = getattr(job, attr)
        if value != default:
            parts.extend([param, str(value)])
    if job.end_problem is not None:
        parts.extend(["--end-problem", str(job.end_problem)])
    if job.preamble:
        parts.extend(["--preamble", job.preamble])
    if job.save_raw:
        parts.append("--save-raw")
    if job.benchmark == "lcb" and job.problems_file != "test5.jsonl":
        parts.extend(["--problems-file", job.problems_file])
    return " ".join(parts)


# --- Running benchmarks ---

def _run_benchmark(job: Job):
    """Run the benchmark generation script as a subprocess."""
    if os.path.exists(job.output_file):
        os.remove(job.output_file)

    cmd = _build_benchmark_cmd(job)
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _run_evaluation(job: Job) -> Optional[Dict[str, Any]]:
    """Run the evaluation for completed benchmark output. Returns metadata if available."""
    if job.benchmark == "human_eval":
        from human3090.human_eval.evaluation import evaluate_functional_correctness as eval_he
        eval_he(job.output_file, k=[1])
        return None
    elif job.benchmark == "lcb":
        from human3090.evaluate_lcb import evaluate_functional_correctness
        results = evaluate_functional_correctness(
            job.output_file,
            job.problems_file,
            k=[1],
            debug=False,
        )
        return results.get("metadata") if isinstance(results, dict) else None
    return None


def _store_results(job: Job, duration: float) -> Dict[str, Any]:
    """Process, store, and return results for a completed benchmark run."""
    score, details = parse_results(job.output_file)

    run_result = {
        "timestamp": datetime.now().isoformat(),
        "benchmark": job.benchmark,
        "model": job.model_shortname,
        "command": _reconstruct_command(job),
        "problems_file": job.problems_file if job.benchmark == "lcb" else None,
        "results": {
            "score": score,
            "time_taken": duration,
            **details,
        },
    }

    all_results = _load_results()
    all_results["runs"].append(run_result)
    _save_results(all_results)

    return run_result


# --- Post-job: README, plot, git commit ---

def _update_readme(readme_updater: ReadmeUpdater, run_result: Dict[str, Any]):
    try:
        readme_updater.update_table(run_result)
        print("  -> README.md updated")
    except Exception as exc:
        print(f"  Warning: Failed to update README: {exc}")


def _update_plot(run_result: Dict[str, Any]):
    """Regenerate the efficient frontier plot (HumanEval only)."""
    if run_result["benchmark"] != "human_eval":
        return
    try:
        from human3090.plot_human_eval import update_plot
        update_plot()
        print("  -> human_eval_scatter.png updated")
    except Exception as exc:
        print(f"  Warning: Failed to update plot: {exc}")


def _git_commit(run_result: Dict[str, Any]):
    """Commit tracked result files to git and push."""
    model = run_result["model"]
    benchmark = run_result["benchmark"]
    score = run_result["results"]["score"]

    # Add LCB version to commit message
    if benchmark == "lcb" and run_result.get("problems_file"):
        version = run_result["problems_file"].replace("test", "LCBv").replace(".jsonl", "")
        benchmark_id = f"{benchmark} {version}"
    else:
        benchmark_id = benchmark

    msg = f"add {model} {benchmark_id} results ({score:.1f}%)"

    try:
        # Stage only the tracked result files that exist
        files_to_add = [f for f in TRACKED_FILES if os.path.exists(f)]
        if not files_to_add:
            return
        subprocess.run(["git", "add"] + files_to_add, check=True, capture_output=True)

        # Check if there's anything staged
        result = subprocess.run(["git", "diff", "--cached", "--quiet"], capture_output=True)
        if result.returncode == 0:
            print("  -> nothing to commit")
            return

        subprocess.run(["git", "commit", "-m", msg], check=True, capture_output=True)
        print(f"  -> committed: {msg}")

        # Push to remote
        subprocess.run(["git", "push"], check=True, capture_output=True)
        print(f"  -> pushed to remote")
    except Exception as exc:
        print(f"  Warning: Failed to commit/push: {exc}")


def _post_job(run_result: Dict[str, Any], readme_updater: ReadmeUpdater,
              do_readme: bool, do_plot: bool, do_commit: bool):
    """Run post-job steps: update README, regenerate plot, git commit."""
    if do_readme:
        _update_readme(readme_updater, run_result)
    if do_plot:
        _update_plot(run_result)
    if do_commit:
        _git_commit(run_result)


# --- Job execution ---

def run_single_job(job: Job, server: ServerManager, readme_updater: ReadmeUpdater,
                   do_readme: bool = True, do_plot: bool = True, do_commit: bool = True) -> Dict[str, Any]:
    """Run a single benchmark job end-to-end."""
    server.ensure_server(job)

    start_time = time.time()
    _run_benchmark(job)
    metadata = _run_evaluation(job)

    # Use timing from metadata if available (more accurate), otherwise measure total time
    if metadata and "total_time" in metadata:
        duration = metadata["total_time"]
    else:
        duration = time.time() - start_time

    run_result = _store_results(job, duration)
    _post_job(run_result, readme_updater, do_readme, do_plot, do_commit)

    return run_result


# --- Queue processing ---

def recover_orphaned_runs(readme_updater: ReadmeUpdater, do_readme: bool = True,
                          do_plot: bool = True, do_commit: bool = True):
    """Find and evaluate completed benchmark runs that weren't evaluated (orphaned by killed processes)."""
    import glob

    # Find all result files
    result_files = glob.glob("*_lcb.jsonl") + glob.glob("*_human_eval.jsonl")
    orphaned = []

    for result_file in result_files:
        # Skip raw files and already-evaluated results
        if "_raw.jsonl" in result_file or "_results.jsonl" in result_file:
            continue

        # Check if evaluation results exist
        results_file = f"{result_file}_results.jsonl"
        if not os.path.exists(results_file) or os.path.getsize(results_file) == 0:
            orphaned.append(result_file)

    if not orphaned:
        return

    print(f"\n{'='*60}")
    print(f"Found {len(orphaned)} orphaned run(s) - evaluating...")
    print(f"{'='*60}\n")

    for result_file in orphaned:
        try:
            print(f"Evaluating: {result_file}")

            # Determine benchmark type and problems file
            if "_lcb.jsonl" in result_file:
                benchmark = "lcb"
                # Try to detect version from metadata in the file
                problems_file = "test5.jsonl"  # default
                has_metadata = False
                try:
                    with open(result_file, 'r') as f:
                        for line in f:
                            data = json.loads(line)
                            if data.get('_metadata'):
                                has_metadata = True
                                if 'problems_file' in data:
                                    problems_file = data['problems_file']
                                break
                except:
                    pass

                # Skip incomplete runs (no metadata footer)
                if not has_metadata:
                    print(f"  ⚠ Skipping incomplete run (no metadata)")
                    continue

                from human3090.evaluate_lcb import evaluate_functional_correctness
                eval_results = evaluate_functional_correctness(
                    result_file,
                    problems_file,
                    k=[1],
                    debug=False,
                )
                metadata = eval_results.get("metadata") if isinstance(eval_results, dict) else None
            else:
                benchmark = "human_eval"
                from human3090.human_eval.evaluation import evaluate_functional_correctness as eval_he
                eval_he(result_file, k=[1])
                metadata = None
                problems_file = None

            # Parse and store results
            score, details = parse_results(result_file)

            # Use timing from metadata if available
            if metadata and "total_time" in metadata:
                duration = metadata["total_time"]
            else:
                duration = 0  # Unknown

            # Extract model name from filename
            model_name = result_file.replace(f"_{benchmark}.jsonl", "").replace("_", "/", 1)

            # Build minimal command reconstruction
            command = f"bench_runner --model {model_name} --benchmark {benchmark}"
            if problems_file and benchmark == "lcb":
                command += f" --problems-file {problems_file}"

            # Store results
            run_result = {
                "timestamp": datetime.now().isoformat(),
                "benchmark": benchmark,
                "model": model_name,
                "command": command,
            }
            if problems_file:
                run_result["problems_file"] = problems_file

            run_result["results"] = {
                "score": score,
                "time_taken": duration,
                **details
            }

            # Update benchmark_results.json
            if os.path.exists(RESULTS_FILE):
                with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = {"runs": []}

            data["runs"].append(run_result)

            with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

            print(f"  ✓ Recovered: {score:.1f}% (duration: {duration:.0f}s)")

            # Update README and commit
            _post_job(run_result, readme_updater, do_readme, do_plot, do_commit)

        except Exception as exc:
            print(f"  ✗ Failed to recover {result_file}: {exc}", file=sys.stderr)

    print()


def run_queue(jobs_dir: str, server: ServerManager, readme_updater: ReadmeUpdater,
              dry_run: bool = False, watch: bool = False,
              do_readme: bool = True, do_plot: bool = True, do_commit: bool = True):
    """Process YAML job files from jobs_dir/queued/, polling for new files between jobs."""
    if dry_run:
        file_jobs = load_queue_dir(jobs_dir)
        if not file_jobs:
            print("No job files found in queued/")
            return
        total_jobs = sum(len(jobs) for _, jobs in file_jobs)
        print(f"\nQueue: {len(file_jobs)} file(s), {total_jobs} job(s)\n")
        _print_dry_run(file_jobs)
        return

    # Recover any orphaned runs on startup
    recover_orphaned_runs(readme_updater, do_readme, do_plot, do_commit)

    done_dir = str(Path(jobs_dir) / "done")
    failed_dir = str(Path(jobs_dir) / "failed")

    succeeded, failed = [], []
    job_num = 0

    try:
        while True:
            entry = next_queued_file(jobs_dir)
            if entry is None:
                if not watch:
                    break
                # In watch mode, idle until a new file appears
                time.sleep(5)
                continue

            yaml_path, jobs = entry

            print(f"\n{'='*60}")
            print(f"File: {yaml_path.name}")
            print(f"{'='*60}")

            file_ok = True
            for job in jobs:
                job_num += 1
                print(f"\n[{job_num}] {job.job_id()}")

                try:
                    result = run_single_job(job, server, readme_updater,
                                            do_readme=do_readme, do_plot=do_plot, do_commit=do_commit)
                    score = result["results"]["score"]
                    duration = result["results"]["time_taken"]
                    print(f"  DONE: {score:.1f}% in {duration:.0f}s")
                    succeeded.append(job.job_id())
                except Exception as exc:
                    print(f"  FAILED: {exc}", file=sys.stderr)
                    failed.append((job.job_id(), str(exc)))
                    file_ok = False

            if file_ok:
                move_job_file(yaml_path, done_dir)
                print(f"  -> moved to done/")
            else:
                move_job_file(yaml_path, failed_dir)
                print(f"  -> moved to failed/")

            # In watch mode, shut down server when queue is drained
            if watch and next_queued_file(jobs_dir) is None:
                server.shutdown()
                print("Queue empty, server stopped. Watching for new jobs...")

    except KeyboardInterrupt:
        print("\n\nInterrupted! Shutting down server...")
    finally:
        server.shutdown()

    print(f"\n{'='*60}")
    print(f"Queue complete: {len(succeeded)} succeeded, {len(failed)} failed")
    for job_id, error in failed:
        print(f"  FAILED: {job_id}: {error}")


def _print_dry_run(file_jobs: list):
    """Print the job queue without running anything."""
    prev_key = None
    job_num = 0

    for yaml_path, jobs in file_jobs:
        print(f"File: {yaml_path.name}")
        for job in jobs:
            job_num += 1
            key = job.server_key()
            restart = "[RESTART]" if key != prev_key else "[REUSE]"
            prev_key = key

            print(f"  {job_num}. {job.job_id()}")
            print(f"     ctx={job.context_size}  max_tokens={job.max_tokens}  "
                  f"temp={job.temperature}  {restart}")
        print()


# --- CLI ---

def main():
    parser = argparse.ArgumentParser(description="Run LLM benchmarks")

    # Queue mode
    parser.add_argument("--queue", nargs="?", const="jobs", default=None,
                        help="Process job files from directory (default: ./jobs/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print job queue without running")
    parser.add_argument("--watch", action="store_true",
                        help="Keep running and poll for new job files")

    # Single-run mode
    parser.add_argument("--model", help="Model file path")
    parser.add_argument("--benchmark", help="Benchmark type (human_eval, lcb)")

    # Shared parameters
    parser.add_argument("--gpu-layers", type=int, default=int(DEFAULT_VALUES["--gpu-layers"]))
    parser.add_argument("--context-size", type=int, default=int(DEFAULT_VALUES["--context-size"]))
    parser.add_argument("--temperature", type=float, default=float(DEFAULT_VALUES["--temperature"]))
    parser.add_argument("--top-p", type=float, default=float(DEFAULT_VALUES["--top-p"]))
    parser.add_argument("--min-p", type=float, default=float(DEFAULT_VALUES["--min-p"]))
    parser.add_argument("--max-tokens", type=int, default=int(DEFAULT_VALUES["--max-tokens"]))
    parser.add_argument("--preamble", help="Optional preamble text")
    parser.add_argument("--start-problem", type=int, default=int(DEFAULT_VALUES["--start-problem"]))
    parser.add_argument("--end-problem", type=int)
    parser.add_argument("--save-raw", action="store_true", help="Save raw model responses")

    # Post-job steps (all enabled by default)
    parser.add_argument("--no-readme", action="store_true", help="Don't update README.md")
    parser.add_argument("--no-plot", action="store_true", help="Don't regenerate scatter plot")
    parser.add_argument("--no-commit", action="store_true", help="Don't git commit results")

    # LCB-specific
    parser.add_argument("--problems-file", help="Problems file for LCB (default: test5.jsonl)")
    parser.add_argument("--start-date", help="Start date for LCB problems (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date for LCB problems (YYYY-MM-DD)")

    args = parser.parse_args()

    server = ServerManager()
    readme_updater = ReadmeUpdater()
    do_readme = not args.no_readme
    do_plot = not args.no_plot
    do_commit = not args.no_commit

    try:
        if args.queue is not None:
            run_queue(args.queue, server, readme_updater, dry_run=args.dry_run,
                      watch=args.watch, do_readme=do_readme, do_plot=do_plot, do_commit=do_commit)
        elif args.model:
            if not args.benchmark:
                parser.error("--benchmark is required when using --model")
            if args.benchmark not in BENCHMARK_MODULES:
                parser.error(f"Unknown benchmark: {args.benchmark}. Choose from: {list(BENCHMARK_MODULES)}")

            job = job_from_cli(args)
            result = run_single_job(job, server, readme_updater,
                                    do_readme=do_readme, do_plot=do_plot, do_commit=do_commit)
            score = result["results"]["score"]
            duration = result["results"]["time_taken"]
            print(f"\nResult: {score:.1f}% in {duration:.0f}s")
            server.shutdown()
        else:
            parser.print_help()
            sys.exit(1)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        server.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()
