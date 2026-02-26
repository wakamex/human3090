#!/usr/bin/env python3
"""Analyze benchmark results: per-problem difficulty, model comparison."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_results(path: str = "benchmark_results.json") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("runs", [])


def analyze(runs: list[dict], benchmark: str = "human_eval"):
    """Return (problem_stats, model_stats) for the given benchmark."""
    problem_perf = defaultdict(lambda: {"passed": 0, "total": 0})
    model_stats = {}

    for run in runs:
        if run.get("benchmark") != benchmark:
            continue

        model = run["model"]
        results = run.get("results", {})
        failed = set(results.get("failed_tasks", []))
        total = results.get("total_problems", 164)
        passed = total - len(failed)
        model_stats[model] = {"passed": passed, "total": total, "score": passed / total * 100 if total else 0}

        for i in range(total):
            task_id = f"HumanEval/{i}"
            problem_perf[task_id]["total"] += 1
            if task_id not in failed:
                problem_perf[task_id]["passed"] += 1

    problem_stats = []
    for task_id, stats in problem_perf.items():
        rate = stats["passed"] / stats["total"] * 100 if stats["total"] else 0
        problem_stats.append({"task_id": task_id, "pass_rate": rate, **stats})

    problem_stats.sort(key=lambda x: (x["pass_rate"], x["task_id"]))
    return problem_stats, model_stats


def print_report(problem_stats, model_stats, top_n=20):
    print(f"\n{'='*60}")
    print(f"  {len(model_stats)} models, {len(problem_stats)} problems")
    print(f"{'='*60}")

    print(f"\n  Hardest problems (bottom {top_n}):")
    for s in problem_stats[:top_n]:
        print(f"    {s['task_id']:<20} {s['pass_rate']:5.1f}%  ({s['passed']}/{s['total']})")

    print(f"\n  Easiest problems (top {top_n}):")
    for s in problem_stats[-top_n:]:
        print(f"    {s['task_id']:<20} {s['pass_rate']:5.1f}%  ({s['passed']}/{s['total']})")

    print(f"\n  Models (by score):")
    for model, s in sorted(model_stats.items(), key=lambda x: -x[1]["score"]):
        print(f"    {model:<40} {s['score']:5.1f}%  ({s['passed']}/{s['total']})")


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("-f", "--file", default="benchmark_results.json", help="Results JSON file")
    parser.add_argument("-b", "--benchmark", default="human_eval", help="Benchmark to analyze")
    parser.add_argument("-n", "--top", type=int, default=20, help="Number of problems to show")
    args = parser.parse_args()

    runs = load_results(args.file)
    problem_stats, model_stats = analyze(runs, args.benchmark)

    if not model_stats:
        print(f"No {args.benchmark} results found")
        return

    print_report(problem_stats, model_stats, args.top)


if __name__ == "__main__":
    main()
