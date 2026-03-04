#!/usr/bin/env python3
"""Generate efficient frontier plots for LCBv5 and LCBv6 scores vs time."""
import json
import matplotlib.pyplot as plt
import numpy as np


def load_lcb_data(results_path="benchmark_results.json"):
    """Load and clean LCB benchmark data, returning separate v5/v6 lists."""
    with open(results_path) as f:
        data = json.load(f)

    v5, v6 = [], []
    for run in data["runs"]:
        if run["benchmark"] != "lcb":
            continue
        score = run["results"]["score"]
        time = run["results"].get("time_taken", 0)
        if time <= 0:
            continue

        pf = run.get("problems_file", "")
        cmd = run.get("command", "")
        model = run["model"]

        # Determine version
        if "test5" in pf or "test5" in cmd:
            ver = "v5"
        elif "test6" in pf or "test6" in cmd:
            ver = "v6"
        else:
            # Pre-versioning runs: skip clearly broken ones (<10%)
            if score < 10:
                continue
            ver = "v5"

        entry = {"model": model, "score": score, "time": time}
        (v5 if ver == "v5" else v6).append(entry)

    # Deduplicate: keep best score per model per version
    def dedup(entries):
        best = {}
        for e in entries:
            m = e["model"]
            if m not in best or e["score"] > best[m]["score"]:
                best[m] = e
        return list(best.values())

    return dedup(v5), dedup(v6)


def shorten_model_name(name):
    """Shorten model names for plot labels."""
    replacements = [
        ("Instruct-2503-", ""),
        ("Instruct-", ""),
        ("-instruct", ""),
        ("mistralai/Mistral-Small-3.1-24B", "Mistral-Small-3.1-24B"),
        ("agentica-org_", ""),
        ("OpenCodeReasoning-Nemotron-1.1-", "OCR-Nemotron-"),
        (".i1-Q4_K_M", " Q4_K_M"),
        (".i1-Q4/K_M", " Q4_K_M"),
        ("DeepSeek-R1-Distill-Qwen-", "DS-R1-Qwen-"),
        ("DeepSeek-R1-Distill-Qwen-14B-Q6/K_L", "DS-R1-Qwen-14B Q6_K_L"),
        ("Devstral-Small-2505-", "Devstral-Small "),
    ]
    for old, new in replacements:
        name = name.replace(old, new)
    return name


def find_efficient_frontier(times, scores):
    """Find non-dominated models (higher score, lower time)."""
    n = len(times)
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and (
                (scores[j] > scores[i] and times[j] <= times[i])
                or (scores[j] >= scores[i] and times[j] < times[i])
            ):
                is_efficient[i] = False
                break
    return is_efficient


def plot_frontier(data, title, output_path, show=False):
    """Create an efficient frontier scatter plot."""
    if not data:
        print(f"No data for {title}")
        return

    times = [d["time"] for d in data]
    scores = [d["score"] for d in data]
    models = [shorten_model_name(d["model"]) for d in data]
    efficient = find_efficient_frontier(times, scores)

    fig, ax = plt.subplots(figsize=(13, 8))

    dominated = [i for i, e in enumerate(efficient) if not e]
    frontier = [i for i, e in enumerate(efficient) if e]

    ax.scatter(
        [times[i] for i in dominated],
        [scores[i] for i in dominated],
        alpha=0.7, s=120, color="#4a90d9", edgecolors="white", linewidth=0.5,
        label="Dominated",
    )
    ax.scatter(
        [times[i] for i in frontier],
        [scores[i] for i in frontier],
        alpha=0.9, s=140, color="#e74c3c", edgecolors="white", linewidth=0.5,
        label="Efficient Frontier",
    )

    # Frontier line
    if len(frontier) >= 2:
        sorted_f = sorted(frontier, key=lambda i: times[i])
        ax.plot(
            [times[i] for i in sorted_f],
            [scores[i] for i in sorted_f],
            "r--", alpha=0.4,
        )

    # Labels with collision avoidance
    for i, model in enumerate(models):
        ax.annotate(
            model,
            (times[i], scores[i]),
            xytext=(12, 0),
            textcoords="offset points",
            fontsize=8,
            ha="left",
            va="center",
        )

    # Log scale if range is wide
    non_zero = [t for t in times if t > 0]
    if non_zero and max(non_zero) / min(non_zero) > 20:
        ax.set_xscale("log")
        ax.set_xlabel("Total Benchmark Time (seconds, log scale)", fontsize=12)
    else:
        ax.set_xlabel("Total Benchmark Time (seconds)", fontsize=12)

    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved {output_path}")
    if show:
        plt.show()
    plt.close()


def update_plots(results_path="benchmark_results.json", show=False):
    """Regenerate both LCB frontier plots."""
    v5, v6 = load_lcb_data(results_path)
    plot_frontier(v5, "LiveCodeBench v5 — Efficient Frontier (RTX 3090)", "lcb_v5_frontier.png", show)
    plot_frontier(v6, "LiveCodeBench v6 — Efficient Frontier (RTX 3090)", "lcb_v6_frontier.png", show)


def main():
    v5, v6 = load_lcb_data()

    for label, entries in [("LCBv5", v5), ("LCBv6", v6)]:
        print(f"\n=== {label} ===")
        times = [d["time"] for d in entries]
        scores = [d["score"] for d in entries]
        models = [d["model"] for d in entries]
        efficient = find_efficient_frontier(times, scores)

        for i in sorted(range(len(entries)), key=lambda i: scores[i], reverse=True):
            tag = " *" if efficient[i] else ""
            print(f"  {shorten_model_name(models[i]):45s} {scores[i]:5.1f}%  {times[i]:8.0f}s{tag}")

    update_plots(show=True)


if __name__ == "__main__":
    main()
