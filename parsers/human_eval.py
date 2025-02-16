"""Parser for HumanEval benchmark results."""

import json
import os
from typing import Any, Dict, Tuple

from .base import BenchmarkParser

class HumanEvalParser(BenchmarkParser):
    """Parser for HumanEval benchmark results."""

    def parse_results(self, output_file: str) -> Tuple[float, Dict[str, Any]]:
        results_file = f"{output_file}_results.jsonl"
        if not os.path.exists(results_file):
            raise FileNotFoundError(f"Results file not found: {results_file}")

        # Parse results
        results_jsonl = []
        with open(results_file, "r", encoding="utf-8") as f:
            results_jsonl.extend(json.loads(line) for line in f)

        # Calculate scores by difficulty
        difficulty_scores = {}
        fails = []
        for result in results_jsonl:
            difficulty = result.get('difficulty', 'unknown')
            if difficulty not in difficulty_scores:
                difficulty_scores[difficulty] = {'passed': 0, 'total': 0}

            difficulty_scores[difficulty]['total'] += 1
            if result['passed']:
                difficulty_scores[difficulty]['passed'] += 1
            else:
                fails.append(result['task_id'].split("/")[-1])

        # Calculate overall score
        total_passed = sum(d['passed'] for d in difficulty_scores.values())
        total_problems = sum(d['total'] for d in difficulty_scores.values())
        score = total_passed / total_problems if total_problems > 0 else 0

        details = {
            "score": score * 100,  # Convert to percentage
            "total_passed": total_passed,
            "total_problems": total_problems,
            "difficulty_scores": difficulty_scores,
            "failed_tasks": fails
        }

        return score * 100, details
