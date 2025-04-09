"""Functions for processing benchmark results."""

import json
import os
from typing import Any, Dict, List, Tuple


def parse_results(output_file: str) -> Tuple[float, Dict[str, Any]]:
    """Parse benchmark results from output file.

    Args:
        output_file: Path to results file (without _results.jsonl suffix)

    Returns:
        Tuple of (score, details) where score is a percentage and details contains:
            - total_passed: Number of passed problems
            - total_problems: Total number of problems
            - difficulty_scores: Dict of difficulty -> {passed, total}
            - failed_tasks: List of failed task IDs
    """
    results_file = f"{output_file}_results.jsonl"
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")

    # Parse results
    results_list = []
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            result = json.loads(line)
            result['passed'] = bool(result['passed']) # Ensure passed is boolean
            results_list.append(result)

    return calculate_scores(results_list)


def calculate_scores(results_list: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
    """Calculate scores from a list of results.
    
    Args:
        results_list: List of result dicts, each containing at least:
            - task_id: str
            - passed: bool
            - difficulty: str

    Returns:
        Tuple of (score, details) where score is a percentage and details contains:
            - total_passed: Number of passed problems
            - total_problems: Total number of problems
            - difficulty_scores: Dict of difficulty -> {passed, total}
            - failed_tasks: List of failed task IDs
    """
    difficulty_scores = {}
    total_passed = 0
    failed_tasks = []

    for result in results_list:
        difficulty = result.get('difficulty', 'unknown')
        if difficulty not in difficulty_scores:
            difficulty_scores[difficulty] = {'passed': 0, 'total': 0}

        difficulty_scores[difficulty]['total'] += 1
        # Handle both boolean and string values for passed
        passed = result['passed']
        if isinstance(passed, str):
            passed = (passed.lower() == 'true')
        if passed:
            difficulty_scores[difficulty]['passed'] += 1
            total_passed += 1
        else:
            failed_tasks.append(result['task_id'])

    total_problems = len(results_list)
    score = (total_passed / total_problems * 100) if total_problems > 0 else 0

    details = {
        'total_passed': total_passed,
        'total_problems': total_problems,
        'difficulty_scores': difficulty_scores,
        'failed_tasks': sorted(failed_tasks)
    }

    return score, details
