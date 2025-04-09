#!/bin/bash
if [[ -z "$1" ]]; then
  echo "Usage: $0 <input_samples_file.jsonl>"
  exit 1 # Exit if no argument is provided
fi

filename=$1

echo "Scoring $filename"

.venv/bin/python -m human_eval.evaluate_functional_correctness $filename

results_filename=${filename}_results.jsonl

inspect_result $results_filename

