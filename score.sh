#!/bin/bash
if [[ -z "$1" ]]; then
  echo "Usage: $0 <input_samples_file.jsonl>"
  exit 1 # Exit if no argument is provided
fi

filename=$1

echo "Scoring $filename"

if [[ "$filename" == *lcb* ]]; then
  echo "Using LCB evaluation for $filename"
  .venv/bin/python ./human3090/evaluate_lcb.py "$filename" "/code/human3090/test5.jsonl"
else
  echo "Using HumanEval evaluation for $filename"
  .venv/bin/python -m human_eval.evaluate_functional_correctness "$filename"
fi

results_filename=${filename}_results.jsonl

.venv/bin/python ./human3090/inspect_result.py $results_filename

