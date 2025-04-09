#!/bin/bash
set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <filename.jsonl>"
    exit 1
fi

filename="$1"

if [ ! -f "$filename" ]; then
    echo "File not found: $filename"
    exit 1
fi

# Run via -m so Python resolves it as a module
.venv/bin/python -m human_eval.evaluate_functional_correctness "$filename"

result_filename="${filename}_results.jsonl"

.venv/bin/python inspect_result.py "$result_filename"
