#!/bin/bash

# Check if an argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <filename.jsonl>"
    exit 1
fi

# Use the first argument as the filename
filename="$1"

# Run the human_eval script with the provided filename
python -m human_eval.evaluate_functional_correctness "$filename"

# Assuming the results are saved with '_results.jsonl' appended to the original filename
result_filename="${filename}_results.jsonl"

# Run the inspect_result script with the results filename
python inspect_result.py "$result_filename"
