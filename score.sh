#!/bin/bash

# Check if an argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <filename.jsonl>"
    exit 1
fi

# Use the first argument as the filename
filename="$1"

# run only if the file exists
if [ ! -f "$filename" ]; then
    echo "File not found: $filename"
    exit 1
fi

# Run the human_eval script with the provided filename
.venv/bin/python -c "from human_eval.evaluate_functional_correctness import main; main()" "$filename"

# Assuming the results are saved with '_results.jsonl' appended to the original filename
result_filename="${filename}_results.jsonl"

# Run the inspect_result script with the results filename
.venv/bin/python inspect_result.py "$result_filename"
