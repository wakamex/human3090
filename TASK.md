# Benchmark Runner Plan

## Overview
Create a generic benchmark runner that can manage different benchmark types (HumanEval, LCB, etc.) while handling server lifecycle and results.

## Components

### 1. Core Structure ✓
```
/code/human3090/
├── bench_runner.py      # Main runner script
├── parsers/             # Benchmark-specific parsers
│   ├── __init__.py     # Parser registry
│   ├── base.py         # Base parser class
│   └── human_eval.py   # HumanEval parser
└── benchmark_results.json
```

### 2. Results Storage ✓
JSON structure for storing benchmark results:
```json
{
  "runs": [
    {
      "timestamp": "2025-02-16T18:15:08",
      "benchmark": "human_eval",
      "model": "model-name",
      "config": {
        "path": "/path/to/model.gguf",
        "gpu_layers": 80
      },
      "results": {
        "score": 89.0,
        "time_taken": 2365.32,
        "total_passed": 145,
        "total_problems": 164,
        "difficulty_scores": {},
        "failed_tasks": []
      }
    }
  ]
}
```

## Usage
```bash
./bench_runner.py \
    --model /path/to/model.gguf \
    --gpu-layers 40 \
    --benchmark human_eval \
    --script run_eval.py \
    --temperature 0.0 \
    --max-tokens 1000
```

## Implementation Progress

✓ 1. Created modular structure with parsers directory
✓ 2. Implemented server lifecycle management
✓ 3. Added results storage and processing
✓ 4. Created HumanEval parser
✓ 5. Added basic CLI interface
✓ 6. Added model name handling and README.md updates
✓ 7. Successfully ran and validated HumanEval benchmark

## Next Steps

1. [ ] Add support for run_lcb.py
2. [ ] Add benchmark-specific CLI arguments (context length, top_p)
3. [ ] Add error handling for server startup
4. [ ] Add progress reporting during long runs
5. [ ] Consider parallel processing for faster evaluation
6. [ ] Consider batched inference if beneficial
