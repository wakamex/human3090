# Benchmark Runner Plan

## Overview
Create a generic benchmark runner that can manage different benchmark types (HumanEval, LCB, etc.) while handling server lifecycle and results.

## Components

### 1. Benchmark Runner (bench_runner.py)
```python
class BenchmarkRunner:
    def __init__(self, model_path, gpu_layers):
        self.model_path = model_path
        self.gpu_layers = gpu_layers
        self.results_file = 'benchmark_results.json'

    @contextmanager
    def llama_server(self, port=8083):
        process = subprocess.Popen([
            "./build/bin/llama-server",
            "-m", self.model_path,
            "--port", str(port),
            "-ngl", str(self.gpu_layers)
        ])
        try:
            time.sleep(5)
            yield port
        finally:
            process.terminate()

    def run_benchmark(self, benchmark_script, **kwargs):
        with self.llama_server() as port:
            start_time = time.time()
            result = subprocess.run(
                ['.venv/bin/python', benchmark_script, **kwargs],
                capture_output=True,
                text=True
            )
            duration = time.time() - start_time
            return self._process_results(result, duration)
```

### 2. Results Storage
Simple JSON structure for all benchmarks:
```json
{
  "runs": [
    {
      "timestamp": "2025-02-16T18:15:08",
      "benchmark": "human_eval",  # or "lcb", etc.
      "model": "model-name",
      "config": {
        "path": "/path/to/model.gguf",
        "gpu_layers": 80
      },
      "results": {
        "score": 89.0,
        "time_taken": 2365.32,
        "raw_output": "..."
      }
    }
  ]
}
```

## Usage
```bash
# Run HumanEval
./bench_runner.py run \
    --benchmark human_eval \
    --script run_eval.py \
    --model /path/to/model.gguf \
    --gpu-layers 80

# Run LCB
./bench_runner.py run \
    --benchmark lcb \
    --script run_lcb.py \
    --model /path/to/model.gguf \
    --gpu-layers 80
```

## Implementation Steps

1. [ ] Create bench_runner.py with server management
2. [ ] Add results storage and processing
3. [ ] Add CLI interface
4. [ ] Create benchmark-specific result parsers
5. [ ] Test with run_eval.py
6. [ ] Add support for run_lcb.py
