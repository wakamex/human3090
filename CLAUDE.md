thinking runs should always be run with --max-tokens=32000
never force-add or commit *.jsonl result files to git. the .gitignore excludes them intentionally - only README.md, benchmark_results.json, and the plot image track scores. raw result files stay local.
use uv run python instead of .venv/bin/python
