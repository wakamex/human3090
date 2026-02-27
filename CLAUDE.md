thinking runs should always be run with --max-tokens=32000
never force-add or commit *.jsonl result files to git. the .gitignore excludes them intentionally - only README.md, benchmark_results.json, and the plot image track scores. raw result files stay local.
use uv run python instead of .venv/bin/python
Search Claude Code session history

usage: sch [-h] [-V] [-p PROJECT] [-t {user,assistant}] [-A N] [-B N] [--tools] [-C CHARS] [--no-color] pattern

positional arguments:
  pattern               Search pattern (supports regex)

options:
  -h, --help            show this help message and exit
  -V, --version         show program's version number and exit
  -p, --project PROJECT
                        Filter to project (substring match on dir name)
  -t, --type {user,assistant}
                        Filter by message role
  -A N                  Show N messages after match
  -B N                  Show N messages before match
  --tools               Include tool_use/tool_result messages (skipped by default)
  -C, --context CHARS   Chars of text context around match (default 200)
  --no-color            Disable colored output
