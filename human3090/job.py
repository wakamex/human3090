"""Job definition and YAML queue loader for benchmark runs."""

import argparse
import shutil
from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class Job:
    """A single benchmark run configuration."""

    model: str
    benchmark: str
    gpu_layers: int = 80
    context_size: int = 4096
    temperature: float = 0.0
    top_p: float = 0.9
    min_p: float = 0.1
    max_tokens: int = 1000
    start_problem: int = 1
    end_problem: int | None = None
    preamble: str | None = None
    save_raw: bool = False
    # LCB-specific
    problems_file: str = "test5.jsonl"
    start_date: str | None = None
    end_date: str | None = None

    @property
    def model_shortname(self) -> str:
        """E.g. 'Qwen3.5-27B-Q4_K_M' from '/seagate/models/Qwen3.5-27B-Q4_K_M.gguf'."""
        return Path(self.model).stem

    @property
    def output_file(self) -> str:
        """The JSONL output file for this benchmark run."""
        suffix = "human_eval" if self.benchmark == "human_eval" else "lcb"
        return f"{self.model_shortname}_{suffix}.jsonl"

    def server_key(self) -> tuple:
        """Jobs with the same server_key can reuse a running server."""
        return (self.model, self.gpu_layers, self.context_size)

    def job_id(self) -> str:
        """Human-readable identifier for this job."""
        return f"{self.model_shortname}/{self.benchmark}"


def _parse_job_file(path: Path) -> list[Job]:
    """Parse a single YAML job file, expanding benchmarks list into Jobs."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    job_field_names = {f.name for f in fields(Job)}

    # Handle benchmarks list (expand to multiple jobs)
    benchmarks = data.pop("benchmarks", None)
    if benchmarks is None:
        benchmarks = [data.pop("benchmark")]
    else:
        data.pop("benchmark", None)

    # Pop any LCB-nested config
    lcb_config = data.pop("lcb", {})

    jobs = []
    for benchmark in benchmarks:
        job_kwargs = {"benchmark": benchmark}
        for key, value in data.items():
            if key in job_field_names:
                job_kwargs[key] = value
        if benchmark == "lcb" and lcb_config:
            for key, value in lcb_config.items():
                if key in job_field_names:
                    job_kwargs[key] = value
        jobs.append(Job(**job_kwargs))

    return jobs


def load_queue_dir(jobs_dir: str = "jobs") -> list[tuple[Path, list[Job]]]:
    """Load all YAML files from jobs_dir/queued/, sorted by filename.

    Returns list of (yaml_path, jobs) tuples. Each file may expand to
    multiple Jobs if it uses `benchmarks: [a, b]`.

    Directory structure:
        jobs/
          queued/   <- pending job files (*.yaml)
          done/     <- completed job files (moved here on success)
          failed/   <- failed job files (moved here on failure)
    """
    queued_dir = Path(jobs_dir) / "queued"
    if not queued_dir.exists():
        raise FileNotFoundError(f"No queued directory found at {queued_dir}")

    # Ensure done/failed dirs exist
    for subdir in ("done", "failed"):
        (Path(jobs_dir) / subdir).mkdir(parents=True, exist_ok=True)

    results = []
    for yaml_file in sorted(queued_dir.glob("*.yaml")):
        jobs = _parse_job_file(yaml_file)
        results.append((yaml_file, jobs))

    return results


def next_queued_file(jobs_dir: str = "jobs") -> tuple[Path, list[Job]] | None:
    """Return the next YAML file from queued/, or None if empty.

    Called in a loop so the queue picks up new files dropped in mid-run.
    """
    queued_dir = Path(jobs_dir) / "queued"
    if not queued_dir.exists():
        return None
    for yaml_file in sorted(queued_dir.glob("*.yaml")):
        return (yaml_file, _parse_job_file(yaml_file))
    return None


def move_job_file(yaml_path: Path, dest_dir: str) -> None:
    """Move a job file to the destination directory (done/ or failed/)."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.move(str(yaml_path), str(dest / yaml_path.name))


def job_from_cli(args: argparse.Namespace) -> Job:
    """Convert CLI argparse Namespace to a single Job (backward-compat)."""
    return Job(
        model=args.model,
        benchmark=args.benchmark,
        gpu_layers=args.gpu_layers,
        context_size=args.context_size,
        temperature=args.temperature,
        top_p=args.top_p,
        min_p=args.min_p,
        max_tokens=args.max_tokens,
        start_problem=args.start_problem,
        end_problem=args.end_problem,
        preamble=args.preamble,
        save_raw=args.save_raw,
        problems_file=args.problems_file or "test5.jsonl",
        start_date=args.start_date,
        end_date=args.end_date,
    )
