"""Base parser for benchmark results."""

from typing import Any, Dict, Tuple

class BenchmarkParser:
    """Base class for benchmark-specific result parsers."""

    def parse_results(self, output_file: str) -> Tuple[float, Dict[str, Any]]:
        """Parse benchmark results and return (score, details)."""
        raise NotImplementedError
