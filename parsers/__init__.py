"""Benchmark result parsers."""

from .human_eval import HumanEvalParser

PARSERS = {
    "human_eval": HumanEvalParser()
}
