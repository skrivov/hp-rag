"""Dataset adapters for benchmarking corpora."""

from .base import DatasetAdapter, SectionFragment, build_section_tree
from .beir import BeirCorpusAdapter
from .hotpotqa import HotpotQAAdapter
from .miracl import MiraclAdapter
from .squad import SquadAdapter


__all__ = [
    "DatasetAdapter",
    "SectionFragment",
    "build_section_tree",
    "BeirCorpusAdapter",
    "HotpotQAAdapter",
    "MiraclAdapter",
    "SquadAdapter",
]
