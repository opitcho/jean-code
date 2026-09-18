"""Paths in the repo that the front-ends use."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = ROOT / "skills"
USAGE_LOG = ROOT / "logs" / "usage.jsonl"
