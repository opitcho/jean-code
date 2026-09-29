"""Paths in the repo that the front-ends use."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = ROOT / "skills"
RESEARCH_DIR = ROOT / "workspace" / "research"  # research run folders; the sandbox sees them at /workspace/research
USAGE_LOG = ROOT / "logs" / "usage.jsonl"
