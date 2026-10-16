"""Paths in the repo."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Check that the package runs from its repo checkout, where skills/, workspace/ and logs/ live.
if not (ROOT / "pyproject.toml").is_file():
    raise RuntimeError(f"jean-code runs from its repo checkout (an editable install); ROOT={ROOT}")
SKILLS_DIR = ROOT / "skills"
WORKSPACE_DIR = ROOT / "workspace"  # the sandbox's /workspace
RESEARCH_DIR = WORKSPACE_DIR / "research"  # research run folders; the sandbox sees them at /workspace/research
USAGE_LOG = ROOT / "logs" / "usage.jsonl"
