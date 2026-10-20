"""Paths in the repo."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Check that the package runs from its repo checkout, where skills/, workspace/ and logs/ live.
if not (ROOT / "pyproject.toml").is_file():
    raise RuntimeError(f"jean-code runs from its repo checkout (an editable install); ROOT={ROOT}")
SKILLS_DIR = ROOT / "skills"
WORKSPACE_DIR = ROOT / "workspace"  # the UIs' workspace: the sandbox's /workspace, with research's runs in research/
USAGE_LOG = ROOT / "logs" / "usage.jsonl"
PROCESS_LOGS_DIR = ROOT / "logs" / "processes"  # one folder per process, for agents' logs
