"""Skill folders: a SKILL.md (YAML frontmatter plus instructions) and the files bundled with it.

A skill loads in three levels, so the model only pays for what it uses:
    1. its name and description, listed in the system prompt (`Skills.prompt`)
    2. its SKILL.md body, when the model calls `load_skill` or the user types `/name` (`Skills.load_skill`)
    3. its bundled files, one at a time through `read_skill_file` (`Skills.read_skill_file`)
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

from llm import tool


@dataclass
class Skill:
    """A skill folder: SKILL.md metadata plus the folder holding its instructions and files."""

    name: str
    description: str
    path: Path


def split_frontmatter(text: str) -> tuple[dict, str]:
    """Split a SKILL.md into its YAML frontmatter and its body."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*(?:\n|$)(.*)", text, re.DOTALL)
    if match is None:
        return {}, text
    return yaml.safe_load(match.group(1)) or {}, match.group(2).lstrip()


def skill_files(skill_dir: Path) -> list[Path]:
    """Files bundled with a skill, relative to its folder, skipping SKILL.md and hidden or dunder paths."""
    files = []
    for path in sorted(skill_dir.rglob("*")):
        rel = path.relative_to(skill_dir)
        hidden = any(part.startswith((".", "__")) for part in rel.parts)
        if path.is_file() and rel != Path("SKILL.md") and not hidden:
            files.append(rel)
    return files


def discover_skills(skills_dir: str | Path) -> dict[str, Skill]:
    """Read the metadata of every `<skills_dir>/<name>/SKILL.md`, keyed by skill name."""
    skills = {}
    for skill_md in sorted(Path(skills_dir).glob("*/SKILL.md")):
        meta, _ = split_frontmatter(skill_md.read_text())
        name = meta.get("name", skill_md.parent.name)
        skills[name] = Skill(name, meta.get("description", ""), skill_md.parent)
    return skills


def instructions(skill: Skill) -> str:
    """A skill's SKILL.md body, followed by the list of its other files."""
    _, body = split_frontmatter((skill.path / "SKILL.md").read_text())
    files = skill_files(skill.path)
    if files:
        listing = "\n".join(f"- {rel}" for rel in files)
        body = f"{body.rstrip()}\n\nFiles in this skill (read with read_skill_file):\n{listing}"
    return body


def read_file(skill: Skill, path: str) -> str:
    """A file bundled with a skill; raises ValueError for paths outside its folder."""
    root = skill.path.resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError(f"'{path}' is not a file in skill '{skill.name}'")
    return target.read_text()


class Skills:
    """An agent's skills: the listing for its system prompt, the two tools its model calls, and the `/name` commands.

    Holds no transcript: whether a skill's instructions are still in context is the model's to judge. A skill can be
    loaded again, e.g. after compaction summarized its instructions away.
    """

    def __init__(self, skills_dir: str | Path | None):
        self.catalog: dict[str, Skill] = discover_skills(skills_dir) if skills_dir else {}
        self.loaded: list[str] = []  # names loaded this session, first load first; for UIs

    def prompt(self) -> str:
        """The level-1 listing for the system prompt: names and descriptions only. Empty without skills."""
        if not self.catalog:
            return ""
        lines = [
            "## Skills",
            "Skills hold instructions for specific tasks. When a task matches a skill, call load_skill with its "
            "name, unless its instructions are already in the conversation.",
            "",
        ]
        lines += [f"- {skill.name}: {skill.description}" for skill in self.catalog.values()]
        return "\n".join(lines)

    def tools(self) -> dict[str, Callable]:
        """The tools to give the model, by name; none without skills."""
        if not self.catalog:
            return {}
        return {"load_skill": self.load_skill, "read_skill_file": self.read_skill_file}

    def commands(self) -> dict[str, Callable[[str], str | None]]:
        """`/name` for each skill, as a function of the rest of the line: loads the skill."""
        return {
            # name=name binds each name now: a closure would see only the loop's last one
            name: lambda rest, name=name: self.load_skill(name) for name in self.catalog
        }

    @tool
    def load_skill(self, name: str) -> str:
        """Load a skill's instructions. Call this before starting a task that matches a skill.

        Args:
            name: Name of the skill, from the list of available skills.

        Returns:
            The skill's instructions, followed by the other files it provides.
        """
        body = instructions(self._get(name))
        if name not in self.loaded:
            self.loaded.append(name)
        return body

    @tool
    def read_skill_file(self, name: str, path: str) -> str:
        """Read a file bundled with a skill, such as a reference document or a script.

        Args:
            name: Name of the skill the file belongs to.
            path: Path of the file relative to the skill's folder, as listed by load_skill.

        Returns:
            The file's text.
        """
        return read_file(self._get(name), path)

    def _get(self, name: str) -> Skill:
        if name not in self.catalog:
            raise ValueError(f"unknown skill '{name}'; available: {', '.join(self.catalog)}")
        return self.catalog[name]
