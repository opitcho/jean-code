"""Skill folders: a SKILL.md (YAML frontmatter plus instructions) and the files bundled with it."""

import re
from dataclasses import dataclass
from pathlib import Path

import yaml


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


def skills_prompt(skills: dict[str, Skill]) -> str:
    """The level-1 skills listing for the system prompt: names and descriptions only."""
    if not skills:
        return ""
    lines = [
        "## Skills",
        "Skills hold instructions for specific tasks. When a task matches a skill, call "
        "load_skill with its name once; its instructions then stay in the conversation.",
        "",
    ]
    lines += [f"- {skill.name}: {skill.description}" for skill in skills.values()]
    return "\n".join(lines)


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
