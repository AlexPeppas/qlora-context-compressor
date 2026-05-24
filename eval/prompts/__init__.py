"""Versioned prompt templates for the Phase B judges.

Each prompt is a plain text file in this directory. The convention is:

    <rubric>_<stage>_v<N>.md

where `<rubric>` is the metric (faithfulness, downstream, ...),
`<stage>` is the role within that rubric (stage1 for extraction,
stage2 for coverage), and `v<N>` is the version. Bumping `<N>`
invalidates the judge result cache (see eval.llm_client.cache_key).

A PromptTemplate is loaded from a file via `load_prompt()`, exposing the
raw content + a stable sha256 hash. The hash is recorded in every
JudgeProvenance so result rows are self-describing.

Format inside a prompt file:

    [SYSTEM]
    <system message>
    [USER]
    <user message template, with {placeholder} substitutions>

We use Python str.format()-style placeholders (single braces). Literal
braces in the template must be escaped as {{ and }}. We do NOT use any
fancy templating (Jinja2 etc.) so the file is the source of truth and
the resulting prompt string is trivially reconstructible from it.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PromptTemplate:
    name: str  # e.g. "faithfulness_stage1_v1"
    system_template: str  # raw [SYSTEM] section
    user_template: str  # raw [USER] section
    content_hash: str  # sha256 of the whole file content

    def render(self, **kwargs: str) -> tuple[str, str]:
        """Return (system, user) strings with placeholders filled in.

        Raises KeyError if a required placeholder is missing.
        """
        system = self.system_template.format(**kwargs) if kwargs else self.system_template
        user = self.user_template.format(**kwargs)
        return system, user


_SECTION_RE = re.compile(r"^\[(SYSTEM|USER)\]\s*$", re.MULTILINE)


def _parse_prompt_file(content: str) -> tuple[str, str]:
    """Parse a prompt file into (system, user) sections.

    Sections are introduced by `[SYSTEM]` and `[USER]` markers on their own
    lines. The content of each section is everything between its marker and
    the next marker (or end of file). Whitespace at the boundary is
    stripped.
    """
    sections = {"SYSTEM": "", "USER": ""}
    matches = list(_SECTION_RE.finditer(content))
    if not matches:
        raise ValueError("prompt file has no [SYSTEM] or [USER] section markers")
    for i, m in enumerate(matches):
        section = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        sections[section] = content[start:end].strip()
    if not sections["USER"]:
        raise ValueError("prompt file must include a non-empty [USER] section")
    return sections["SYSTEM"], sections["USER"]


def load_prompt(name: str) -> PromptTemplate:
    """Load a prompt template by name (without `.md` extension).

    The file `{name}.md` must exist in this directory.
    """
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"prompt template not found: {path}")
    content = path.read_text(encoding="utf-8")
    system, user = _parse_prompt_file(content)
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return PromptTemplate(
        name=name,
        system_template=system,
        user_template=user,
        content_hash=content_hash,
    )


__all__ = ["PromptTemplate", "load_prompt", "PROMPTS_DIR"]
