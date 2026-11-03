# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""MkDocs hooks for the documentation site (see ``mkdocs.yml``).

The pages in ``docs/`` are written for GitHub's renderer first, and the site
adapts four things rather than asking every page to be written twice:

* **Heading ids use GitHub's slug rules**, so ``page.md#some-heading`` resolves
  on the site exactly when it resolves on GitHub.  (Set here rather than in
  ``mkdocs.yml``, where it needs a ``!!python/...`` tag that pre-commit's
  check-yaml rejects.)
* **Links that leave ``docs/``** (``../README.md``, ``../oasr/engine/...``) name
  files the site does not publish, so they are rewritten to GitHub URLs.
* **Each page gets its own meta description**, taken from its first paragraph.
  Without one, every page repeats ``site_description``, which search engines
  treat as no description at all.
* **``llms.txt`` is generated at the site root** from the navigation, as the
  llms.txt proposal (https://llmstxt.org/) describes, so the index a coding
  agent reads cannot drift from the pages it lists.
"""

from __future__ import annotations

import posixpath
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig
    from mkdocs.structure.files import Files
    from mkdocs.structure.nav import Navigation
    from mkdocs.structure.pages import Page

BRANCH = "main"
DESCRIPTION_CHARS = 160

# A notice an agent needs before it acts on anything else in llms.txt.
LLMS_NOTES = (
    "OASR installs from source: `git clone https://github.com/chiendb97/oasr.git`, "
    '`git submodule update --init`, then `pip install -e ".[all]"`. The package is named '
    "`oasr-server` and imports as `oasr`; the `oasr` package on PyPI is an unrelated project.",
    "OASR serves on NVIDIA GPUs of compute capability 7.5 or newer; it does not serve on CPU.",
)

_FENCE = re.compile(r"^\s*(```|~~~)")
_LINK = re.compile(r"\]\((\.\./[^)\s#]*)(#[^)\s]*)?\)")
_MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_NOT_PROSE = ("#", "|", "-", "*", ">", "<", "```", "~~~", "!", "[")

_nav: Optional["Navigation"] = None


def _github_url(config: "MkDocsConfig", repo_path: str) -> str:
    root = Path(config["docs_dir"]).parent
    kind = "tree" if (root / repo_path).is_dir() else "blob"
    return f"{config['repo_url'].rstrip('/')}/{kind}/{BRANCH}/{repo_path}"


def _rewrite_external_links(markdown: str, page: "Page", config: "MkDocsConfig") -> str:
    page_dir = posixpath.join("docs", posixpath.dirname(page.file.src_uri))
    out: List[str] = []
    in_fence = False
    for line in markdown.split("\n"):
        if _FENCE.match(line):
            in_fence = not in_fence
        if not in_fence:

            def repl(m: "re.Match[str]") -> str:
                target = posixpath.normpath(posixpath.join(page_dir, m.group(1)))
                if target == "docs" or target.startswith("docs/"):
                    return m.group(0)
                return f"]({_github_url(config, target)}{m.group(2) or ''})"

            line = _LINK.sub(repl, line)
        out.append(line)
    return "\n".join(out)


def _first_paragraph(markdown: str) -> str:
    lines: List[str] = []
    in_fence = False
    for line in markdown.split("\n"):
        if _FENCE.match(line):
            in_fence = not in_fence
            if lines:
                break
            continue
        stripped = line.strip()
        if in_fence or not stripped:
            if lines:
                break
            continue
        if stripped.startswith(_NOT_PROSE):
            if lines:
                break
            continue
        lines.append(stripped)
    text = _MD_LINK.sub(r"\1", " ".join(lines))
    # Code spans keep their text verbatim (``oasr.from_pretrained``, ``*.cuh``);
    # only emphasis markers outside them are dropped.
    parts = re.split(r"(`[^`]*`)", text)
    for i, part in enumerate(parts):
        if part.startswith("`"):
            parts[i] = part.strip("`")
        else:
            parts[i] = re.sub(r"(?<!\w)[*_]{1,2}(?=\S)|(?<=\S)[*_]{1,2}(?!\w)", "", part)
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def _summary(text: str) -> str:
    if text.endswith(":"):
        text = text[:-1] + "."
    if len(text) <= DESCRIPTION_CHARS:
        return text
    cut = text[: DESCRIPTION_CHARS - 1].rsplit(" ", 1)[0].rstrip(",;:—-")
    return cut + "…"


def on_config(config: "MkDocsConfig") -> "MkDocsConfig":
    from pymdownx.slugs import slugify

    config["mdx_configs"].setdefault("toc", {})["slugify"] = slugify(case="lower")
    return config


def on_nav(nav: "Navigation", config: "MkDocsConfig", files: "Files") -> "Navigation":
    global _nav
    _nav = nav
    return nav


def on_page_markdown(markdown: str, page: "Page", config: "MkDocsConfig", files: "Files") -> str:
    # The theme titles the home page with the bare site name; give it the
    # descriptive title search results show instead.
    home_title = config["extra"].get("home_title")
    if page.is_homepage and home_title and not page.meta.get("title"):
        page.meta["title"] = home_title
    if not page.meta.get("description"):
        paragraph = _first_paragraph(markdown)
        if paragraph:
            page.meta["description"] = _summary(paragraph)
    return _rewrite_external_links(markdown, page, config)


def on_post_build(config: "MkDocsConfig") -> None:
    if _nav is None:
        return
    raw = config["repo_url"].rstrip("/").replace("github.com", "raw.githubusercontent.com")
    raw = f"{raw}/{BRANCH}"
    out = [f"# {config['site_name']}", "", f"> {' '.join(config['site_description'].split())}", ""]
    out += [f"- {note}" for note in LLMS_NOTES]
    out += ["", "## Overview", ""]
    out.append(f"- [README]({raw}/README.md): overview, install, quick start, supported models")
    out.append(f"- [AGENTS.md]({raw}/AGENTS.md): engineering rules for contributors and agents")

    def page_line(page: Any) -> str:
        desc = page.meta.get("description", "")
        line = f"- [{page.title}]({raw}/docs/{page.file.src_uri})"
        return f"{line}: {desc}" if desc else line

    for item in _nav.items:
        if item.is_page:
            continue  # the docs index; its content is the README links above
        if item.is_section:
            out += ["", f"## {item.title}", ""]
            out += [page_line(child) for child in item.children if child.is_page]
    Path(config["site_dir"], "llms.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
