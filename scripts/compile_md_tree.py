#!/usr/bin/env python3
"""Flatten the paper/MAST.md note tree into paper/arxiv.md, then (optionally) build TeX and PDF.

Adapted from catskillsresearch/lyasoff2016 ``scripts/compile_md_tree.py`` (Lean listings, AI-prose
imports, gist boxes and hole tracking removed). Mechanism, unchanged:

* Walk the outline bullets of a note (``- [Title](Child.md)`` or ``1. [Title](Child.md)``) in
  document order. Each link hop adds one heading level, so LaTeX numbering runs 1 -> 1.1 -> 1.1.1.
  Section titles come from the link text, not from the child's own ``#`` heading (which is
  dropped so every note stays readable on its own).
* ``@import "file.md"`` (e.g. generated tables) is expanded inline; ``@import "x.dot"`` is
  rendered with Graphviz into ``paper/figures`` and inserted as a figure.
* Relative image links in child notes are rebased so they resolve from ``paper/``.
* The root's YAML front matter (``bibliography: references.bib``) is kept; a ``## References``
  heading is appended for the BibTeX bibliography.

    python3 scripts/compile_md_tree.py              # flatten + tex + pdf
    python3 scripts/compile_md_tree.py --md-only    # write paper/arxiv.md only
    python3 scripts/compile_md_tree.py --tex-only   # arxiv.md + arxiv.tex
"""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
PAPER = ROOT / "paper"
DEFAULT_ROOT_MD = PAPER / "MAST.md"
ARXIV_MD = PAPER / "arxiv.md"

TOC_BULLET_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?:[-*][ \t]+|\d+\.\s+)\[(?P<title>.+?)\]\((?P<href>[^)]+\.md)\)[ \t]*$",
    re.MULTILINE,
)
FIRST_H1_RE = re.compile(r"^#\s+.+?(?:\n+|$)", re.MULTILINE)
IMPORT_RE = re.compile(r'^@import[ \t]+"([^"]+)"(?:[ \t]*\{([^}]*)\})?[ \t]*$', re.MULTILINE)
HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+)$", re.MULTILINE)
IMAGE_RE = re.compile(r"(!\[[^\]]*\]\()([^)\s]+)(\))")
FRONT_MATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)


def indent_columns(indent: str) -> int:
    return len(indent.expandtabs(4))


def shift_headings(text: str, delta: int) -> str:
    if delta == 0:
        return text

    def repl(m: re.Match[str]) -> str:
        return "#" * min(max(len(m.group(1)) + delta, 1), 6) + " " + m.group(2)

    return HEADING_RE.sub(repl, text)


def align_body_headings(text: str, base_level: int) -> str:
    """Shift headings so the shallowest one in ``text`` is ``base_level``."""
    outside = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    levels = [len(m.group(1)) for m in HEADING_RE.finditer(outside)]
    return shift_headings(text, base_level - min(levels)) if levels else text


def rebase_images(text: str, note: Path) -> str:
    def repl(m: re.Match[str]) -> str:
        href = m.group(2)
        if re.match(r"^[a-z]+://", href) or href.startswith("/"):
            return m.group(0)
        tgt = (note.parent / href).resolve()
        try:
            rel = tgt.relative_to(PAPER).as_posix()
        except ValueError:
            rel = href
        return f"{m.group(1)}{rel}{m.group(3)}"

    return IMAGE_RE.sub(repl, text)


def expand_import(note: Path, rel: str, opts: str) -> str:
    target = (note.parent / rel).resolve()
    if not target.is_file():
        return f"\n\n*Missing import:* `{rel}`\n\n"
    suffix = target.suffix.lower()
    if suffix == ".md":
        body = target.read_text(encoding="utf-8").rstrip() + "\n"
        return "\n\n" + rebase_images(body, target) + "\n"
    if suffix == ".dot":
        cap = re.search(r'caption="([^"]*)"', opts or "")
        caption = cap.group(1) if cap else target.stem
        digest = hashlib.sha1(target.read_bytes()).hexdigest()[:10]
        out = PAPER / "figures" / f"{target.stem}-{digest}.pdf"
        if not out.is_file():
            subprocess.run(["dot", "-Tpdf", str(target), "-o", str(out)], check=True)
        return f"\n\n![{caption}]({out.relative_to(PAPER).as_posix()})\n\n"
    body = target.read_text(encoding="utf-8").rstrip() + "\n"
    return f"\n\n```\n{body}```\n\n"


def expand_imports(note: Path, text: str) -> str:
    return IMPORT_RE.sub(lambda m: expand_import(note, m.group(1), m.group(2) or ""), text)


def child_md_targets(note: Path, text: str) -> list[tuple[int, str, Path]]:
    matches = list(TOC_BULLET_RE.finditer(text))
    if not matches:
        return []
    base = min(indent_columns(m.group("indent")) for m in matches)
    seen: set[Path] = set()
    out = []
    for m in matches:
        tgt = (note.parent / m.group("href")).resolve()
        if tgt in seen or not tgt.is_file():
            continue
        seen.add(tgt)
        out.append(((indent_columns(m.group("indent")) - base) // 2, m.group("title").strip(), tgt))
    return out


def flatten_file(note: Path, *, level: int, title: str | None, visited: set[Path]) -> str:
    note = note.resolve()
    if note in visited:
        return f"\n\n*(Already included: `{note.name}`.)*\n\n"
    visited.add(note)
    raw = note.read_text(encoding="utf-8")
    children = child_md_targets(note, raw)
    body = TOC_BULLET_RE.sub("", raw) if children else raw
    if title is not None:
        body = FIRST_H1_RE.sub("", body, count=1)
    body = expand_imports(note, body)
    body = rebase_images(body, note)
    if title is not None:
        body = align_body_headings(body, min(level + 1, 6))
    body = re.sub(r"\n{3,}", "\n\n", body).strip() + "\n"
    parts = [f"{'#' * level} {title}\n\n{body}" if title is not None else body]
    for rel_indent, child_title, child in children:
        parts.append(flatten_file(child, level=min(level + 1 + rel_indent, 6), title=child_title, visited=visited))
    return "\n".join(parts)


def flatten_md_tree(root_md: Path = DEFAULT_ROOT_MD, out_path: Path = ARXIV_MD) -> Path:
    visited: set[Path] = set()
    body = flatten_file(root_md, level=1, title=None, visited=visited)
    if not re.search(r"^#{1,6}\s+References\s*$", body, re.MULTILINE):
        body = body.rstrip() + "\n\n## References\n"
    header = ("<!-- AUTO-GENERATED by scripts/compile_md_tree.py -- do not edit -->\n"
              f"<!-- Root: {root_md.relative_to(ROOT).as_posix()} ({len(visited)} notes) -->\n\n")
    fm = FRONT_MATTER_RE.match(body)
    if fm:  # keep YAML front matter first so pandoc / viewers see it
        body = fm.group(0) + "\n" + header + body[fm.end():].lstrip("\n")
    else:
        body = header + body
    out_path.write_text(body, encoding="utf-8")
    print(f"wrote {out_path.relative_to(ROOT)} ({out_path.stat().st_size:,} bytes, {len(visited)} notes)")
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT_MD)
    ap.add_argument("--out", type=Path, default=ARXIV_MD)
    ap.add_argument("--md-only", action="store_true")
    ap.add_argument("--tex-only", action="store_true")
    a = ap.parse_args(argv)
    root = a.root if a.root.is_absolute() else ROOT / a.root
    out = a.out if a.out.is_absolute() else ROOT / a.out
    flatten_md_tree(root, out)
    if a.md_only:
        return 0
    subprocess.run([sys.executable, str(SCRIPTS / "build_paper_tex.py")], cwd=ROOT, check=True)
    if a.tex_only:
        return 0
    return subprocess.run(["bash", str(SCRIPTS / "build_paper_pdf.sh"), "--no-flatten"], cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
