#!/usr/bin/env python3
r"""Convert paper/arxiv.md (flattened note tree) to paper/arxiv.tex.

Adapted from catskillsresearch/lyasoff2016 ``scripts/build_arxiv_tex.py`` with the Lean listing,
mermaid, gist-box and appendix machinery removed. Steps:

1. Drop the YAML front matter and HTML comments; take the ``#`` title line as the PDF title.
2. Lift ``## Abstract`` into ``\begin{abstract}``.
3. Strip manual section numbers so LaTeX numbers everything once.
4. Rewrite pandoc citations ``[@a; @b]`` to ``\cite{a,b}`` (BibTeX, natbib numbers).
5. pandoc -> LaTeX with ``--shift-heading-level-by=-1`` (``##`` -> ``\section``).
6. Clean-up: figures placed [H] and width-capped, PNG -> vector PDF when one exists, labelled
   display maths -> ``equation``, References section unnumbered.
7. Splice preamble + title page (title, author, affiliation, date, abstract, TOC) + body + bibliography.
"""
from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
SRC = PAPER / "arxiv.md"
OUT = PAPER / "arxiv.tex"
PREAMBLE = ROOT / "scripts" / "tex_preamble_paper.tex"

AUTHOR = "Lars Warren Ericson"
COMPANY = "Catskills Research Company"
GITHUB_URL = "https://github.com/catskillsresearch/mast-flux-imputation"
EMAIL = "lars.ericson@catskillsresearch.com"
ORCID = "0000-0001-8299-9361"

HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
FRONT_MATTER = re.compile(r"\A\s*---\n.*?\n---\n", re.DOTALL)
MANUAL_SECTION_NUM = re.compile(r"^(#{1,6})[ \t]+(?:§\s*)?\d+(?:\.\d+)*\.?[ \t]+", re.MULTILINE)
_PANDOC_CITE_RE = re.compile(r"\[(@[A-Za-z][\w\-:]*(?:\s*,\s*[^;\]]+)?(?:\s*;\s*@[A-Za-z][\w\-:]*(?:\s*,\s*[^;\]]+)?)*)\]")


def pandoc_cites_to_latex_cite(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        parts = [p.strip() for p in m.group(1).split(";") if p.strip()]
        keys, opt = [], None
        for i, part in enumerate(parts):
            if not part.startswith("@"):
                return m.group(0)
            key, _, loc = part[1:].partition(",")
            keys.append(key.strip())
            if i == 0 and loc.strip():
                opt = re.sub(r"\bpp?\.\s*", lambda x: x.group(0).strip() + "~", loc.strip())
        return f"\\cite[{opt}]{{{','.join(keys)}}}" if opt else f"\\cite{{{','.join(keys)}}}"

    return _PANDOC_CITE_RE.sub(repl, text)


def extract_title(text: str) -> tuple[str, str]:
    m = re.search(r"^#\s+(.+)\n", text, re.MULTILINE)
    if not m:
        return "Untitled", text
    return m.group(1).strip(), text[: m.start()] + text[m.end():]


def extract_abstract(text: str) -> tuple[str, str]:
    m = re.search(r"^##\s+Abstract\s*\n(.*?)(?=^##\s)", text, re.DOTALL | re.MULTILINE)
    if not m:
        return "", text
    return m.group(1).strip(), text[: m.start()] + text[m.end():]


def pandoc_to_latex(markdown: str, shift: bool = True) -> str:
    cmd = ["pandoc", "-f", "markdown+tex_math_dollars+raw_tex+smart+implicit_figures", "-t", "latex", "--wrap=none"]
    if shift:
        cmd += ["--shift-heading-level-by=-1"]
    proc = subprocess.run(cmd, input=markdown, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError("pandoc failed")
    return proc.stdout


def labelled_display_math(latex: str) -> str:
    r"""``\[ ... \label{eq:x} ... \]`` -> numbered ``equation`` (or ``align`` if it has ``\\``)."""
    def repl(m: re.Match[str]) -> str:
        body = m.group(1)
        if "\\label{" not in body:
            return m.group(0)
        env = "align" if "\\\\" in body and "aligned" not in body else "equation"
        return f"\\begin{{{env}}}{body}\\end{{{env}}}"

    return re.sub(r"\\\[(.*?)\\\]", repl, latex, flags=re.DOTALL)


def cleanup_pandoc_latex(latex: str) -> str:
    latex = latex.replace("\\pandocbounded{", "{")
    latex = re.sub(r"\\tightlist\n?", "", latex)
    latex = re.sub(r"\\begin\{figure\}(\[[^\]]*\])?", r"\\begin{figure}[H]", latex)

    def img(m: re.Match[str]) -> str:
        path = m.group(1)
        pdf = re.sub(r"\.png$", ".pdf", path)
        if pdf != path and (PAPER / pdf).is_file():
            path = pdf
        return f"\\includegraphics[max width=0.95\\linewidth,max height=0.62\\textheight,keepaspectratio]{{{path}}}"

    latex = re.sub(r"\\includegraphics(?:\[[^\]]*\])?\{(figures/[^}]+)\}", img, latex)
    for cmd in ("section", "subsection", "subsubsection", "paragraph"):
        latex = re.sub(rf"(\\{cmd}\{{)\d+(?:\.\d+)*\.?\s+", r"\1", latex)
    latex = labelled_display_math(latex)
    latex = re.sub(r"\\section\{References\}\\label\{references\}",
                   r"\\section*{References}\\addcontentsline{toc}{section}{References}", latex)
    latex = re.sub(r"\\section\{References\}",
                   r"\\section*{References}\\addcontentsline{toc}{section}{References}", latex)
    latex = breakable_paths(latex)
    return re.sub(r"\n{3,}", "\n\n", latex)


_PATH_TT = re.compile(r"\\texttt\{((?:[A-Za-z0-9./:\-]|\\_)+)\}")


def breakable_paths(latex: str) -> str:
    """Render file paths / dotted module names as \\nolinkurl so they can break
    at / and . (body text only; headings and captions are left alone)."""

    def repl(m: re.Match[str]) -> str:
        body = m.group(1)
        if "/" not in body and body.count(".") < 1:
            return m.group(0)
        return "\\nolinkurl{" + body.replace("\\_", "_") + "}"

    out = []
    for line in latex.split("\n"):
        if re.match(r"\\(sub)*section|\\caption|\\label|\\affil", line) or "\\caption{" in line:
            out.append(line)
        else:
            out.append(_PATH_TT.sub(repl, line))
    return "\n".join(out)


def build_title_page(title: str, abstract_latex: str) -> str:
    return textwrap.dedent(f"""
        \\title{{\\textbf{{{title}}}}}
        \\author[1]{{\\textbf{{{AUTHOR}}}}}
        \\affil[1]{{{COMPANY}}}
        \\affil[1]{{\\url{{{GITHUB_URL}}}}}
        \\affil[1]{{\\texttt{{{EMAIL}}}}}
        \\date{{\\today}}

        \\begin{{document}}
        \\maketitle
        \\begin{{center}}\\small\\textbf{{ORCID:}} {ORCID}\\end{{center}}
        \\begin{{abstract}}
        {abstract_latex.strip()}
        \\end{{abstract}}
        \\clearpage
        \\tableofcontents
        \\clearpage
        """).strip()


def main() -> int:
    if not SRC.is_file():
        print(f"error: missing {SRC}; run scripts/compile_md_tree.py --md-only first", file=sys.stderr)
        return 1
    raw = SRC.read_text(encoding="utf-8")
    body = FRONT_MATTER.sub("", HTML_COMMENT.sub("", raw)).lstrip()
    body = HTML_COMMENT.sub("", body)
    title, body = extract_title(body)
    abstract_md, body = extract_abstract(body)
    body = MANUAL_SECTION_NUM.sub(r"\1 ", body)
    body = pandoc_cites_to_latex_cite(body)
    abstract_md = pandoc_cites_to_latex_cite(abstract_md)
    latex_body = cleanup_pandoc_latex(pandoc_to_latex(body, shift=True))
    if "References}" not in latex_body:
        latex_body += "\n\\section*{References}\\addcontentsline{toc}{section}{References}\n"
    latex_body += "\n\\bibliographystyle{plainnat}\n\\bibliography{references}\n"
    latex_body = re.sub(r"(?<!\\clearpage\n)(\\section\{)", r"\\clearpage\n\1", latex_body)
    abstract_latex = cleanup_pandoc_latex(pandoc_to_latex(abstract_md, shift=False)) if abstract_md else ""
    preamble = PREAMBLE.read_text(encoding="utf-8")
    preamble += "\n\\hypersetup{pdftitle={" + title + "},pdfauthor={" + AUTHOR + "}}\n"
    doc = preamble + "\n\n" + build_title_page(title, abstract_latex) + "\n\n" + latex_body + "\n\\end{document}\n"
    OUT.write_text(doc, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
