"""Build a static, blog-style HTML page from the markdown notes in notes/.

Usage (from the repo root):
    uv run --no-project --with markdown python notes/site/build.py

Outputs:
    notes/site/index.html   full HTML document, open it directly in a browser
    <out-dir>/body.html     same page without the <html>/<head> skeleton (for publishing), if --body-out is given
"""

import argparse
import html
import re
from pathlib import Path

import markdown
from markdown.extensions.toc import slugify

ROOT = Path(__file__).resolve().parents[2]
NOTES = ROOT / "notes"
REPO = "https://github.com/jamwithai/production-agentic-rag-course"

# (id, file, short title, git tag)
WEEKS = [
    ("w2", "week2-data-ingestion.md", "Data ingestion", "week2.0"),
    ("w3", "week3-keyword-search.md", "BM25 keyword search", "week3.0"),
    ("w4", "week4-chunking-hybrid-search.md", "Chunking + hybrid search", "week4.0"),
    ("w5", "week5-complete-rag.md", "Complete RAG", "week5.0"),
    ("w6", "week6-monitoring-caching.md", "Monitoring + caching", "week6.0"),
    ("w7", "week7-agentic-rag-telegram.md", "Agentic RAG + Telegram", "week7.0"),
]
FILE_TO_ID = {f: wid for wid, f, _, _ in WEEKS}
FILE_TO_ID["README.md"] = "home"


def rewrite_links(md_text: str, page_id: str, img_prefix: str) -> str:
    """Point code links at GitHub, note links at in-page sections, images at the static folder."""

    def link(m: re.Match) -> str:
        bang, label, target = m.group(1), m.group(2), m.group(3)
        if bang:  # image
            name = target.split("/")[-1]
            return f"![{label}]({img_prefix}{name})"
        if target.startswith(("http://", "https://", "mailto:")):
            return m.group(0)
        if target.startswith("#"):
            return f"[{label}](#{page_id}-{target[1:]})"
        path, _, frag = target.partition("#")
        fname = path.split("/")[-1]
        if path.endswith(".md") and fname in FILE_TO_ID:
            dest = FILE_TO_ID[fname]
            return f"[{label}](#{dest}-{frag})" if frag else f"[{label}](#{dest})"
        if path.startswith("../"):
            rel = path[3:]
            # Line anchors like L10-L20 work on GitHub blob URLs.
            kind = "tree" if rel.endswith("/") else "blob"
            url = f"{REPO}/{kind}/main/{rel.rstrip('/')}"
            return f"[{label}]({url}#{frag})" if frag else f"[{label}]({url})"
        return m.group(0)

    md_text = re.sub(r"(!?)\[([^\]]*)\]\(([^)\s]+)\)", link, md_text)
    md_text = md_text.replace('<a id="gotchas"></a>', f'<a id="{page_id}-gotchas"></a>')
    return md_text


def render(md_text: str, page_id: str) -> tuple[str, list[tuple[int, str, str]]]:
    md = markdown.Markdown(
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
        extension_configs={"toc": {"slugify": lambda v, sep: f"{page_id}-{slugify(v, sep)}", "toc_depth": "2-2"}},
    )
    body = md.convert(md_text)
    toc = [(t["level"], t["id"], t["name"]) for t in md.toc_tokens]
    # Turn the first H1 into the article title block (handled by the template).
    body = re.sub(r"<h1[^>]*>.*?</h1>\s*", "", body, count=1, flags=re.S)
    # Wrap tables so they scroll on their own at phone width.
    body = body.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    # Render "- [ ]" checklist items as checkboxes.
    body = body.replace("<li>[ ] ", '<li class="check"><input type="checkbox" disabled> ')
    body = body.replace("<li>[x] ", '<li class="check"><input type="checkbox" checked disabled> ')
    # Style the lead summary blockquote.
    body = re.sub(r"<blockquote>\s*<p><strong>One-line summary:</strong>", '<blockquote class="lead"><p>', body, count=1)
    return body, toc


def strip_number(heading: str) -> str:
    """'4. Concepts' -> 'Concepts' (the TOC is already an ordered list)."""
    return re.sub(r"^\d+\.\s*", "", heading)


def split_title(md_text: str) -> tuple[str, str]:
    m = re.search(r"^# (.+)$", md_text, re.M)
    title = m.group(1).strip() if m else ""
    if " — " in title:
        week, sub = title.split(" — ", 1)
        return week, sub
    return "", title


def build(img_prefix: str) -> str:
    sections = []
    nav_items = ['<li><a href="#home" data-page="home">Overview</a></li>',
                 '<li><a class="ext" href="https://jamwithai.substack.com/p/the-infrastructure-that-powers-rag" '
                 'target="_blank" rel="noopener">Week 1 · Infrastructure <span aria-hidden="true">↗</span></a></li>']

    # Overview page
    readme = (NOTES / "README.md").read_text()
    readme_body, _ = render(rewrite_links(readme, "home", img_prefix), "home")
    sections.append(
        f'<article class="page" id="home" data-page="home">'
        f'<header class="post-head"><p class="eyebrow">Production Agentic RAG · course notes</p>'
        f'<h1>arXiv Paper Curator</h1>'
        f'<p class="dek">Study notes for weeks 2–7, written from the code in this repo. Pick a week on the left, '
        f'or start with the roadmap below.</p></header>'
        f'<div class="prose">{readme_body}</div></article>'
    )

    for i, (wid, fname, short, tag) in enumerate(WEEKS):
        text = (NOTES / fname).read_text()
        week_label, subtitle = split_title(text)
        body, toc = render(rewrite_links(text, wid, img_prefix), wid)
        toc_html = "".join(f'<li><a href="#{tid}">{html.escape(strip_number(name))}</a></li>' for _, tid, name in toc)
        prev_link = f'<a class="pn prev" href="#{WEEKS[i-1][0]}"><span>Previous</span>{WEEKS[i-1][2]}</a>' if i > 0 else '<a class="pn prev" href="#home"><span>Previous</span>Overview</a>'
        next_link = f'<a class="pn next" href="#{WEEKS[i+1][0]}"><span>Next</span>{WEEKS[i+1][2]}</a>' if i < len(WEEKS) - 1 else '<span></span>'
        nav_items.append(f'<li><a href="#{wid}" data-page="{wid}">{html.escape(week_label)} · {html.escape(short)}</a></li>')
        sections.append(
            f'<article class="page" id="{wid}" data-page="{wid}">'
            f'<header class="post-head"><p class="eyebrow"><span class="tag">{tag}</span>{html.escape(week_label)}</p>'
            f'<h1>{html.escape(subtitle)}</h1></header>'
            f'<details class="toc" open><summary>On this page</summary><ol>{toc_html}</ol></details>'
            f'<div class="prose">{body}</div>'
            f'<nav class="prevnext" aria-label="Week navigation">{prev_link}{next_link}</nav></article>'
        )

    return TEMPLATE.replace("{{NAV}}", "".join(nav_items)).replace("{{PAGES}}", "".join(sections))


TEMPLATE = r"""<title>arXiv Curator Notes</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600;700&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap">
<style>
/* Layout: fixed left index of weeks (like a paper's table of contents), one long-form article at a time. */
:root {
  --paper: #f6f7f5;      /* page ground, faint green-grey */
  --sheet: #ffffff;      /* code + table surfaces */
  --ink: #1d2328;        /* body text */
  --ink-soft: #56616b;   /* secondary text */
  --rule: #d9ded9;       /* hairlines */
  --accent: #a3201d;     /* arXiv crimson: links, tags, marks */
  --accent-soft: #f6e4e2;
  --code-bg: #eef1ee;
  --hl-key: #1f5f8b; --hl-str: #2f6b3a; --hl-com: #7a8288; --hl-num: #8a4b0f;
  --font-display: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif;
  --font-body: "Source Serif 4", Georgia, "Times New Roman", serif;
  --font-mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  --measure: 70ch;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper: #15191c; --sheet: #1c2226; --ink: #e3e6e3; --ink-soft: #9aa5ad; --rule: #2e363b;
    --accent: #f08a7e; --accent-soft: #3a2422; --code-bg: #1f262a;
    --hl-key: #8cc4ef; --hl-str: #9fd39a; --hl-com: #7f8a91; --hl-num: #f0b77a;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --paper: #15191c; --sheet: #1c2226; --ink: #e3e6e3; --ink-soft: #9aa5ad; --rule: #2e363b;
  --accent: #f08a7e; --accent-soft: #3a2422; --code-bg: #1f262a;
  --hl-key: #8cc4ef; --hl-str: #9fd39a; --hl-com: #7f8a91; --hl-num: #f0b77a;
  color-scheme: dark;
}

* { box-sizing: border-box; }
body { background: var(--paper); color: var(--ink); font: 400 1.0625rem/1.65 var(--font-body); margin: 0; }
a { color: var(--accent); text-underline-offset: 0.18em; text-decoration-thickness: 1px; }
a:hover { text-decoration-thickness: 2px; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 2px; }

.shell { display: grid; grid-template-columns: 17rem minmax(0, 1fr); min-height: 100vh; }
.side {
  position: sticky; top: env(safe-area-inset-top, 0px); align-self: start; height: 100vh; overflow-y: auto;
  border-right: 1px solid var(--rule); padding: 2rem 1.25rem; display: flex; flex-direction: column; gap: 1.5rem;
}
.brand { font: 700 1.15rem/1.2 var(--font-display); letter-spacing: 0.01em; color: var(--ink); text-decoration: none; }
.brand small { display: block; font: 500 0.72rem/1.4 var(--font-mono); color: var(--ink-soft); letter-spacing: 0.04em; margin-top: 0.35rem; text-transform: uppercase; }
.side ol { list-style: none; margin: 0; padding: 0; display: grid; gap: 0.15rem; }
.side a[data-page], .side a.ext {
  display: block; padding: 0.45rem 0.65rem; border-radius: 4px; color: var(--ink-soft); text-decoration: none;
  font: 500 0.95rem/1.3 var(--font-display);
}
.side a[data-page]:hover, .side a.ext:hover { color: var(--ink); background: var(--code-bg); }
.side a[aria-current="page"] { color: var(--accent); background: var(--accent-soft); }
.theme { margin-top: auto; display: flex; gap: 0.35rem; font: 500 0.75rem var(--font-mono); }
.theme button {
  font: inherit; color: var(--ink-soft); background: transparent; border: 1px solid var(--rule); border-radius: 4px;
  padding: 0.3rem 0.55rem; cursor: pointer;
}
.theme button[aria-pressed="true"] { color: var(--ink); border-color: var(--ink-soft); }

main { padding-inline: clamp(1rem, 5vw, 4.5rem); padding-block: 3rem 5rem; min-width: 0; }
.page { max-width: var(--measure); margin-inline: auto; }
.page[hidden] { display: none !important; }

.post-head { padding-bottom: 1.5rem; margin-bottom: 1.75rem; border-bottom: 1px solid var(--rule); }
.eyebrow { font: 500 0.8rem/1.4 var(--font-mono); text-transform: uppercase; letter-spacing: 0.08em; color: var(--ink-soft); margin: 0 0 0.6rem; display: flex; gap: 0.6rem; align-items: center; flex-wrap: wrap; }
.tag { color: var(--accent); border: 1px solid currentColor; border-radius: 3px; padding: 0.05rem 0.4rem; text-transform: none; letter-spacing: 0.02em; }
.post-head h1 { font: 700 clamp(2rem, 4.5vw, 2.9rem)/1.08 var(--font-display); margin: 0; text-wrap: balance; letter-spacing: -0.005em; }
.dek { font-size: 1.15rem; color: var(--ink-soft); margin: 0.9rem 0 0; }

.toc { margin: 0 0 2rem; font: 500 0.92rem/1.4 var(--font-display); }
.toc summary { cursor: pointer; font: 500 0.75rem var(--font-mono); text-transform: uppercase; letter-spacing: 0.08em; color: var(--ink-soft); }
.toc ol { margin: 0.75rem 0 0; padding-left: 1.4rem; columns: 2 16rem; column-gap: 2rem; }
.toc li { break-inside: avoid; padding: 0.12rem 0; }
.toc a { color: var(--ink); text-decoration: none; }
.toc a:hover { color: var(--accent); }

.prose > * + * { margin-top: 1.1em; }
.prose h2 { font: 600 1.6rem/1.2 var(--font-display); margin-top: 2.6em; padding-top: 0.4em; text-wrap: balance; scroll-margin-top: 1rem; }
.prose h3 { font: 600 1.2rem/1.3 var(--font-display); margin-top: 2em; scroll-margin-top: 1rem; }
.prose hr { border: 0; border-top: 1px solid var(--rule); margin: 2.5rem 0; }
.prose p, .prose li { max-width: var(--measure); }
.prose ul, .prose ol { padding-left: 1.4rem; }
.prose li + li { margin-top: 0.3em; }
.prose strong { font-weight: 600; }
.prose img { display: block; max-width: 100%; height: auto; border: 1px solid var(--rule); border-radius: 4px; background: #fff; margin-inline: auto; }

.prose blockquote { margin: 1.5rem 0; padding: 0.9rem 1.1rem; border-left: 3px solid var(--accent); background: var(--accent-soft); border-radius: 0 4px 4px 0; }
.prose blockquote p { margin: 0; }
.prose blockquote p + p { margin-top: 0.6em; }
.prose blockquote.lead { border: 0; background: none; padding: 0; font-size: 1.22rem; line-height: 1.5; color: var(--ink); }

code { font: 400 0.86em/1.5 var(--font-mono); background: var(--code-bg); padding: 0.1em 0.32em; border-radius: 3px; overflow-wrap: anywhere; }
pre { background: var(--sheet); border: 1px solid var(--rule); border-radius: 4px; padding: 1rem 1.1rem; overflow-x: auto; line-height: 1.5; font-size: 0.95rem; }
pre code { background: none; padding: 0; font-size: 0.84rem; overflow-wrap: normal; white-space: pre; }
.hljs-keyword, .hljs-built_in, .hljs-literal, .hljs-attr { color: var(--hl-key); }
.hljs-string, .hljs-title { color: var(--hl-str); }
.hljs-comment, .hljs-meta { color: var(--hl-com); font-style: italic; }
.hljs-number { color: var(--hl-num); }

.table-wrap { overflow-x: auto; border: 1px solid var(--rule); border-radius: 4px; background: var(--sheet); }
table { border-collapse: collapse; width: 100%; font: 400 0.92rem/1.45 var(--font-body); font-variant-numeric: tabular-nums; }
th { font: 600 0.82rem/1.3 var(--font-display); text-transform: uppercase; letter-spacing: 0.04em; color: var(--ink-soft); text-align: left; background: var(--code-bg); }
th, td { padding: 0.55rem 0.8rem; border-bottom: 1px solid var(--rule); vertical-align: top; }
tr:last-child td { border-bottom: 0; }
td code { font-size: 0.8rem; }

.prose input[type="checkbox"] { accent-color: var(--accent); }

.prevnext { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 4rem; padding-top: 1.5rem; border-top: 1px solid var(--rule); }
.pn { text-decoration: none; color: var(--ink); font: 600 1.05rem/1.3 var(--font-display); padding: 0.8rem 1rem; border: 1px solid var(--rule); border-radius: 4px; }
.pn span { display: block; font: 500 0.72rem var(--font-mono); text-transform: uppercase; letter-spacing: 0.08em; color: var(--ink-soft); margin-bottom: 0.2rem; }
.pn:hover { border-color: var(--accent); }
.pn.next { text-align: right; grid-column: 2; }

@media (max-width: 860px) {
  .shell { grid-template-columns: minmax(0, 1fr); }
  .side { position: static; height: auto; border-right: 0; border-bottom: 1px solid var(--rule); padding: 1.25rem 1rem; gap: 0.9rem; }
  .side ol { display: flex; flex-wrap: wrap; gap: 0.35rem; }
  .side a[data-page], .side a.ext { border: 1px solid var(--rule); padding: 0.35rem 0.6rem; font-size: 0.88rem; }
  .theme { margin-top: 0; }
  main { padding-block: 2rem 4rem; }
  .prevnext { grid-template-columns: 1fr; }
  .pn.next { grid-column: 1; }
}
@media (prefers-reduced-motion: no-preference) { html { scroll-behavior: smooth; } }
</style>

<div class="shell">
  <aside class="side">
    <a class="brand" href="#home">arXiv Paper Curator<small>Production agentic RAG · notes</small></a>
    <nav aria-label="Weeks"><ol>{{NAV}}</ol></nav>
    <div class="theme" role="group" aria-label="Theme">
      <button type="button" id="theme-system" data-theme-choice="system" aria-pressed="true">System</button>
      <button type="button" id="theme-light" data-theme-choice="light" aria-pressed="false">Light</button>
      <button type="button" id="theme-dark" data-theme-choice="dark" aria-pressed="false">Dark</button>
    </div>
  </aside>
  <main>{{PAGES}}</main>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script>
(function () {
  var pages = Array.prototype.slice.call(document.querySelectorAll("article.page"));
  var links = Array.prototype.slice.call(document.querySelectorAll(".side a[data-page]"));
  var ids = pages.map(function (p) { return p.id; });

  function show(hash) {
    var target = (hash || "").replace(/^#/, "");
    var pageId = ids.indexOf(target) >= 0 ? target : target.split("-")[0];
    if (ids.indexOf(pageId) < 0) pageId = "home";
    pages.forEach(function (p) { p.hidden = p.id !== pageId; });
    links.forEach(function (a) {
      if (a.dataset.page === pageId) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
    });
    var el = target && target !== pageId ? document.getElementById(target) : null;
    if (el) el.scrollIntoView(); else window.scrollTo(0, 0);
  }
  window.addEventListener("hashchange", function () { show(location.hash); });
  show(location.hash);

  // Theme toggle (remembered per browser when storage is available)
  var buttons = Array.prototype.slice.call(document.querySelectorAll("[data-theme-choice]"));
  function applyTheme(choice) {
    if (choice === "light" || choice === "dark") document.documentElement.setAttribute("data-theme", choice);
    else document.documentElement.removeAttribute("data-theme");
    buttons.forEach(function (b) { b.setAttribute("aria-pressed", String(b.dataset.themeChoice === choice)); });
  }
  var saved = null;
  try { saved = localStorage.getItem("notes-theme"); } catch (e) {}
  if (saved) applyTheme(saved);
  buttons.forEach(function (b) {
    b.addEventListener("click", function () {
      applyTheme(b.dataset.themeChoice);
      try { localStorage.setItem("notes-theme", b.dataset.themeChoice); } catch (e) {}
    });
  });

  if (window.hljs) {
    document.querySelectorAll("pre code").forEach(function (block) {
      if (!/language-/.test(block.className)) block.classList.add("nohighlight");
      else window.hljs.highlightElement(block);
    });
  }
})();
</script>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-out", type=Path, help="Also write the page without the document skeleton to this file")
    args = parser.parse_args()

    local = build(img_prefix="../../static/")
    out = NOTES / "site" / "index.html"
    out.write_text(
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        + local.replace("<div class=\"shell\">", "</head>\n<body>\n<div class=\"shell\">", 1)
        + "\n</body>\n</html>\n"
    )
    print(f"wrote {out}")

    if args.body_out:
        args.body_out.parent.mkdir(parents=True, exist_ok=True)
        args.body_out.write_text(build(img_prefix="static/"))
        print(f"wrote {args.body_out}")


if __name__ == "__main__":
    main()
