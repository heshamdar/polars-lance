"""Render a `SuiteRun` as a single self-contained HTML page.

No external assets and no build step -- the file is openable from a CI artifact, a scratch
directory, or an email attachment. The layout answers, in order: did it pass, where did it fail,
what exactly failed, and how do I reproduce it.

The reproduction spec on each failure is the part worth the effort. A conformance failure that
cannot be re-run in ten lines gets triaged into a backlog and forgotten.
"""

from __future__ import annotations

import html as _html
import json
from collections import Counter, defaultdict

from plioc.run import CONTRACTS, Check, HarnessRun, Status, SuiteRun

_MARK = {
    Status.PASS: ("pass", "pass"),
    Status.XFAIL: ("xfail", "declared"),
    Status.XPASS: ("xpass", "stale"),
    Status.SKIP: ("skip", "skip"),
    Status.FAIL: ("fail", "fail"),
}

_CSS = """
:root {
  --bg: #ffffff; --fg: #16181d; --muted: #5c6370; --line: #e3e6ea; --panel: #f7f8fa;
  --pass: #157f4b; --pass-bg: #e6f4ec; --fail: #b3261e; --fail-bg: #fdeceb;
  --xfail: #7a5a00; --xfail-bg: #fdf4dd; --skip: #6b7280; --skip-bg: #f1f2f4;
  --accent: #2f5fd0;
  color-scheme: light dark;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14161a; --fg: #e7e9ec; --muted: #9aa1ab; --line: #2a2e35; --panel: #1b1e24;
    --pass: #58c98c; --pass-bg: #16301f; --fail: #ff8b80; --fail-bg: #331a18;
    --xfail: #e3bd57; --xfail-bg: #322a12; --skip: #8b919b; --skip-bg: #22262c;
    --accent: #7fa4f5;
  }
}
:root[data-theme="dark"] {
  --bg: #14161a; --fg: #e7e9ec; --muted: #9aa1ab; --line: #2a2e35; --panel: #1b1e24;
  --pass: #58c98c; --pass-bg: #16301f; --fail: #ff8b80; --fail-bg: #331a18;
  --xfail: #e3bd57; --xfail-bg: #322a12; --skip: #8b919b; --skip-bg: #22262c;
  --accent: #7fa4f5;
}
:root[data-theme="light"] {
  --bg: #ffffff; --fg: #16181d; --muted: #5c6370; --line: #e3e6ea; --panel: #f7f8fa;
  --pass: #157f4b; --pass-bg: #e6f4ec; --fail: #b3261e; --fail-bg: #fdeceb;
  --xfail: #7a5a00; --xfail-bg: #fdf4dd; --skip: #6b7280; --skip-bg: #f1f2f4;
  --accent: #2f5fd0;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
main { max-width: 1180px; margin: 0 auto; padding: 2.5rem 1.25rem 5rem; }
h1 { font-size: 1.6rem; margin: 0 0 .35rem; letter-spacing: -.01em; }
h2 { font-size: 1.15rem; margin: 2.75rem 0 .4rem; letter-spacing: -.01em; }
h3 { font-size: .95rem; margin: 1.5rem 0 .5rem; }
p.lede { color: var(--muted); margin: 0 0 1.75rem; max-width: 70ch; }
p.note { color: var(--muted); margin: .25rem 0 1rem; max-width: 80ch; font-size: .875rem; }
code, pre, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
a { color: var(--accent); }

.meta { display: flex; flex-wrap: wrap; gap: .4rem 1.5rem; font-size: .82rem; color: var(--muted);
        border-top: 1px solid var(--line); border-bottom: 1px solid var(--line);
        padding: .7rem 0; margin-bottom: 2rem; }
.meta b { color: var(--fg); font-weight: 600; }

.verdict { display: inline-flex; align-items: center; gap: .5rem; font-weight: 650;
           padding: .3rem .7rem; border-radius: 999px; font-size: .8rem; letter-spacing: .02em;
           text-transform: uppercase; }
.verdict.pass { background: var(--pass-bg); color: var(--pass); }
.verdict.fail { background: var(--fail-bg); color: var(--fail); }

.tiles { display: grid; gap: .75rem; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }
.tile { border: 1px solid var(--line); border-radius: 10px; padding: .85rem 1rem; background: var(--panel); }
.tile .n { font-size: 1.75rem; font-weight: 650; line-height: 1.1; font-variant-numeric: tabular-nums; }
.tile .k { font-size: .78rem; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
.tile.pass .n { color: var(--pass); }
.tile.fail .n { color: var(--fail); }
.tile.xfail .n { color: var(--xfail); }
.tile.skip .n { color: var(--skip); }

.scroll { overflow-x: auto; border: 1px solid var(--line); border-radius: 10px; }
table { border-collapse: collapse; width: 100%; font-size: .86rem; }
th, td { text-align: left; padding: .45rem .7rem; border-bottom: 1px solid var(--line); white-space: nowrap; }
th { background: var(--panel); font-weight: 600; font-size: .78rem; text-transform: uppercase;
     letter-spacing: .04em; color: var(--muted); position: sticky; top: 0; }
tbody tr:last-child td { border-bottom: none; }
td.subject { white-space: normal; word-break: break-word; font-family: ui-monospace, Menlo, monospace; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }

.pill { display: inline-block; min-width: 3.7rem; text-align: center; padding: .1rem .45rem;
        border-radius: 5px; font-size: .74rem; font-weight: 650; letter-spacing: .02em; }
.pill.pass { background: var(--pass-bg); color: var(--pass); }
.pill.fail { background: var(--fail-bg); color: var(--fail); }
.pill.xpass { background: var(--fail-bg); color: var(--fail); }
.pill.xfail { background: var(--xfail-bg); color: var(--xfail); }
.pill.skip { background: var(--skip-bg); color: var(--skip); }
.pill.none { color: var(--muted); opacity: .45; }

.bar { display: flex; height: 8px; border-radius: 4px; overflow: hidden; background: var(--skip-bg);
       min-width: 120px; }
.bar span { display: block; }
.bar .b-pass { background: var(--pass); }
.bar .b-fail { background: var(--fail); }
.bar .b-xfail { background: var(--xfail); }
.bar .b-skip { background: var(--skip); opacity: .35; }

details { border: 1px solid var(--line); border-radius: 10px; margin: .6rem 0; background: var(--panel); }
details[open] { background: transparent; }
summary { cursor: pointer; padding: .65rem .9rem; font-weight: 600; font-size: .9rem; }
summary::marker { color: var(--muted); }
details > div.body { padding: 0 .9rem .9rem; }
pre { background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
      padding: .7rem .85rem; overflow-x: auto; font-size: .8rem; margin: .5rem 0 0; }
.detail { color: var(--fail); font-size: .84rem; margin: .1rem 0 .5rem; word-break: break-word; }
.detail.declared { color: var(--xfail); }

.legend { display: flex; flex-wrap: wrap; gap: .5rem 1rem; font-size: .8rem; color: var(--muted);
          margin: .5rem 0 1.25rem; }
.controls { display: flex; gap: .5rem; align-items: center; margin: .6rem 0 1rem; flex-wrap: wrap; }
.controls input { font: inherit; font-size: .85rem; padding: .35rem .6rem; border-radius: 7px;
                  border: 1px solid var(--line); background: var(--bg); color: var(--fg); min-width: 16rem; }
.controls button { font: inherit; font-size: .8rem; padding: .35rem .7rem; border-radius: 7px;
                   border: 1px solid var(--line); background: var(--panel); color: var(--fg); cursor: pointer; }
.controls button[aria-pressed="true"] { border-color: var(--accent); color: var(--accent); }
tr.hidden { display: none; }
.empty { color: var(--muted); font-size: .875rem; padding: .5rem 0; }
"""

_JS = """
(function () {
  var q = document.getElementById('filter');
  var buttons = Array.prototype.slice.call(document.querySelectorAll('[data-status]'));
  var rows = Array.prototype.slice.call(document.querySelectorAll('tr[data-subject]'));
  var active = null;
  function apply() {
    var needle = (q && q.value || '').toLowerCase();
    rows.forEach(function (row) {
      var matchText = !needle || row.getAttribute('data-subject').indexOf(needle) !== -1;
      var matchStat = !active || row.getAttribute('data-statuses').indexOf(active) !== -1;
      row.classList.toggle('hidden', !(matchText && matchStat));
    });
  }
  if (q) { q.addEventListener('input', apply); }
  buttons.forEach(function (b) {
    b.addEventListener('click', function () {
      var want = b.getAttribute('data-status');
      active = active === want ? null : want;
      buttons.forEach(function (o) {
        o.setAttribute('aria-pressed', String(o.getAttribute('data-status') === active));
      });
      apply();
    });
  });
})();
"""


def esc(text: object) -> str:
    return _html.escape(str(text), quote=True)


def _pill(status: Status | None) -> str:
    if status is None:
        return '<span class="pill none">&middot;</span>'
    cls, label = _MARK[status]
    return f'<span class="pill {cls}">{label}</span>'


def _bar(counts: Counter[Status]) -> str:
    total = sum(counts.values()) or 1
    order = [
        (Status.PASS, "b-pass"),
        (Status.XFAIL, "b-xfail"),
        (Status.FAIL, "b-fail"),
        (Status.XPASS, "b-fail"),
        (Status.SKIP, "b-skip"),
    ]
    parts = [
        f'<span class="{cls}" style="width:{counts[s] / total * 100:.4f}%"></span>'
        for s, cls in order
        if counts[s]
    ]
    return f'<div class="bar">{"".join(parts)}</div>'


def render(run: SuiteRun, *, title: str = "IO plugin conformance report") -> str:
    subject_of = _primary(run)
    body = [
        _header(run, subject_of, title),
        _summary(run, subject_of),
        _matrix(run),
        _per_contract(run, subject_of),
        _failures(run),
        _declarations(run),
        _corpus(run),
    ]
    return (
        f"<style>{_CSS}</style>\n<main>\n"
        + "\n".join(body)
        + f"\n</main>\n<script>{_JS}</script>\n"
    )


def _primary(run: SuiteRun) -> HarnessRun:
    """The harness the report is *about*.

    The last one given, on the convention that the reference harnesses come first and the plugin
    under test is appended. A report whose headline numbers describe Parquet would be useless.
    """
    return run.runs[-1] if run.runs else HarnessRun(name="none", capabilities=None)  # type: ignore[arg-type]


def _header(run: SuiteRun, subject: HarnessRun, title: str) -> str:
    verdict = "fail" if subject.failures else "pass"
    if subject.failures:
        word = f"{len(subject.failures)} undeclared failures"
    elif subject.count(Status.XFAIL):
        # "Conformant" alone would read as "no limits", which is rarely true and never what the
        # declaration says.
        word = f"conformant &mdash; {subject.count(Status.XFAIL)} declared limits"
    else:
        word = "fully conformant"
    meta = [
        ("harness", subject.name),
        ("polars", run.polars_version),
        ("python", run.python_version),
        ("cases", f"{run.case_count:,}"),
        ("queries", f"{run.query_count:,}"),
        ("rows generated", f"{run.row_count:,}"),
        ("elapsed", f"{run.duration_s:.1f}s"),
        ("run", run.started),
    ]
    return (
        f"<h1>{esc(title)}</h1>\n"
        '<p class="lede">Generated by <code>plioc</code>. The corpus is a pure function of '
        "<code>(spec, seed)</code> &mdash; nothing here was read from a fixture file, and every "
        "case below can be rebuilt from the spec it carries.</p>\n"
        f'<p><span class="verdict {verdict}">{esc(subject.name)}: {word}</span></p>\n'
        '<div class="meta">'
        + "".join(f"<div>{esc(k)} <b>{esc(v)}</b></div>" for k, v in meta)
        + "</div>"
    )


def _summary(run: SuiteRun, subject: HarnessRun) -> str:
    counts = Counter(c.status for c in subject.checks)
    tiles = [
        ("checks", sum(counts.values()), ""),
        ("passed", counts[Status.PASS], "pass"),
        ("failed", counts[Status.FAIL], "fail"),
        ("stale declarations", counts[Status.XPASS], "fail"),
        ("declared failures", counts[Status.XFAIL], "xfail"),
        ("skipped", counts[Status.SKIP], "skip"),
    ]
    cards = "".join(
        f'<div class="tile {cls}"><div class="n">{n:,}</div><div class="k">{esc(k)}</div></div>'
        for k, n, cls in tiles
    )
    return (
        f"<h2>Summary &mdash; {esc(subject.name)}</h2>\n"
        f'<div class="tiles">{cards}</div>\n'
        '<p class="note"><b>Declared failures</b> are limits the harness states in its '
        "<code>Capabilities</code>; they are expected and do not fail the run. A "
        "<b>stale declaration</b> is one that now passes &mdash; treated as a failure, because a "
        "capability matrix that is quietly pessimistic is worse than none.</p>"
    )


def _matrix(run: SuiteRun) -> str:
    """Per-harness totals side by side. The comparison is the point: a case that Parquet and IPC
    also fail is a format-inherent limit, not the plugin's bug."""
    rows = []
    for h in run.runs:
        counts = Counter(c.status for c in h.checks)
        rows.append(
            "<tr>"
            f'<td class="subject">{esc(h.name)}</td>'
            f"<td>{_bar(counts)}</td>"
            f'<td class="num">{counts[Status.PASS]:,}</td>'
            f'<td class="num">{counts[Status.FAIL]:,}</td>'
            f'<td class="num">{counts[Status.XPASS]:,}</td>'
            f'<td class="num">{counts[Status.XFAIL]:,}</td>'
            f'<td class="num">{counts[Status.SKIP]:,}</td>'
            f"<td>{_strictness(h)}</td>"
            f"<td>{esc(', '.join(sorted(h.capabilities.pushdown))) or '&mdash;'}</td>"
            "</tr>"
        )
    return (
        "<h2>Harnesses</h2>\n"
        '<p class="note">The reference harnesses run the identical corpus. <code>memory</code> is '
        "the identity and must pass everything; <code>parquet</code> and <code>ipc</code> are real "
        "formats, so a case they also fail is a format-inherent limit rather than a plugin bug.</p>\n"
        '<div class="scroll"><table><thead><tr>'
        "<th>harness</th><th>outcome</th><th>pass</th><th>fail</th><th>stale</th>"
        "<th>declared</th><th>skip</th><th>strictness</th><th>pushdown claimed</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _strictness(h: HarnessRun) -> str:
    caps = h.capabilities
    return esc(caps.strictness.name.lower()) if caps is not None else "&mdash;"


def _per_contract(run: SuiteRun, subject: HarnessRun) -> str:
    """One table per contract, every harness as a column, one row per subject."""
    out = [
        "<h2>Checks</h2>",
        '<div class="legend">'
        + "".join(
            f"<span>{_pill(s)} {esc(_MARK[s][1])}</span>"
            for s in (Status.PASS, Status.FAIL, Status.XPASS, Status.XFAIL, Status.SKIP)
        )
        + "</div>",
        '<div class="controls">'
        '<input id="filter" type="search" placeholder="filter by case or query id…" '
        'aria-label="filter rows">'
        + "".join(
            f'<button type="button" data-status="{_MARK[s][0]}" aria-pressed="false">'
            f"only {esc(_MARK[s][1])}</button>"
            for s in (Status.FAIL, Status.XPASS, Status.XFAIL, Status.SKIP)
        )
        + "</div>",
    ]
    names = [h.name for h in run.runs]
    for contract, question in CONTRACTS.items():
        index: dict[str, dict[str, Check]] = defaultdict(dict)
        for h in run.runs:
            for c in h.checks:
                if c.contract == contract:
                    index[c.subject][h.name] = c
        if not index:
            continue
        subjects = sorted(index, key=lambda s: (_rank(index[s], subject.name), s))
        rows = []
        for s in subjects:
            cells = index[s]
            statuses = " ".join(sorted({_MARK[c.status][0] for c in cells.values()}))
            own = cells.get(subject.name)
            detail = own.detail if own is not None and own.detail else ""
            rows.append(
                f'<tr data-subject="{esc(s.lower())}" data-statuses="{esc(statuses)}">'
                f'<td class="subject">{esc(s)}</td>'
                + "".join(
                    f"<td>{_pill(cells[n].status if n in cells else None)}</td>" for n in names
                )
                + f'<td class="subject" style="color:var(--muted);font-size:.8rem">{esc(detail[:180])}</td>'
                "</tr>"
            )
        out.append(
            f"<h3>{esc(contract)}</h3>"
            f'<p class="note">{esc(question)}</p>'
            '<div class="scroll"><table><thead><tr><th>subject</th>'
            + "".join(f"<th>{esc(n)}</th>" for n in names)
            + "<th>note</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div>"
        )
    return "\n".join(out)


def _rank(cells: dict[str, Check], subject: str) -> int:
    """Sort failures to the top of each table, then stale, then declared, then the rest."""
    own = cells.get(subject)
    if own is None:
        return 5
    return {Status.FAIL: 0, Status.XPASS: 1, Status.XFAIL: 2, Status.SKIP: 4}.get(own.status, 3)


def _failures(run: SuiteRun) -> str:
    blocks = []
    for h in run.runs:
        bad = h.failures
        if not bad:
            continue
        items = []
        for c in sorted(bad, key=lambda c: (c.contract, c.subject)):
            repro = h.repros.get(c.subject)
            body = [f'<div class="detail">{esc(c.detail)}</div>']
            if repro:
                body.append(
                    '<p class="note">Reproduce by loading this spec with '
                    "<code>plioc.codec.loads</code> &mdash; it is the whole input, "
                    "there is no data file.</p>"
                    f"<pre>{esc(_trim(repro))}</pre>"
                )
            items.append(
                "<details><summary>"
                f'{_pill(c.status)} <span class="mono">{esc(c.subject)}</span> '
                f'<span style="color:var(--muted);font-weight:400">&mdash; {esc(c.contract)}</span>'
                "</summary>"
                f'<div class="body">{"".join(body)}</div></details>'
            )
        blocks.append(f"<h3>{esc(h.name)}</h3>" + "".join(items))
    if not blocks:
        return (
            "<h2>Failures</h2>"
            '<p class="empty">None. Every check either passed or was a declared limit.</p>'
        )
    return "<h2>Failures</h2>" + "".join(blocks)


def _trim(text: str, limit: int = 2600) -> str:
    return text if len(text) <= limit else text[:limit] + "\n… truncated"


def _declarations(run: SuiteRun) -> str:
    blocks = []
    for h in run.runs:
        caps = h.capabilities
        if caps is None or not caps.known_failures:
            continue
        rows = "".join(
            f'<tr><td class="subject">{esc(k)}</td><td class="subject">{esc(v)}</td></tr>'
            for k, v in sorted(caps.known_failures.items())
        )
        blocks.append(
            f"<h3>{esc(h.name)}</h3>"
            '<div class="scroll"><table><thead><tr><th>subject</th><th>declared reason</th>'
            f"</tr></thead><tbody>{rows}</tbody></table></div>"
        )
    if not blocks:
        return ""
    return (
        "<h2>Declared limits</h2>"
        '<p class="note">Each of these is a strict expectation of failure: when one is fixed '
        "upstream, the run reports it as a <b>stale declaration</b> and the entry has to be "
        "removed. That is what stops the list outliving the bugs.</p>" + "".join(blocks)
    )


def _corpus(run: SuiteRun) -> str:
    subject = _primary(run)
    per_tag: dict[str, Counter[Status]] = defaultdict(Counter)
    for c in subject.checks:
        for tag in c.tags or {"untagged"}:
            per_tag[tag][c.status] += 1
    rows = "".join(
        f'<tr><td class="subject">{esc(tag)}</td><td>{_bar(counts)}</td>'
        f'<td class="num">{sum(counts.values()):,}</td>'
        f'<td class="num">{counts[Status.PASS]:,}</td>'
        f'<td class="num">{counts[Status.FAIL] + counts[Status.XPASS]:,}</td>'
        f'<td class="num">{counts[Status.XFAIL]:,}</td>'
        f'<td class="num">{counts[Status.SKIP]:,}</td></tr>'
        for tag, counts in sorted(per_tag.items())
    )
    return (
        f"<h2>By axis &mdash; {esc(subject.name)}</h2>"
        '<p class="note">Each case sits on exactly one axis so that a failure localises to a '
        "dtype, a null pattern, or a shape rather than to &ldquo;the corpus&rdquo;.</p>"
        '<div class="scroll"><table><thead><tr><th>tag</th><th>outcome</th><th>checks</th>'
        "<th>pass</th><th>fail</th><th>declared</th><th>skip</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></div>"
    )


def to_json(run: SuiteRun) -> str:
    """The same run as data, for diffing two commits or feeding a dashboard."""
    return json.dumps(
        {
            "started": run.started,
            "duration_s": round(run.duration_s, 3),
            "polars": run.polars_version,
            "python": run.python_version,
            "cases": run.case_count,
            "queries": run.query_count,
            "harnesses": [
                {
                    "name": h.name,
                    "strictness": h.capabilities.strictness.name,
                    "pushdown": sorted(h.capabilities.pushdown),
                    "totals": {s.value: h.count(s) for s in Status},
                    "checks": [
                        {
                            "contract": c.contract,
                            "subject": c.subject,
                            "status": c.status.value,
                            "detail": c.detail,
                        }
                        for c in h.checks
                    ],
                }
                for h in run.runs
            ],
        },
        indent=2,
    )


def document(run: SuiteRun, *, title: str = "IO plugin conformance report") -> str:
    """A complete standalone HTML file."""
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{esc(title)}</title>\n</head>\n<body>\n"
        + render(run, title=title)
        + "</body>\n</html>\n"
    )
