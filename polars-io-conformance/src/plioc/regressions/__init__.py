"""Committed counterexamples.

Every Hypothesis discovery is shrunk to a spec and written here, which turns a probabilistic
find into a fast deterministic case that runs on every commit forever after. The files are the
whole record -- there is no accompanying data.

A regression file holds a `CaseSpec`, optionally paired with the `QuerySpec` that broke it:

    {"case": {...}, "query": {...} | null, "note": "why this exists"}
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from plioc import codec
from plioc.query import QuerySpec
from plioc.spec import CaseSpec

DIRECTORY = Path(__file__).parent


@dataclass(frozen=True)
class Regression:
    case: CaseSpec
    query: QuerySpec | None
    note: str
    path: Path


def load_all() -> list[Regression]:
    out = []
    for path in sorted(DIRECTORY.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        out.append(
            Regression(
                case=codec.decode(raw["case"]),
                query=codec.decode(raw["query"]) if raw.get("query") else None,
                note=raw.get("note", ""),
                path=path,
            )
        )
    return out


def load_regressions() -> list[CaseSpec]:
    return [r.case for r in load_all()]


def record(case: CaseSpec, query: QuerySpec | None, note: str) -> Path:
    """Write a counterexample. Called by the `--record-regression` flag, not by hand."""
    slug = re.sub(r"[^a-z0-9]+", "-", case.id.lower()).strip("-") or "case"
    path = DIRECTORY / f"{slug}.json"
    n = 2
    while path.exists():
        path = DIRECTORY / f"{slug}-{n}.json"
        n += 1
    payload = {
        "note": note,
        "case": codec.encode(case),
        "query": codec.encode(query) if query is not None else None,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
