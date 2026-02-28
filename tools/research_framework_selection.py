#!/usr/bin/env python3
"""Harvest external research signals for Framework-with-selection test planning.

This tool is intentionally lightweight:
- queries arXiv and Crossref
- stores machine-readable evidence
- emits concrete test execution-key recommendations
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

ARXIV_API = "https://export.arxiv.org/api/query"
CROSSREF_API = "https://api.crossref.org/works"
TIMEOUT_S = 20


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def score_text(text: str, patterns: list[str]) -> int:
    lowered = text.lower()
    return sum(1 for p in patterns if p.lower() in lowered)


def default_queries() -> list[str]:
    return [
        "entanglement entropy finite size scaling ising cyclic boundary",
        "heisenberg spin chain entanglement entropy finite size",
        "tensor network MERA model selection AIC BIC",
        "cyclic boundary conditions 1D spin chain benchmark",
    ]


def parse_arxiv(xml_text: str) -> list[dict[str, Any]]:
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_text)
    rows: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", ns):
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        link = ""
        for link_node in entry.findall("atom:link", ns):
            href = link_node.attrib.get("href", "")
            if href.startswith("http"):
                link = href
                break
        published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()
        rows.append(
            {
                "source": "arxiv",
                "title": title,
                "abstract": summary,
                "url": link,
                "published": published,
            }
        )
    return rows


def fetch_arxiv(query: str, max_results: int) -> tuple[list[dict[str, Any]], str | None]:
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    try:
        url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response:  # noqa: S310
            data = response.read().decode("utf-8", errors="replace")
        return parse_arxiv(data), None
    except Exception as exc:  # noqa: BLE001
        return [], f"arxiv_error: {exc}"


def fetch_crossref(query: str, rows: int) -> tuple[list[dict[str, Any]], str | None]:
    params = {"query": query, "rows": rows, "sort": "relevance"}
    try:
        url = f"{CROSSREF_API}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response:  # noqa: S310
            data = response.read().decode("utf-8", errors="replace")
        payload = json.loads(data)
        items = payload.get("message", {}).get("items", [])
        out: list[dict[str, Any]] = []
        for item in items:
            title_parts = item.get("title", []) or []
            title = title_parts[0] if title_parts else ""
            abstract = item.get("abstract") or ""
            abstract = re.sub("<[^>]+>", " ", abstract)
            url = item.get("URL") or ""
            issued = item.get("issued", {}).get("date-parts", [])
            year = ""
            if issued and isinstance(issued, list) and issued[0]:
                year = str(issued[0][0])
            out.append(
                {
                    "source": "crossref",
                    "title": title,
                    "abstract": abstract.strip(),
                    "url": url,
                    "published": year,
                }
            )
        return out, None
    except Exception as exc:  # noqa: BLE001
        return [], f"crossref_error: {exc}"


def rank_and_recommend(items: list[dict[str, Any]], underdetermined_cycles: int) -> dict[str, Any]:
    signal_patterns = [
        "cyclic",
        "boundary",
        "finite size",
        "heisenberg",
        "entanglement entropy",
        "tensor network",
        "mera",
        "aic",
        "bic",
    ]
    for row in items:
        text = f"{row.get('title', '')} {row.get('abstract', '')}"
        row["relevance_score"] = score_text(text, signal_patterns)

    ranked = sorted(items, key=lambda x: int(x.get("relevance_score", 0)), reverse=True)
    top = ranked[:12]

    cumulative = sum(int(x.get("relevance_score", 0)) for x in top)
    escalate_tier_c = underdetermined_cycles >= 2 and cumulative >= 6
    if any("heisenberg" in f"{x.get('title', '')} {x.get('abstract', '')}".lower() for x in top):
        escalate_tier_c = escalate_tier_c or underdetermined_cycles >= 1

    recommended_execution_keys: list[str] = [
        "claim3_optionb_regime_check",
        "claim2_seed_perturbation",
    ]
    if escalate_tier_c:
        recommended_execution_keys.extend(
            [
                "claim3p_ising_cyclic_l8",
                "claim3p_heisenberg_cyclic_l8",
                "claim3p_l16_gate",
            ]
        )

    return {
        "top_evidence": top,
        "escalate_tier_c": escalate_tier_c,
        "recommended_execution_keys": recommended_execution_keys,
        "signal_score": cumulative,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    evidence = payload.get("recommendations", {}).get("top_evidence", [])
    rec = payload.get("recommendations", {})
    lines = [
        "# Research Signals: Framework with Selection",
        "",
        f"Generated: {payload.get('generated_at_utc', '')}",
        f"Underdetermined cycles observed: `{payload.get('underdetermined_cycles', 0)}`",
        f"Signal score: `{rec.get('signal_score', 0)}`",
        f"Escalate Tier C: `{rec.get('escalate_tier_c', False)}`",
        "",
        "## Recommended Execution Keys",
    ]
    for key in rec.get("recommended_execution_keys", []):
        lines.append(f"- `{key}`")
    lines.extend(["", "## Top External Evidence", "| Source | Title | Published | URL | Score |", "|---|---|---|---|---:|"])
    if evidence:
        for row in evidence:
            title = str(row.get("title", "")).replace("|", " ").strip()
            url = str(row.get("url", "")).strip()
            pub = str(row.get("published", "")).strip()
            src = str(row.get("source", "")).strip()
            score = int(row.get("relevance_score", 0))
            lines.append(f"| {src} | {title[:120]} | {pub} | {url} | {score} |")
    else:
        lines.append("| none | no evidence retrieved | - | - | 0 |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Harvest external research and recommend next tests")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-results-per-query", type=int, default=5)
    parser.add_argument("--underdetermined-cycles", type=int, default=1)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_json = args.output_json or (run_dir / "results" / "research" / "research_signal.json")
    output_md = args.output_md or (run_dir / "results" / "research" / "research_signal.md")
    queries = default_queries()

    collected: list[dict[str, Any]] = []
    query_logs: list[dict[str, Any]] = []
    errors: list[str] = []

    for query in queries:
        arxiv_rows, arxiv_err = fetch_arxiv(query, args.max_results_per_query)
        if arxiv_err:
            errors.append(arxiv_err)
        cross_rows, cross_err = fetch_crossref(query, args.max_results_per_query)
        if cross_err:
            errors.append(cross_err)
        rows = arxiv_rows + cross_rows
        collected.extend(rows)
        query_logs.append(
            {
                "query": query,
                "arxiv_count": len(arxiv_rows),
                "crossref_count": len(cross_rows),
            }
        )

    recommendations = rank_and_recommend(collected, args.underdetermined_cycles)
    payload = {
        "generated_at_utc": utc_now(),
        "run_dir": str(run_dir),
        "underdetermined_cycles": args.underdetermined_cycles,
        "queries": query_logs,
        "total_documents": len(collected),
        "errors": errors,
        "recommendations": recommendations,
    }
    write_json(output_json, payload)
    write_markdown(output_md, payload)
    print(str(output_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
