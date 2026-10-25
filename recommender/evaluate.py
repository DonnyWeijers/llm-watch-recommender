#!/usr/bin/env python3
"""Backtest the recommender per user with a time split. Writes data/eval-<date>.json and prints a summary.

For each user and each cutoff (CUTS, default "0.75 0.85": the dates by which 75 % / 85 % of their dated history rows
were watched; scores are averaged over them) the recommender only sees rows before the cutoff (watchlists are left
out: undated, and they would leak later intent). Truth = titles first watched after the cutoff (not abandoned, not
thumbs-down), matched to TMDB. Scored on rankings of the candidate pool:
  hits@N, precision@N = hits/N, recall@N = hits/|truth|, pool = share of the truth that is a candidate at all.
Rankings:
  ranking  the whole pool by score
  rows     what the page shows without an LLM: best-scored per row, rows interleaved
  popular  the same pool sorted by TMDB popularity: the baseline any model should beat
  llm      (EVAL_LLM=1, costs money) the LLM's own picks per row, interleaved
  old      (OLD=<path to an older recommend.py>) the previous version on the same data
Per candidate source: the share of the truth it brings into the pool, and the share only it brings.
Usage: [OLD=/path/recommend_old.py] [USERS="alice bob"] [EVAL_LLM=1] python3 evaluate.py
"""
import csv
import datetime as dt
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

DIR = Path(__file__).resolve().parent
NS = (12, 24, 48)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rec = load(DIR / "recommend.py", "rec_new")
# replays run the AI candidates on a cheap model: the replay flatters the AI anyway (it knows later hits)
rec.CAND_MODEL = os.environ.get("EVAL_CAND_MODEL", "openai/gpt-6-luna")
rec.CAND_EFFORT = os.environ.get("EVAL_CAND_EFFORT", "medium")
rec.AI_CANDIDATES = os.environ.get("EVAL_LLM") == "1"
old = load(Path(os.environ["OLD"]), "rec_old") if os.environ.get("OLD") else None


def dated_rows(f):
    with f.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def cutoffs_for(hist):
    dates = sorted(d for f in hist.glob("*history*.csv") for r in dated_rows(f) if (d := rec.row_date(r)))
    if len(dates) < 20:
        return []
    return [dates[int(len(dates) * float(q))] for q in os.environ.get("CUTS", "0.75 0.85").split()]


def filtered_copy(hist, before, dst):
    """The history as it was on <before>: dated rows before it; watchlists left out."""
    for f in hist.glob("*.csv"):
        if "watchlist" in f.stem or f.name == "overrides.csv":
            continue
        rows = dated_rows(f)
        if not rows:
            continue
        keep = [r for r in rows if not rec.row_date(r) or rec.row_date(r) < before]
        with (dst / f.name).open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(keep)
    for name in ("overrides.csv", "prefs.json"):
        if (hist / name).exists():
            shutil.copy(hist / name, dst)


def truth_for(tmdb, hist, pre):
    """Titles first watched after the cutoff: in the full history, not in the replayed one."""
    full = rec.read_history(hist)
    ov = rec.read_overrides(hist)
    before = {k for k, it in pre.items() if "watched" in it["kinds"]}
    out = set()
    for k, it in full.items():
        kinds = it["kinds"]
        if k in before or "watched" not in kinds or "disliked" in kinds or "abandoned" in kinds or ov.get(k) == "skip":
            continue
        if ov.get(k):
            kind, tid = ov[k].split(":")
            out.add((kind, int(tid)))
            continue
        m, _ = rec.match(tmdb, it)
        if m:
            out.add(m)
    return out


def interleave(lists):
    ranking = []
    for i in range(max(map(len, lists), default=0)):
        ranking += [lst[i] for lst in lists if i < len(lst) and lst[i][0] and lst[i] not in ranking]
    return ranking


def rows_ranking(res):
    """The page without an LLM: best-scored per row, rows interleaved (the fair baseline for the LLM's picks)."""
    return interleave([[rec.key_of(s) for s in lst[:rec.PER_ROW]] for lst in rec.row_lists(res).values()])


def llm_ranking(res):
    """The LLM's picks per row (as on the page in pick mode), interleaved row by row."""
    rows = rec.row_lists(res)
    out = rec.llm_rows(res["profile"], rows, reasons_only=False)
    return interleave([[(next((s["kind"] for s in rows[n] if s["id"] == p.get("id")), None), p.get("id"))
                        for p in out.get(n, []) if isinstance(p, dict)] for n in rows])


def score(ranking, truth):
    r = {"pool": round(len(truth & set(ranking)) / len(truth), 3) if truth else None}
    for n in NS:
        hits = len(truth & set(ranking[:n]))
        r[f"hits@{n}"] = hits
        r[f"precision@{n}"] = round(hits / n, 3)
        r[f"recall@{n}"] = round(hits / len(truth), 3) if truth else None
    return r


def source_coverage(res, truth):
    """Per candidate source: share of the truth it brings into the scored pool, and the share only it brings
    (without it those titles would not be candidates at all)."""
    in_pool = {rec.key_of(s): set(s.get("src") or ["similar"]) for s in res["pool"]}
    hit = {k: v for k, v in in_pool.items() if k in truth}
    return {src: {"pool": round(sum(src in v for v in hit.values()) / len(truth), 3),
                  "only": round(sum(v == {src} for v in hit.values()) / len(truth), 3)}
            for src in sorted({x for v in in_pool.values() for x in v})}


def main():
    rec.load_env()
    tmdb = rec.Tmdb(os.environ["TMDB_API_KEY"])
    users = os.environ.get("USERS", "").split() or sorted(d.name for d in rec.HIST.iterdir() if d.is_dir())
    report = {"date": f"{dt.date.today()}", "users": {}}
    try:
        for who in users:
            hist = rec.HIST / who
            runs = []
            for cut in cutoffs_for(hist):
                rec.NOW = cut
                with tempfile.TemporaryDirectory() as tmp:
                    d = Path(tmp)
                    filtered_copy(hist, cut, d)
                    truth = truth_for(tmdb, hist, rec.read_history(d))
                    if not truth:
                        continue
                    rec.COST["user"] = f"eval:{who}"
                    res = rec.build(tmdb, d)
                    models = {"ranking": score([rec.key_of(s) for s in res["pool"]], truth),
                              "rows": score(rows_ranking(res), truth),
                              "popular": score([rec.key_of(s) for s in sorted(res["pool"], key=lambda s: -s["pop"])], truth)}
                    if os.environ.get("EVAL_LLM") == "1" and rec.openrouter_key():
                        models["llm"] = score(llm_ranking(res), truth)
                    if old:
                        old.NOW = cut
                        ores = old.build(tmdb, d)
                        models["old"] = score([old.key_of(s) for s in ores["pool"]], truth)
                runs.append({"cutoff": f"{cut}", "truth": len(truth), "anchors": len(res["anchors"]),
                             "pool": len(res["pool"]), "models": models, "sources": source_coverage(res, truth)})
            if not runs:
                print(f"{who}: too little dated history, skipped", file=sys.stderr)
                continue
            avg = {m: {k: round(sum(r["models"][m][k] or 0 for r in runs) / len(runs), 3)
                       for k in runs[0]["models"][m]} for m in runs[0]["models"]}
            srcs = sorted({x for r in runs for x in r["sources"]})
            report["users"][who] = {"runs": runs, "avg": avg, "sources": {
                x: {k: round(sum((r["sources"].get(x) or {}).get(k, 0) for r in runs) / len(runs), 3)
                    for k in ("pool", "only")} for x in srcs}}
    finally:
        tmdb.save()
    rec.DATA.mkdir(parents=True, exist_ok=True)
    (rec.DATA / f"eval-{dt.date.today()}.json").write_text(json.dumps(report, indent=1))
    for who, u in report["users"].items():
        cuts = ", ".join(f"{r['cutoff']} ({r['truth']} new titles after it)" for r in u["runs"])
        print(f"\n{who}: cutoffs {cuts}; averaged:")
        for name, m in u["avg"].items():
            print(f"  {name:8} pool {m['pool']:.0%}  " + "  ".join(
                f"@{n}: {m[f'hits@{n}']:.1f} hits (P {m[f'precision@{n}']:.0%}, R {m[f'recall@{n}']:.0%})" for n in NS))
        print("  sources (share of later-watched titles in the pool via it · only via it): " + ", ".join(
            f"{x} {v['pool']:.0%}·{v['only']:.0%}" for x, v in u["sources"].items()))
    print(f"\nLLM cost ${rec.COST['usd']:.4f}")


if __name__ == "__main__":
    main()
