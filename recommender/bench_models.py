#!/usr/bin/env python3
"""Benchmark LLM model / reasoning effort for the recommender's three AI jobs. Costs real money (OpenRouter).

Per user and cutoff (the same time split as evaluate.py: history replayed up to the 75 % / 85 % date, truth = titles
first watched after it), each config does:
  1. candidates: ai_candidates() -> TMDB-resolved titles; hits on the truth, and hits no other source found
  2. pick:       the LLM picks per row -> hits in the interleaved top 12
  3. reasons:    reasons for the score's rows -> coverage, unknown ids, > 90 characters, missing languages
Caveat: the LLM knows which titles became popular after a cutoff (titles released after it are dropped, older ones
that became hits later are not), so compare configs with each other, not with the TMDB sources. Small numbers:
1-3 hits difference is noise.
Usage: [USERS="alice bob"] [CUTS="0.75 0.85"] [JOBS="candidates pick reasons"]
       [CONFIGS="google/gemini-3.7-flash:,openai/gpt-6-luna:low"] python3 bench_models.py
Use a key with its own monthly cap (OPENROUTER_KEY_FILE), so a benchmark can't stop production.
"""
import importlib.util
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

spec = importlib.util.spec_from_file_location("ev", Path(__file__).resolve().parent / "evaluate.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
rec = ev.rec

CONFIGS = [(m, e or None) for m, e in (c.rsplit(":", 1) for c in os.environ.get(
    "CONFIGS", "google/gemini-3.7-flash:,openai/gpt-6-luna:low,openai/gpt-6-luna:medium,openai/gpt-6-luna:xhigh").split(","))]
JOBS = set(os.environ.get("JOBS", "candidates pick reasons").split())


def main():
    rec.load_env()
    tmdb = rec.Tmdb(os.environ["TMDB_API_KEY"])
    users = os.environ.get("USERS", "").split() or sorted(d.name for d in rec.HIST.iterdir() if d.is_dir())
    results = {f"{m} {e or ''}".strip(): {"cand": [], "pick": [], "reasons": [], "usd": 0.0, "secs": 0.0, "errors": 0}
               for m, e in CONFIGS}
    samples = {}
    try:
        for who in users:
            hist = rec.HIST / who
            for cut in ev.cutoffs_for(hist):
                rec.NOW = cut
                with tempfile.TemporaryDirectory() as tmp:
                    d = Path(tmp)
                    ev.filtered_copy(hist, cut, d)
                    truth = ev.truth_for(tmdb, hist, rec.read_history(d))
                    print(f"truth {who} {cut}: {len(truth)} titles", file=sys.stderr, flush=True)
                    rec.AI_CANDIDATES = False
                    res = rec.build(tmdb, d)
                other = set(res["cand_src"])
                rows = rec.row_lists(res)
                small = {n: lst[:rec.PER_ROW] for n, lst in rows.items() if lst}
                for m, e in CONFIGS:
                    k = f"{m} {e or ''}".strip()
                    rec.LLM_MODEL, rec.LLM_EFFORT = m, e
                    rec.CAND_MODEL, rec.CAND_EFFORT = m, e
                    r = results[k]
                    usd0, t0 = rec.COST["usd"], time.time()
                    try:
                        if "candidates" in JOBS:
                            got = rec.ai_candidates(tmdb, res["pos"], res["neg"], res["prefs"])
                            keys = {(kind, x["id"]) for kind, x in got}
                            r["cand"].append({"resolved": len(got), "hits": len(keys & truth),
                                              "new_hits": len((keys - other) & truth), "new": len(keys - other),
                                              "truth": len(truth), "who": who})
                        if "pick" in JOBS:
                            r["pick"].append(len(set(ev.llm_ranking(res)[:12]) & truth))
                        if "reasons" in JOBS:
                            out = rec.llm_rows(res["profile"], small, reasons_only=True)
                            ids = {(n, s["id"]) for n, lst in small.items() for s in lst}
                            got_r = [(n, x) for n, lst in out.items() if isinstance(lst, list) for x in lst if isinstance(x, dict)]
                            ok = [(n, x) for n, x in got_r if (n, x.get("id")) in ids]
                            texts = [[x.get(f"reason_{lang}") or "" for lang in rec.REASON_LANGS] for _, x in ok]
                            flat = [t for ts in texts for t in ts]
                            r["reasons"].append({"cover": len({(n, x.get("id")) for n, x in ok}) / max(len(ids), 1),
                                                 "bad_ids": len(got_r) - len(ok),
                                                 "too_long": sum(len(t) > 90 for t in flat) / max(len(flat), 1),
                                                 "missing_lang": sum(not all(ts) for ts in texts) / max(len(texts), 1),
                                                 "avg_len": statistics.mean(len(t) for t in flat) if flat else 0})
                            samples.setdefault(k, []).extend(" | ".join(ts) for ts in texts[:2])
                    except Exception as ex:  # noqa: BLE001
                        r["errors"] += 1
                        print(f"{k} {who} {cut}: ERROR {ex}", file=sys.stderr)
                    r["usd"] += rec.COST["usd"] - usd0
                    r["secs"] += time.time() - t0
                    print(f"done {k} {who} {cut}", file=sys.stderr, flush=True)
    finally:
        tmdb.save()
    print("\n== results (summed over users x cutoffs)")
    for k, r in results.items():
        c, rs = r["cand"], r["reasons"]
        print(f"\n{k}: ${r['usd']:.3f}, {r['secs']:.0f} s, errors {r['errors']}")
        if c:
            print(f"  candidates: resolved {sum(x['resolved'] for x in c)}, hits {sum(x['hits'] for x in c)} of truth "
                  f"{sum(x['truth'] for x in c)}, new (no other source) {sum(x['new'] for x in c)} of which hits "
                  f"{sum(x['new_hits'] for x in c)}")
        if r["pick"]:
            print(f"  pick mode: hits@12 {r['pick']} (total {sum(r['pick'])})")
        if rs:
            print(f"  reasons: cover {statistics.mean(x['cover'] for x in rs):.0%}, bad ids {sum(x['bad_ids'] for x in rs)}, "
                  f">90 chars {statistics.mean(x['too_long'] for x in rs):.0%}, missing language "
                  f"{statistics.mean(x['missing_lang'] for x in rs):.0%}, avg length {statistics.mean(x['avg_len'] for x in rs):.0f}")
        for smp in samples.get(k, [])[:3]:
            print("    ·", smp[:220])


if __name__ == "__main__":
    main()
