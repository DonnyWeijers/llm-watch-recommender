#!/usr/bin/env python3
"""Netflix data request ("Get my info") -> extra taste signals for one profile.

Usage: netflix_import.py <CONTENT_INTERACTION dir> <profile name> <history/<user> dir>
Writes (Title,Date[,Rating]; never commit them):
  netflix-ratings.csv    thumbs: 1 = down, 2 = up, 3 = love (0 = rating removed, skipped)
  netflix-abandoned.csv  titles started but stopped early: a film < 20 min in total, a series < 15 min over all
                         its episodes. Autoplay previews (HOOK, TRAILER, ...) are ignored: they say nothing about taste.
The plain viewing history comes from the per-profile export (Account -> profile -> Viewing activity -> Download all),
saved as netflix-history.csv. Only rows of <profile name> are used, so shared accounts stay separated.
"""
import csv
import importlib.util
import sys
from pathlib import Path


def main():
    src, profile, out = Path(sys.argv[1]), sys.argv[2].strip(), Path(sys.argv[3])
    spec = importlib.util.spec_from_file_location("rec", Path(__file__).with_name("recommend.py"))
    rec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rec)

    def rows(name):
        f = src / name
        if not f.exists():
            return []
        with f.open(encoding="utf-8-sig", newline="") as fh:
            return [r for r in csv.DictReader(fh) if r.get("Profile Name", "").strip() == profile]

    def write(name, header, data):
        with (out / name).open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(data)
        print(f"{name}: {len(data)}")

    ratings = []
    for r in rows("Ratings.csv"):
        val = {"1": "down", "2": "up", "3": "love"}.get(r.get("Thumbs Value", "").strip())
        if val and r.get("Title Name", "").strip():
            ratings.append([r["Title Name"].strip(), r.get("Event Utc Ts", "")[:10], val])
    write("netflix-ratings.csv", ["Title", "Date", "Rating"], ratings)

    views = [r for r in rows("ViewingActivity.csv") if not r.get("Supplemental Video Type", "").strip()]
    heads = rec.series_heads(views)
    total = {}
    for r in views:
        raw = r["Title"].strip()
        name, season, *_ = rec.clean_title(raw)
        parts = raw.split(": ")
        is_tv = bool(season) or bool(len(parts) >= 2 and (parts[0] in heads or rec.EPISODE_WORD_RE.match(parts[1])))
        if is_tv and season is None:
            name = rec.clean_title(parts[0])[0]
        if not name:
            continue
        h, m, s = (int(x) for x in r["Duration"].split(":"))
        t = total.setdefault(rec.norm(name), {"name": name, "tv": is_tv, "min": 0.0, "last": ""})
        t["min"] += h * 60 + m + s / 60
        t["tv"] |= is_tv
        t["last"] = max(t["last"], r["Start Time"][:10])
    abandoned = sorted([t["name"], t["last"]] for t in total.values() if t["min"] < (15 if t["tv"] else 20))
    write("netflix-abandoned.csv", ["Title", "Date"], abandoned)


if __name__ == "__main__":
    main()
