"""Unit tests that need no network: title parsing, history reading, weights, the backtest metrics, and a full
build() against a fake TMDB.

  python3 -m unittest discover tests
"""
import csv
import datetime as dt
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ["REC_AI_CANDIDATES"] = "0"
os.environ["OPENROUTER_KEY_FILE"] = "/nonexistent/key"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "recommender" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rec = load("recommend")
ev = load("evaluate")


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


class CleanTitle(unittest.TestCase):
    def test_season(self):
        self.assertEqual(rec.clean_title("Some Show: Season 2: The Return")[:2], ("Some Show", 2))

    def test_dash_season_and_year(self):
        self.assertEqual(rec.clean_title("Other Show (2019) - Season 10")[:3], ("Other Show", 10, 2019))

    def test_limited_series(self):
        self.assertEqual(rec.clean_title("The Crown: Limited Series: Episode 3")[:2], ("The Crown", 1))

    def test_two_part_film_stays_film(self):
        self.assertEqual(rec.clean_title("Deathly Hallows: Part 1")[:2], ("Deathly Hallows: Part 1", None))

    def test_trailer_and_alt_title(self):
        name, season, year, alt, trailer = rec.clean_title("Lokale titel (Original Title)")
        self.assertEqual((name, alt), ("Lokale titel", "Original Title"))
        self.assertTrue(rec.clean_title("Episode 0: Big Show Trailer")[4])


class History(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.hist = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_series_without_season_grouped(self):
        write_csv(self.hist / "netflix-history.csv", ["Title", "Date"],
                  [["Show X: Pilot", "01/02/24"], ["Show X: The Second One", "01/03/24"],
                   ["Show X: Third", "01/04/24"], ["A Film: The Sequel", "02/01/24"]])
        items = rec.read_history(self.hist)
        self.assertEqual(items["show x"]["episodes"], 3)
        self.assertIn("a film the sequel", items)

    def test_before_cutoff_and_kinds(self):
        write_csv(self.hist / "netflix-history.csv", ["Title", "Date"],
                  [["Early: Season 1: E1", "2024-01-01"], ["Late: Season 1: E1", "2024-06-01"]])
        write_csv(self.hist / "netflix-ratings.csv", ["Title", "Date", "Rating"], [["Early", "2024-01-02", "love"]])
        items = rec.read_history(self.hist, before=dt.date(2024, 3, 1))
        self.assertNotIn("late", items)
        self.assertEqual(items["early"]["kinds"], {"watched", "loved"})

    def test_watchlist_category_filter(self):
        write_csv(self.hist / "hbomax-watchlist.csv", ["Title", "Category"],
                  [["Mine", "My items"], ["Theirs", "Recommended for you"]])
        self.assertEqual(set(rec.read_history(self.hist)), {"mine"})


class Weights(unittest.TestCase):
    def item(self, kinds, episodes=0, last=None):
        return {"kinds": set(kinds), "episodes": episodes, "last": last}

    def test_disliked_and_abandoned_are_zero(self):
        self.assertEqual(rec.anchor_weight(self.item({"watched", "disliked"}, 10), "tv"), 0)
        self.assertEqual(rec.anchor_weight(self.item({"watched", "abandoned"}, 1), "tv"), 0)

    def test_thumbs_add_on_top(self):
        base = rec.anchor_weight(self.item({"watched"}, 7), "tv")
        self.assertAlmostEqual(rec.anchor_weight(self.item({"watched", "loved"}, 7), "tv"), base + 2)

    def test_recency_halves_after_half_life_with_floor(self):
        rec.NOW = dt.date(2026, 1, 1)
        self.assertAlmostEqual(rec.recency({"last": dt.date(2023, 1, 1)}), 0.5, places=2)
        self.assertEqual(rec.recency({"last": dt.date(2000, 1, 1)}), 0.25)
        rec.NOW = dt.date.today()


class Metrics(unittest.TestCase):
    def test_score(self):
        ranking = [("tv", i) for i in range(60)]
        truth = {("tv", 3), ("tv", 30), ("tv", 99)}
        s = ev.score(ranking, truth)
        self.assertEqual((s["hits@12"], s["hits@48"]), (1, 2))
        self.assertAlmostEqual(s["pool"], 0.667, places=3)

    def test_interleave(self):
        self.assertEqual(ev.interleave([[("tv", 1), ("tv", 2)], [("movie", 9)]]), [("tv", 1), ("movie", 9), ("tv", 2)])


class FakeTmdb:
    """Ten series: 1-3 watched; their recommendations point at 4-10, of which 4 shares their genre and keyword."""

    def __init__(self):
        self.calls = 0

    def show(self, i):
        crime = i in (1, 2, 3, 4)
        return {"id": i, "name": f"Show {i}", "first_air_date": "2022-01-01", "genres": [{"id": 80 if crime else 35,
                "name": "Crime" if crime else "Comedy"}], "keywords": {"results": [{"id": 500 if crime else 600,
                "name": "detective" if crime else "office"}]}, "origin_country": ["US"], "vote_average": 7.5,
                "vote_count": 500, "popularity": 50 + i,
                "recommendations": {"results": [{"id": j} for j in range(4, 11)] if i <= 3 else []},
                "similar": {"results": []}}

    def get(self, path, ttl_days=7, **params):
        if path.startswith("/search/"):
            q = params.get("query", "")
            return {"results": [{"id": int(q.split()[-1]), "name": q, "media_type": "tv", "popularity": 10}]
                    if q.startswith("Show ") else []}
        return {"results": []}

    def details(self, kind, tid):
        return self.show(tid) if kind == "tv" and 1 <= tid <= 10 else None

    def save(self):
        pass


class BuildEndToEnd(unittest.TestCase):
    def test_taste_ranks_the_matching_candidate_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            hist = Path(tmp)
            write_csv(hist / "netflix-history.csv", ["Title", "Date"],
                      [[f"Show {i}: Season 1: Episode {e}", "2025-01-0" + str(e)] for i in (1, 2, 3) for e in (1, 2, 3)])
            res = rec.build(FakeTmdb(), hist)
        self.assertEqual(len(res["anchors"]), 3)
        self.assertEqual(res["pool"][0]["id"], 4)  # the crime/detective candidate beats the comedies
        self.assertTrue(all(s["id"] >= 4 for s in res["pool"]))  # watched titles are never suggested
        self.assertEqual(res["pool"][0]["because"] in {"Show 1", "Show 2", "Show 3"}, True)


if __name__ == "__main__":
    unittest.main()
