#!/usr/bin/env python3
"""Personal series and film recommendations from streaming-service viewing history, with LLM help.

Input:  history/<user>/*.csv: exports from streaming services (never commit them):
          *-history.csv    Title[,Date][,Count][,Type]  watched (Count = episodes; Type "TV Show"/"Movie" helps matching)
          netflix*.csv     Title,Date  Netflix viewing activity ("Show: Season 1: Episode 3") is understood as-is
          *-watchlist.csv  Title[,Category]  lists the user made
          *-ratings.csv    Title,Rating  love | up | down (Netflix thumbs): love/up weigh more, down = never again
          *-abandoned.csv  Title  started but stopped within minutes: no taste signal, never suggested again
          (netflix_import.py makes ratings/abandoned from a Netflix data request)
        history/<user>/overrides.csv  Title,tmdb   fixes for matches: tv:<id> | movie:<id> | skip
        history/<user>/prefs.json     {"exclude": ["anime", "animation", "kids"], "persona": "a retired teacher"}
Output: data/<user>/recommendations.json, data/<user>/index.html, data/<user>/unmatched.txt, data/costs.csv
Data:   TMDB (cached on disk) for matching, details and candidates; OpenRouter (optional) for extra candidates and
        one-line reasons. Only titles leave the machine, never the raw history.

How it works:
  1. history -> titles, matched to TMDB; each gets a weight (episodes watched, thumbs, recency, lists)
  2. taste profile: genres, keywords (idf-weighted: distinctive keywords count more), countries; thumbs-down and
     abandoned titles pull theirs down
  3. candidates from several sources: TMDB recommendations/similar of the strongest titles, discover on the most
     distinctive keywords ("themes"), popular on the user's own services, other work of the creators and lead actors
     ("people"), well-rated films in their genres, trending, and free-form LLM suggestions resolved on TMDB ("ai")
  4. score = 0.4 support (how many of their titles point here, log-scaled, popularity-damped) + 0.35 taste fit
     + 0.15 rating + 0.1 recency; rows (series, films, true crime, local) are filled by score
  5. the LLM writes one short reason per pick (REC_LLM_MODE=reasons, default) or picks the rows itself (pick)
evaluate.py replays the history at earlier dates and scores what the user watched afterwards.
"""
import collections
import csv
import datetime as dt
import difflib
import html
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import requests

DIR = Path(__file__).resolve().parent
HIST = Path(os.environ.get("HISTORY_DIR", DIR.parent / "history"))
DATA = Path(os.environ.get("DATA_DIR", DIR.parent / "data"))

# region and language for TMDB (search, providers, discover) and the reasons
REGION = os.environ.get("REGION", "NL")
TMDB_LANGUAGE = os.environ.get("TMDB_LANGUAGE", "en-US")
LOCAL_COUNTRY = os.environ.get("LOCAL_COUNTRY", REGION)  # a "local" row if the user watches enough of it
REASON_LANGS = os.environ.get("REASON_LANGS", "en").split(",")  # e.g. "nl,en": one reason per language

# LLM via OpenRouter: one key with its own monthly cap, exact cost per call (usage.cost)
OPENROUTER_KEY_FILE = Path(os.path.expanduser(os.environ.get("OPENROUTER_KEY_FILE", "~/.config/openrouter/recommendations.key")))
LLM_MODEL = os.environ.get("REC_LLM_MODEL", "openai/gpt-6-luna")
LLM_EFFORT = os.environ.get("REC_LLM_EFFORT", "medium")
CAND_MODEL = os.environ.get("REC_CAND_MODEL", "deepseek/deepseek-v4.1-flash")
CAND_EFFORT = os.environ.get("REC_CAND_EFFORT", "high")
LLM_MODE = os.environ.get("REC_LLM_MODE", "reasons")  # reasons: the score picks, the LLM explains | pick: the LLM picks
AI_CANDIDATES = os.environ.get("REC_AI_CANDIDATES", "1") == "1"
EUR_PER_USD = float(os.environ.get("EUR_PER_USD", "0.86"))

# tuning (measured with evaluate.py)
HALF_LIFE_YEARS = 3.0
POP_DAMP = float(os.environ.get("REC_POP_DAMP", 0.1))
ANCHORS = int(os.environ.get("REC_ANCHORS", 60))  # strongest titles whose TMDB recommendations become candidates
POOL = int(os.environ.get("REC_POOL", 400))  # candidates that get full details and a score
PEOPLE_ANCHORS = 8  # strongest titles whose creators/directors and lead actors bring their other work
PER_ROW = 12

# streaming services: history file prefix -> TMDB watch-provider id
SERVICE_PROVIDERS = {"netflix": 8, "videoland": 72, "skyshowtime": 1773, "hbomax": 1899, "prime": 119, "nlziet": 472,
                     "disney": 337, "appletv": 350}
DEFAULT_PROVIDERS = [int(x) for x in os.environ.get("DEFAULT_PROVIDERS", "8,119,337,350").split(",")]
PROVIDER_NAMES = {8: "Netflix", 175: "Netflix", 72: "Videoland", 1773: "SkyShowtime", 1899: "HBO Max",
                  119: "Prime Video", 472: "NLZIET", 337: "Disney+", 350: "Apple TV+"}
WATCHLIST_CATEGORIES = {"", "my items", "mijn items"}  # HBO exports mix the user's list with HBO's own advice
ANIME_KW = 210024
ALWAYS_SKIP = {10763, 10767}  # news, talk: daily broadcasts, not something to recommend
PREF_GENRES = {"animation": 16, "kids": 10762}  # only skipped for users whose prefs.json says so

NOW = dt.date.today()  # evaluate.py sets this to its cutoff


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def load_env():
    env = DIR.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# ---------------------------------------------------------------- TMDB (cached on disk)

class Tmdb:
    def __init__(self, key, cache_file=None):
        self.s = requests.Session()
        if len(key) > 40:  # v4 read access token
            self.s.headers["Authorization"] = f"Bearer {key}"
            self.params = {}
        else:
            self.params = {"api_key": key}
        self.cache_file = Path(cache_file or DATA / "tmdb-cache.json")
        try:
            self.cache = json.loads(self.cache_file.read_text())
        except (OSError, ValueError):
            self.cache = {}
        self.calls = 0

    def get(self, path, ttl_days=7, **params):
        key = path + "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        hit = self.cache.get(key)
        if hit and time.time() - hit[0] < ttl_days * 86400:
            return hit[1]
        p = {"language": TMDB_LANGUAGE, **self.params, **params}
        for attempt in range(4):
            r = self.s.get("https://api.themoviedb.org/3" + path, params=p, timeout=20)
            if r.status_code == 429:
                time.sleep(2 + attempt * 2)
                continue
            if r.status_code == 404:
                val = None
                break
            if r.status_code == 401:
                raise SystemExit("TMDB: key rejected (401), check TMDB_API_KEY")
            r.raise_for_status()
            val = r.json()
            break
        else:
            raise RuntimeError(f"TMDB rate limit on {path}")
        self.calls += 1
        self.cache[key] = [time.time(), val]
        return val

    def details(self, kind, tid):
        return self.get(f"/{kind}/{tid}", append_to_response="keywords,recommendations,similar,watch/providers")

    def save(self):
        now = time.time()
        self.cache = {k: v for k, v in self.cache.items() if now - v[0] < 45 * 86400}
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.cache))
        tmp.replace(self.cache_file)


# ---------------------------------------------------------------- reading the exports

SEASON_RE = re.compile(r"(?:\s*[-–:,]\s*|\s+|\s*\()\b(?:season|seizoen|s)\s*0*(\d+)\b\)?.*$", re.I)
# Netflix series without "Season N": 'The Crown: Limited Series: Episode 3', 'Lupin: Part 1: Chapter 1'. Only with 3+
# segments, so two-part film titles ('… Deathly Hallows: Part 1') stay films.
SERIES_PART_RE = re.compile(r"(?:limited series|miniseries|miniserie|(?:part|deel|volume|vol\.|collection|book|boek)\s*(\d+)?"
                            r"|episode|aflevering|afl\.|chapter|hoofdstuk)\b", re.I)
EPISODE_WORD_RE = re.compile(r"(?:aflevering|afl\.|episode|pilot|pilotaflevering|hoofdstuk|chapter)\b", re.I)
DATE_FORMATS = ("%m/%d/%y", "%B %d, %Y", "%Y-%m-%d", "%d-%m-%Y")


def clean_title(raw):
    """'Some Show - Season 10' -> ('Some Show', 10, None, None, False): name, season, year, alternative title, trailer."""
    t = raw.strip().strip('"')
    trailer = bool(re.match(r"^Episode \d+:", t)) or "trailer" in t.lower()
    t = re.sub(r"^Episode \d+:\s*", "", t)
    t = re.sub(r"^Trailer:\s*", "", t, flags=re.I)
    t = re.sub(r"\s*\((?:4K UHD|TV)\)", "", t, flags=re.I)
    season = None
    m = SEASON_RE.search(t)
    if m:
        season = int(m.group(1))
        t = t[:m.start()]
    else:
        parts = t.split(": ")
        for i in range(1, len(parts)) if len(parts) >= 3 else ():
            pm = SERIES_PART_RE.match(parts[i])
            if pm:
                season, t = int(pm.group(1) or 1), ": ".join(parts[:i])
                break
    t = re.sub(r"\s*[-–:]?\s*(?:official\s+)?trailer\b.*$", "", t, flags=re.I)
    year = alt = None
    m = re.search(r"\s*\((\d{4})\)\s*$", t)
    if m:
        year, t = int(m.group(1)), t[:m.start()]
    m = re.search(r"\s*\(([^)]+)\)\s*$", t)  # 'Local title (Original Title)'
    if m:
        alt, t = m.group(1), t[:m.start()]
    t = t.strip(" -–:,")
    if re.fullmatch(r"(?i)(season|seizoen)?\s*\d*", t):
        t = ""
    return t, season, year, alt, trailer


def norm(s):
    s = s.lower().replace("&", "and").replace("’", "'")
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def row_date(row):
    v = (row.get("Date") or "").strip()
    for fmt in DATE_FORMATS:
        try:
            return dt.datetime.strptime(v, fmt).date()
        except ValueError:
            pass
    return None


def series_heads(rows):
    """Netflix titles without a season, 'Show: Episode title': a name before ': ' that comes back with 3+ different
    episode titles is a series (films with a colon appear once). Also 'Show: Episode 3' / 'Show: Pilot'."""
    seen = {}
    for row in rows:
        raw = (row.get("Title") or "").strip()
        parts = raw.split(": ")
        if len(parts) >= 2 and clean_title(raw)[1] is None:
            seen.setdefault(parts[0], set()).add(": ".join(parts[1:]))
    return {h for h, rest in seen.items() if len(rest) >= 3}


def read_history(hist, before=None):
    """All exports of one user -> {normalised name: item}. before (date): only rows dated before it (evaluate.py);
    undated rows always count."""
    items = {}
    for f in sorted(hist.glob("*.csv")):
        if f.name == "overrides.csv":
            continue
        service = f.stem.split("-")[0].lower()
        kind = next((k for k in ("watchlist", "ratings", "abandoned") if k in f.stem), "watched")
        with f.open(newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        if before:
            rows = [r for r in rows if not row_date(r) or row_date(r) < before]
        heads = series_heads(rows) if kind == "watched" else set()
        for row in rows:
            raw = (row.get("Title") or "").strip()
            cat = (row.get("Category") or "").strip().lower()
            if not raw or cat not in WATCHLIST_CATEGORIES:
                continue
            name, season, year, alt, trailer = clean_title(raw)
            parts = raw.split(": ")
            if season is None and len(parts) >= 2 and (parts[0] in heads or EPISODE_WORD_RE.match(parts[1])):
                name, season, year, alt, trailer = clean_title(parts[0])[0], 1, None, None, False
            if not name:
                continue
            count = int(row.get("Count") or 1) if (row.get("Count") or "1").isdigit() else 1
            it = items.setdefault(norm(name), {"name": name, "alt": alt, "year": year, "hint": None,
                                                "episodes": 0, "seasons": set(), "kinds": set(),
                                                "services": set(), "last": None})
            it["services"].add(service)
            d = row_date(row)
            if d and kind == "watched" and (not it["last"] or d > it["last"]):
                it["last"] = d
            typ = (row.get("Type") or "").strip().lower()  # e.g. Prime's export: "TV Show" / "Movie"
            if typ in ("tv show", "movie") and not it["hint"]:
                it["hint"] = "tv" if typ == "tv show" else "movie"
            if season:
                it["seasons"].add(season)
                it["hint"] = "tv"
            if kind == "ratings":  # Title,Rating: love | up | down (Netflix thumbs)
                r = (row.get("Rating") or "").strip().lower()
                it["kinds"].add({"love": "loved", "up": "liked", "down": "disliked"}.get(r, "rated"))
            elif kind == "abandoned":
                it["kinds"].add("abandoned")
            elif kind == "watchlist":
                it["kinds"].add("watchlist")
            elif trailer:
                it["kinds"].add("trailer")
            else:
                it["kinds"].add("watched")
                it["episodes"] += count
    return items


def read_overrides(hist):
    out = {}
    f = hist / "overrides.csv"
    if f.exists():
        with f.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                out[norm(row["Title"])] = row["tmdb"].strip()
    return out


def read_prefs(hist):
    """prefs.json (optional): {"exclude": ["anime", "animation", "kids"], "persona": "..."}. Without it only news and
    talk shows are skipped: the taste profile decides."""
    try:
        return json.loads((hist / "prefs.json").read_text())
    except (OSError, ValueError):
        return {}


def skip_filter(prefs):
    ex = set(prefs.get("exclude", []))
    genres = ALWAYS_SKIP | {g for name, g in PREF_GENRES.items() if name in ex}
    anime = "anime" in ex
    return lambda s: (anime and ANIME_KW in s["keywords"]) or bool(genres & set(s["genres"]))


def match(tmdb, it):
    """Best TMDB match for a cleaned title: ((kind, id), None), or (None, best guess for unmatched.txt)."""
    queries = [q for q in (it["name"], it["alt"]) if q]
    best, best_s = None, 0.0
    for q in queries:
        for lang in dict.fromkeys((TMDB_LANGUAGE, "en-US")):
            params = {"query": q, "language": lang}
            if it["year"]:
                params["year"] = it["year"]
            path = {"tv": "/search/tv", "movie": "/search/movie"}.get(it["hint"], "/search/multi")
            res = (tmdb.get(path, ttl_days=30, **params) or {}).get("results", [])[:8]
            for i, r in enumerate(res):
                mt = r.get("media_type", {"/search/tv": "tv", "/search/movie": "movie"}.get(path))
                if mt not in ("tv", "movie"):
                    continue
                names = [r.get(k) for k in ("name", "original_name", "title", "original_title") if r.get(k)]
                sim = max(difflib.SequenceMatcher(None, norm(q), norm(n)).ratio() for n in names)
                score = sim + (0.05 if mt == it["hint"] else 0) + 0.02 * max(0, 3 - i) \
                    + min(r.get("popularity", 0), 100) / 2000
                if score > best_s:
                    best, best_s = (mt, r["id"], names[0], sim), score
        if best and best[3] >= 0.9:
            break
    if best and best[3] >= 0.82:
        return (best[0], best[1]), None
    return None, (f"{best[0]}:{best[1]} {best[2]} (sim {best[3]:.2f})" if best else "no results")


# ---------------------------------------------------------------- profile + candidates

def kw_ids(d):
    k = d.get("keywords") or {}
    return [x["id"] for x in k.get("keywords", k.get("results", []))]


def providers(d):
    local = ((d.get("watch/providers") or {}).get("results") or {}).get(REGION) or {}
    names = []
    for p in local.get("flatrate", []) + local.get("free", []) + local.get("ads", []):
        n = PROVIDER_NAMES.get(p["provider_id"])
        if n and n not in names:
            names.append(n)
    return names


def summary(kind, d):
    title = d.get("name") or d.get("title") or "?"
    date = d.get("first_air_date") or d.get("release_date") or ""
    return {"kind": kind, "id": d["id"], "title": title, "year": date[:4], "poster": d.get("poster_path"),
            "overview": (d.get("overview") or "")[:300], "genres": [g["id"] for g in d.get("genres", [])],
            "genre_names": [g["name"] for g in d.get("genres", [])], "keywords": kw_ids(d),
            "countries": d.get("origin_country") or [c["iso_3166_1"] for c in d.get("production_countries", [])],
            "vote": d.get("vote_average") or 0, "votes": d.get("vote_count") or 0,
            "pop": d.get("popularity") or 0, "providers": providers(d)}


def key_of(s):
    return (s["kind"], s["id"])


def recency(it):
    """Tastes drift: a title last watched 3 years ago counts half, never less than a quarter; undated = 1."""
    last = it.get("last")
    if not last:
        return 1.0
    return max(0.5 ** (max((NOW - last).days, 0) / 365 / HALF_LIFE_YEARS), 0.25)


def anchor_weight(it, kind, movie_base=2.0):
    k = it["kinds"]
    if "disliked" in k or ("abandoned" in k and not {"loved", "liked"} & k):
        return 0.0  # thumbs down, or gave up within minutes: no taste signal, and never suggested again
    w = 0.0
    if "watched" in k:
        w = min(1 + math.log2(1 + it["episodes"]), 6) if kind == "tv" else movie_base
    elif "trailer" in k:
        w = 0.4
    if "loved" in k:  # thumbs add on top of the watched weight
        w += 2.0
    elif "liked" in k:
        w += 1.0
    if "watchlist" in k:
        w += 1.0
    return w * recency(it)


def build(tmdb, hist):
    """History -> anchors, taste profile, scored candidate pool. Returns everything evaluate.py and the page need."""
    items = read_history(hist)
    overrides = read_overrides(hist)
    anchors, unmatched = [], []
    for key, it in items.items():
        ov = overrides.get(key)
        if ov == "skip":
            continue
        if ov:
            kind, tid = ov.split(":")
            m = (kind, int(tid))
        else:
            m, guess = match(tmdb, it)
            if not m:
                unmatched.append(f"{it['name']}\t{guess}")
                continue
        d = tmdb.details(*m)
        if not d:
            unmatched.append(f"{it['name']}\tTMDB {m[0]}:{m[1]} not found")
            continue
        same = next((a for a in anchors if (a["kind"], a["id"]) == m), None)
        if same:  # two spellings of one show in one export: one show, episodes add up
            o = same["item"]
            o["episodes"] += it["episodes"]
            for k in ("seasons", "kinds", "services"):
                o[k] |= it[k]
            continue
        anchors.append({"kind": m[0], "id": m[1], "item": it, "details": d,
                        "title": d.get("name") or d.get("title")})
    log(f"history: {len(items)} titles, {len(anchors)} matched, {len(unmatched)} unmatched")

    # films vs series: a film counts 2 for someone who mostly watches series, up to 4 for a film watcher
    watched = [a for a in anchors if "watched" in a["item"]["kinds"]]
    film_share = sum(a["kind"] == "movie" for a in watched) / len(watched) if watched else 0.5
    for a in anchors:
        a["weight"] = anchor_weight(a["item"], a["kind"], movie_base=2.0 + 2.0 * film_share)
    seen = {(a["kind"], a["id"]) for a in anchors
            if {"watched", "disliked", "abandoned", "loved", "liked"} & a["item"]["kinds"]}
    watchlist = {(a["kind"], a["id"]) for a in anchors if "watchlist" in a["item"]["kinds"]} - seen

    # taste profile. Keywords weigh by how distinctive they are (idf: "murder" is in half their titles, "amnesia" in
    # three); thumbs down pulls its genres and keywords down, abandoned titles a little.
    prefs = read_prefs(hist)
    skip = skip_filter(prefs)
    pos = [a for a in anchors if a["weight"] > 0]
    df = collections.Counter(k for a in pos for k in set(kw_ids(a["details"])))
    idf = {k: math.log((len(pos) + 1) / (n + 1)) + 1 for k, n in df.items()}
    neg = []
    for a in anchors:
        k = a["item"]["kinds"]
        if "disliked" in k:
            neg.append((a["details"], -1.5))
        elif "abandoned" in k and not {"loved", "liked"} & k:
            neg.append((a["details"], -0.3))
    genre_w, kw_w, country_w = {}, {}, {}
    for d, w in [(a["details"], a["weight"]) for a in pos] + neg:
        s = summary("tv" if "name" in d else "movie", d)
        for g in s["genres"]:
            genre_w[g] = genre_w.get(g, 0) + w
        for k in s["keywords"]:
            kw_w[k] = kw_w.get(k, 0) + w * idf.get(k, 1.0)
        for c in s["countries"][:1]:
            country_w[c] = country_w.get(c, 0) + w
    top_kw = sum(sorted(kw_w.values(), reverse=True)[:12]) or 1
    top_genre = sum(sorted(genre_w.values(), reverse=True)[:3]) or 1
    total_w = sum(a["weight"] for a in pos) or 1

    cand = {}

    def add(kind, r, score, because=None, tag=None):
        key = (kind, r["id"])
        if key in seen or key in watchlist:
            return
        c = cand.setdefault(key, {"anchor": 0.0, "because": {}, "tags": set(), "src": set()})
        c["anchor"] += score
        c["src"].add(tag or "similar")
        if because:
            c["because"][because] = c["because"].get(because, 0) + score
        if tag:
            c["tags"].add(tag)

    # 1. TMDB recommendations / similar of their strongest titles
    for a in sorted(anchors, key=lambda a: -a["weight"])[:ANCHORS]:
        d = a["details"]
        for i, r in enumerate((d.get("recommendations") or {}).get("results", [])[:20]):
            add(a["kind"], r, a["weight"] * (1 - i / 40), a["title"])
        for i, r in enumerate((d.get("similar") or {}).get("results", [])[:10]):
            add(a["kind"], r, a["weight"] * 0.4 * (1 - i / 20), a["title"])

    # 2. true crime / local titles, only if that is >= 4 % of what they watch
    true_crime = [k["id"] for k in (tmdb.get("/search/keyword", ttl_days=90, query="true crime") or {})
                  .get("results", []) if k["name"].lower() == "true crime"]
    base = {"region": REGION, "sort_by": "popularity.desc", "vote_count.gte": 3}
    tc_w = sum(a["weight"] for a in pos if (true_crime and true_crime[0] in kw_ids(a["details"]))
               or {99, 80} <= {g["id"] for g in a["details"].get("genres", [])})
    likes_tc = tc_w / total_w >= 0.04
    likes_local = country_w.get(LOCAL_COUNTRY, 0) / total_w >= 0.04
    for kind in ("tv", "movie"):
        if true_crime and likes_tc:
            for page in (1, 2):
                for r in (tmdb.get(f"/discover/{kind}", ttl_days=3, with_keywords=true_crime[0], page=page,
                                   **base) or {}).get("results", []):
                    add(kind, r, 1.5, tag="truecrime")
        for page in (1, 2) if likes_local else ():
            for r in (tmdb.get(f"/discover/{kind}", ttl_days=3, with_origin_country=LOCAL_COUNTRY, page=page,
                               **base) or {}).get("results", []):
                add(kind, r, 1.0, tag="local")
    # 3. trending: candidates for everyone, so users with little history still get full rows
    for kind in ("tv", "movie"):
        for r in (tmdb.get(f"/trending/{kind}/week", ttl_days=2) or {}).get("results", []):
            add(kind, r, 0.8, tag="popular")
    # 4. films in their strongest genres (for series watchers, whose anchors bring few films)
    movie_genres = {28, 12, 35, 80, 99, 18, 10751, 14, 36, 27, 10402, 9648, 10749, 878, 10770, 53, 10752, 37}
    tv_to_movie = {10759: 28, 10765: 878, 10768: 10752}
    fav = []
    for g, _ in sorted(genre_w.items(), key=lambda kv: -kv[1]):
        g = tv_to_movie.get(g, g)
        if g in movie_genres and g not in fav and g != 99:
            fav.append(g)
    for g in fav[:3]:
        for page in (1, 2):
            for r in (tmdb.get("/discover/movie", ttl_days=3, with_genres=g, page=page, region=REGION,
                               sort_by="popularity.desc", **{"vote_count.gte": 150,
                                                             "primary_release_date.gte": "2015-01-01"}) or {}
                      ).get("results", []):
                add("movie", r, 1.2, tag="film")
    # 5. themes: discover on the most distinctive keywords of what they watched
    for k, w in sorted(kw_w.items(), key=lambda kv: -kv[1])[:6]:
        if w <= 0:
            continue
        for kind in ("tv", "movie"):
            for r in (tmdb.get(f"/discover/{kind}", ttl_days=3, with_keywords=k, sort_by="popularity.desc",
                               **{"vote_count.gte": 20}) or {}).get("results", []):
                add(kind, r, 1.0 * w / top_kw * 6, tag="themes")
    # 6. their own services: popular right now on the services in their history
    svc = sorted({x for a in anchors for x in a["item"]["services"]} & set(SERVICE_PROVIDERS))
    prov = "|".join(str(SERVICE_PROVIDERS[x]) for x in svc) or "|".join(map(str, DEFAULT_PROVIDERS))
    for kind in ("tv", "movie"):
        for page in (1, 2):
            for r in (tmdb.get(f"/discover/{kind}", ttl_days=3, with_watch_providers=prov, watch_region=REGION,
                               sort_by="popularity.desc", page=page, **{"vote_count.gte": 50}) or {}).get("results", []):
                add(kind, r, 0.9, tag="services")
    # 7. people: creators/directors and the two lead actors of their strongest titles, and their other work
    people = {}
    for a in sorted(pos, key=lambda a: -a["weight"])[:PEOPLE_ANCHORS]:
        d = a["details"]
        crew = [p["id"] for p in d.get("created_by", [])]
        cr = tmdb.get(f"/{a['kind']}/{a['id']}/{'aggregate_credits' if a['kind'] == 'tv' else 'credits'}", ttl_days=30) or {}
        crew += [p["id"] for p in cr.get("crew", []) if p.get("job") == "Director"][:1]
        for pid in crew + [p["id"] for p in cr.get("cast", [])[:2]]:
            people[pid] = max(people.get(pid, 0), a["weight"])
    for pid, w in people.items():
        works = tmdb.get(f"/person/{pid}/combined_credits", ttl_days=30) or {}
        by_id = {}
        for r in works.get("cast", []) + works.get("crew", []):
            if r.get("media_type") in ("tv", "movie") and (r.get("vote_count") or 0) >= 50 and r["id"] not in by_id:
                by_id[r["id"]] = r
        for r in sorted(by_id.values(), key=lambda r: -(r.get("popularity") or 0))[:10]:
            add(r["media_type"], r, 0.4 * w, tag="people")
    # 8. AI: free-form suggestions from the taste profile, resolved on TMDB
    if AI_CANDIDATES and openrouter_key() and len(pos) >= 3:
        try:
            for kind, r in ai_candidates(tmdb, pos, neg, prefs):
                add(kind, r, 1.5, tag="ai")
        except Exception as e:  # noqa: BLE001 - the other sources still fill the page
            log(f"ai candidates: {e}")
    src_count = collections.Counter(x for c in cand.values() for x in c["src"])
    log(f"candidates: {len(cand)} ({', '.join(f'{t} {n}' for t, n in sorted(src_count.items()))})")

    def taste(s):
        """0..1: how well a title fits their taste (keywords + genres, weighted by what they watched)."""
        return 0.6 * max(min(sum(kw_w.get(k, 0) for k in s["keywords"]) / top_kw, 1), 0) \
            + 0.4 * max(min(sum(genre_w.get(g, 0) for g in s["genres"]) / top_genre, 1), 0)

    max_anchor = max((c["anchor"] for c in cand.values()), default=1) or 1
    pool = []
    ranked = sorted(cand.items(), key=lambda kv: -kv[1]["anchor"])
    for (kind, tid), c in ranked[:POOL] + [kv for kv in ranked[POOL:] if kv[1]["tags"]]:
        d = tmdb.details(kind, tid)
        if not d:
            continue
        s = summary(kind, d)
        if skip(s):
            continue
        is_local = LOCAL_COUNTRY in s["countries"]
        if s["votes"] < (2 if is_local else 25) or (s["votes"] >= 25 and s["vote"] < 5.8):
            continue
        content = taste(s)
        year = int(s["year"] or 0)
        new = 1 if year >= NOW.year - 6 else 0.6 if year >= NOW.year - 16 else 0.3
        quality = s["vote"] / 10 if s["votes"] >= 20 else 0.55
        # how many of their titles point here, log-scaled against the strongest candidate; popular titles sit in
        # everyone's TMDB lists, so they are damped a little
        support = math.log1p(c["anchor"]) / math.log1p(max_anchor) / (1 + POP_DAMP * math.log10(1 + s["pop"]))
        s["score"] = 0.4 * support + 0.35 * content + 0.15 * quality + 0.1 * new
        s["parts"] = {"support": round(support, 3), "taste": round(content, 3), "quality": round(quality, 3), "new": new}
        s["because"] = max(c["because"], key=c["because"].get) if c["because"] else None
        s["because_all"] = c["because"]
        s["src"] = sorted(c["src"])
        s["truecrime"] = bool(likes_tc and ("truecrime" in c["tags"] or (true_crime and true_crime[0] in s["keywords"])
                                            or (99 in s["genres"] and 80 in s["genres"])))
        s["local"] = likes_local and is_local
        pool.append(s)
    pool.sort(key=lambda s: -s["score"])

    profile = {"top_titles": [f"{a['title']} ({a['kind']}, weight {a['weight']:.1f})"
                              for a in sorted(pos, key=lambda a: -a["weight"])[:25]],
               "persona": prefs.get("persona") or "someone",
               "not_for_them": [summary("tv" if "name" in d else "movie", d)["title"] for d, w in neg if w <= -1.5][:15]}
    return {"anchors": anchors, "pool": pool, "seen": seen, "unmatched": unmatched, "profile": profile,
            "src_count": dict(src_count), "cand_src": {k: sorted(c["src"]) for k, c in cand.items()},
            "pos": pos, "neg": neg, "prefs": prefs}


# ---------------------------------------------------------------- LLM (OpenRouter)

COST = {"usd": 0.0, "user": ""}


def openrouter_key():
    try:
        return OPENROUTER_KEY_FILE.read_text().strip() or None
    except OSError:
        return None


def openrouter_call(text, model=None, effort=None):
    """One JSON-answer call; the exact cost (usage.cost) goes to data/costs.csv."""
    model = model or LLM_MODEL
    effort = effort if effort is not None else (LLM_EFFORT if model == LLM_MODEL else None)
    r = requests.post("https://openrouter.ai/api/v1/chat/completions", timeout=180,
                      headers={"Authorization": f"Bearer {openrouter_key()}", "Content-Type": "application/json"},
                      json={"model": model, "temperature": 0.4, "usage": {"include": True},
                            "response_format": {"type": "json_object"},
                            **({"reasoning": {"effort": effort}} if effort else {}),
                            "messages": [{"role": "user", "content": text}]})
    if r.status_code != 200:
        raise RuntimeError(f"OpenRouter HTTP {r.status_code}: {r.text[:200]}")
    j = r.json()
    u = j.get("usage") or {}
    log_cost(model, u.get("prompt_tokens", 0), u.get("completion_tokens", 0), float(u.get("cost") or 0))
    content = j["choices"][0]["message"]["content"] or ""
    m = re.search(r"\{.*\}", content, re.S)
    return json.loads(m.group(0) if m else content)


def log_cost(model, tin, tout, usd):
    COST["usd"] += usd
    DATA.mkdir(parents=True, exist_ok=True)
    f = DATA / "costs.csv"
    new = not f.exists()
    with f.open("a") as fh:
        if new:
            fh.write("time,user,model,tokens_in,tokens_out,usd\n")
        fh.write(f"{dt.datetime.now():%Y-%m-%d %H:%M},{COST['user']},{model},{tin},{tout},{usd:.5f}\n")


def ai_candidates(tmdb, pos, neg, prefs):
    """~20 titles the LLM thinks fit (series and films), resolved via TMDB search; one cheap call per user/run."""
    top = [f"{a['title']} ({a['kind']})" for a in sorted(pos, key=lambda a: -a["weight"])[:30]]
    nope = [summary("tv" if "name" in d else "movie", d)["title"] for d, w in neg if w <= -1.5][:15]
    text = json.dumps({"task": f"You help {prefs.get('persona') or 'someone'} find new series and films. Name 20 titles "
                               "(series and films, lesser-known ones too) that fit what they watched and are not in "
                               "the list; nothing like 'not_for_them'. Answer as JSON: "
                               '{"titles": [{"title": "...", "year": 2020, "type": "tv"|"movie"}]}.',
                       "today": f"{NOW}", "what_they_watched": top, "not_for_them": nope}, ensure_ascii=False)
    out = []
    for t in (openrouter_call(text, CAND_MODEL, CAND_EFFORT).get("titles") or [])[:25]:
        kind = "tv" if t.get("type") == "tv" else "movie"
        q = {"query": t.get("title") or ""}
        if t.get("year"):
            q["first_air_date_year" if kind == "tv" else "year"] = t["year"]
        res = (tmdb.get(f"/search/{kind}", ttl_days=30, **q) or {}).get("results", [])
        # nothing released after NOW: evaluate.py replays an earlier date, and the LLM knows what came out since
        if res and (res[0].get("first_air_date") or res[0].get("release_date") or "9999") <= f"{NOW}":
            out.append((kind, res[0]))
    return out


SECTION_DEFS = [
    ("series", "Picked for you", lambda s: s["kind"] == "tv" and not s["local"] and not s["truecrime"]),
    ("films", "Films for a night in", lambda s: s["kind"] == "movie" and not s["local"] and not s["truecrime"]),
    ("truecrime", "True stories & true crime", lambda s: s["truecrime"] and not s["local"]),
    ("local", "Local", lambda s: s["local"]),
]


def row_lists(res, n=35):
    return {name: [s for s in res["pool"] if test(s)][:n] for name, _, test in SECTION_DEFS}


def llm_rows(profile, rows, reasons_only=True):
    """reasons_only: one short reason per given title, in every REASON_LANGS language. Else: the LLM picks 8-12
    titles per row from the candidates and writes the reasons. Returns {row: [{"id", "reason_<lang>", ...}]}."""
    langs = ", ".join(REASON_LANGS)
    keys = ", ".join(f'"reason_{lang}": "..."' for lang in REASON_LANGS)
    if reasons_only:
        task = (f"You help {profile['persona']} find new series and films, based on what they watched. For EVERY title "
                f"in every section, write one short, warm reason (max 90 characters) in each of these languages: {langs}; "
                "address them as 'you' and name a title they watched where possible. Use only ids from the sections. "
                f'Answer as JSON: {{"<section>": [{{"id": <id>, {keys}}}]}}.')
    else:
        task = (f"You help {profile['persona']} find new series and films, based on what they watched. Pick 8 to "
                f"{PER_ROW} titles per section from the candidates (best first, varied; fewer only if there really "
                f"aren't enough good ones) and write one short, warm reason per title (max 90 characters) in each of "
                f"these languages: {langs}; name a title they watched where possible. Nothing like 'not_for_them'. "
                f'Use only ids from the candidates. Answer as JSON: {{"<section>": [{{"id": <id>, {keys}}}]}}.')
    prompt = {"task": task, "what_they_watched_and_how_much": profile["top_titles"],
              "not_for_them": profile.get("not_for_them", []),
              "sections": {name: [{"id": s["id"], "title": s["title"], "year": s["year"], "genres": s["genre_names"],
                                   "because_they_watched": list(s["because_all"])[:3], "overview": s["overview"][:200]}
                                  for s in cands] for name, cands in rows.items() if cands}}
    return openrouter_call(json.dumps(prompt, ensure_ascii=False))


def sections(res):
    """The page: per row the best-scored titles (or the LLM's picks in pick mode), each with a reason."""
    rows = {name: lst[:PER_ROW] for name, lst in row_lists(res).items() if lst}
    by_id = {(name, s["id"]): s for name, lst in row_lists(res).items() for s in lst}
    out = {name: [dict(s) for s in lst] for name, lst in rows.items()}
    if not openrouter_key():
        return out
    try:
        if LLM_MODE == "pick":
            got = llm_rows(res["profile"], row_lists(res), reasons_only=False)
            out = {name: [dict(by_id[(name, p.get("id"))], **p) for p in got.get(name, []) if (name, p.get("id")) in by_id]
                   for name in rows}
        else:
            got = llm_rows(res["profile"], rows)
            for name, lst in out.items():
                reasons = {p.get("id"): p for p in got.get(name, []) if isinstance(p, dict)}
                for s in lst:
                    s.update({k: v for k, v in (reasons.get(s["id"]) or {}).items() if k.startswith("reason_")})
    except Exception as e:  # noqa: BLE001 - the page still works with "because you watched"
        log(f"llm: {e}")
    return out


# ---------------------------------------------------------------- output

def render(user, secs):
    lang = REASON_LANGS[0]
    rows = []
    for name, title, _ in SECTION_DEFS:
        cards = []
        for s in secs.get(name, []):
            reason = s.get(f"reason_{lang}") or (f"Because you watched {s['because']}" if s.get("because") else "")
            img = f'<img src="https://image.tmdb.org/t/p/w185{s["poster"]}" alt="" loading="lazy">' if s.get("poster") else ""
            cards.append(f'<a class="card" href="https://www.themoviedb.org/{s["kind"]}/{s["id"]}">{img}'
                         f'<b>{html.escape(s["title"])}</b> <span>{s["year"]}</span>'
                         f'<p>{html.escape(reason)}</p></a>')
        if cards:
            rows.append(f'<h2>{html.escape(title)}</h2><div class="row">{"".join(cards)}</div>')
    return f"""<!doctype html><html lang="{lang}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>For {html.escape(user)}</title>
<style>body{{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:0;padding:16px}}
.row{{display:flex;gap:12px;overflow-x:auto;padding-bottom:8px}}.card{{flex:0 0 150px;color:inherit;text-decoration:none}}
.card img{{width:150px;border-radius:6px}}.card p{{font-size:13px;color:#aaa;margin:4px 0}}span{{color:#888}}</style>
</head><body><h1>For {html.escape(user)}</h1>{"".join(rows)}<p><small>Data and images: TMDB.</small></p></body></html>"""


def run_user(tmdb, user):
    COST["user"] = user
    res = build(tmdb, HIST / user)
    secs = sections(res)
    out = DATA / user
    out.mkdir(parents=True, exist_ok=True)
    (out / "unmatched.txt").write_text("\n".join(res["unmatched"]) + "\n")
    (out / "recommendations.json").write_text(json.dumps(
        {"date": f"{NOW}", "sections": secs, "sources": res["src_count"]}, ensure_ascii=False, indent=1, default=list))
    (out / "index.html").write_text(render(user, secs))
    return sum(len(v) for v in secs.values())


def main():
    load_env()
    tmdb = Tmdb(os.environ["TMDB_API_KEY"])
    users = sys.argv[1:] or sorted(d.name for d in HIST.iterdir() if d.is_dir())
    try:
        for user in users:
            n = run_user(tmdb, user)
            log(f"{user}: {n} recommendations")
    finally:
        tmdb.save()
    log(f"TMDB calls {tmdb.calls}, LLM ${COST['usd']:.4f}")


if __name__ == "__main__":
    main()
