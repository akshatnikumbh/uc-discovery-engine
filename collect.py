"""Step 1 to 2: collect public conversations about home services and clean them.

Run:  python collect.py
Out:  data/raw.csv, one row per record, each traceable to its source URL.

Sources
- Google Play reviews of Urban Company (com.urbanclap.urbanclap), pulled EQUALLY from each
  star rating 1 to 5. This is deliberate: public reviews over-represent angry customers,
  so a balanced pull lets the dashboard compare themes inside each star bucket.
- App Store reviews (id 1032480595), most recent pages of the India feed.
- Reddit: Indian city subreddits. Two kinds of thread:
    * threads that mention Urban Company
    * "need" threads (someone asking for a plumber, AC service, cleaner...) whether or not
      they mention Urban Company. These are how we see customers who never came back:
      they leave no review, but their needs still show up, answered by someone else.
- data/manual_sources.csv (optional): anything you collect by hand from Trustpilot,
  consumer complaint forums, YouTube comments or social posts. Columns:
  source,url,date,rating,text  (rating may be blank)

Reddit reliability: unauthenticated requests often get blocked. If you create a free
Reddit "script" app (reddit.com/prefs/apps) and set REDDIT_CLIENT_ID and
REDDIT_CLIENT_SECRET, the script uses the official API instead.
Every source prints how many rows it returned and any error, so a zero is never silent.
"""
import hashlib
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import pandas as pd
import requests

PLAY_ID = "com.urbanclap.urbanclap"
APPSTORE_ID = "1032480595"
PLAY_PER_STAR = int(os.environ.get("PLAY_PER_STAR", 300))
APPSTORE_PAGES = 10
SUBREDDITS = ["bangalore", "mumbai", "delhi", "gurgaon", "noida", "pune", "hyderabad",
              "chennai", "kolkata", "ahmedabad"]
UC_QUERIES = ["urban company", "urbanclap"]
NEED_QUERIES = ["plumber recommendation", "electrician recommendation", "AC service",
                "deep cleaning", "pest control", "carpenter", "salon at home"]
MAX_THREADS_WITH_COMMENTS = int(os.environ.get("MAX_THREADS", 150))
UA = "web:uc-repeat-research:0.2 (student product research)"
os.makedirs("data", exist_ok=True)
STATUS = []


def log(source, msg):
    STATUS.append(f"{source}: {msg}")
    print(f"{source}: {msg}")


def scrub(text: str) -> str:
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "[email]", str(text))
    text = re.sub(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", "[pan]", text)
    text = re.sub(r"(\+?91[\s-]?)?\b[6-9]\d{9}\b", "[phone]", text)
    text = re.sub(r"\b\d{11,18}\b", "[number]", text)
    return re.sub(r"\s+", " ", text).strip()


def rid(prefix, key):
    return f"{prefix}-{hashlib.sha1(str(key).encode()).hexdigest()[:8]}"


def hashed(name):
    return hashlib.sha1(name.encode()).hexdigest()[:10] if name else ""


# ---------------- Google Play ----------------
def play_store(rows):
    try:
        from google_play_scraper import Sort, reviews
    except ImportError:
        log("play_store", "google-play-scraper not installed, skipped")
        return
    for star in [1, 2, 3, 4, 5]:
        try:
            result, _ = reviews(PLAY_ID, lang="en", country="in", sort=Sort.NEWEST,
                                count=PLAY_PER_STAR, filter_score_with=star)
        except Exception as exc:
            log("play_store", f"{star}-star pull failed: {exc}")
            continue
        kept = 0
        for r in result:
            text = r.get("content") or ""
            if len(text) < 25:
                continue
            kept += 1
            rows.append({
                "record_id": rid("PS", r["reviewId"]), "source": "play_store",
                "url": f"https://play.google.com/store/apps/details?id={PLAY_ID}",
                "date": r["at"].date().isoformat() if r.get("at") else "",
                "rating": star, "author_hash": "", "thread": "", "text": scrub(text),
            })
        log("play_store", f"{star}-star: {kept} kept of {len(result)}")
        time.sleep(1)


# ---------------- App Store ----------------
def app_store(rows):
    kept = 0
    for page in range(1, APPSTORE_PAGES + 1):
        url = (f"https://itunes.apple.com/in/rss/customerreviews/page={page}/"
               f"id={APPSTORE_ID}/sortby=mostrecent/json")
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
            r.raise_for_status()
            entries = r.json().get("feed", {}).get("entry", [])
        except Exception as exc:
            log("app_store", f"page {page} failed: {exc}")
            break
        if isinstance(entries, dict):
            entries = [entries]
        if not entries:
            break
        for e in entries:
            if "content" not in e:
                continue
            text = f"{e['title']['label']}. {e['content']['label']}"
            if len(text) < 25:
                continue
            kept += 1
            rows.append({
                "record_id": rid("AS", e["id"]["label"]), "source": "app_store",
                "url": f"https://apps.apple.com/in/app/id{APPSTORE_ID}",
                "date": e.get("updated", {}).get("label", "")[:10],
                "rating": int(e.get("im:rating", {}).get("label", 0) or 0),
                "author_hash": "", "thread": "", "text": scrub(text),
            })
        time.sleep(1)
    log("app_store", f"{kept} kept")


# ---------------- Reddit ----------------
class Reddit:
    """Official API if credentials exist, else public JSON, else RSS (posts only)."""

    def __init__(self):
        self.token = None
        cid, secret = os.environ.get("REDDIT_CLIENT_ID"), os.environ.get("REDDIT_CLIENT_SECRET")
        if cid and secret:
            try:
                r = requests.post("https://www.reddit.com/api/v1/access_token",
                                  auth=(cid, secret), data={"grant_type": "client_credentials"},
                                  headers={"User-Agent": UA}, timeout=30)
                r.raise_for_status()
                self.token = r.json()["access_token"]
                log("reddit", "using official API (authenticated)")
            except Exception as exc:
                log("reddit", f"auth failed, falling back to public endpoints: {exc}")
        if not self.token:
            log("reddit", "no credentials, using public endpoints (may be rate limited)")

    def get(self, path, params):
        if self.token:
            url, headers = f"https://oauth.reddit.com{path}", {
                "User-Agent": UA, "Authorization": f"bearer {self.token}"}
            sleep = 1
        else:
            url, headers, sleep = f"https://www.reddit.com{path}.json", {"User-Agent": UA}, 3
        for attempt in range(3):
            r = requests.get(url, params=params, headers=headers, timeout=30)
            if r.status_code == 429:
                time.sleep(30 * (attempt + 1))
                continue
            time.sleep(sleep)
            if r.status_code != 200:
                return None, r.status_code
            return r.json(), 200
        return None, 429

    def search_rss(self, sub, q):
        r = requests.get(f"https://www.reddit.com/r/{sub}/search.rss",
                         params={"q": q, "restrict_sr": 1, "sort": "relevance", "t": "year"},
                         headers={"User-Agent": UA}, timeout=30)
        time.sleep(3)
        if r.status_code != 200:
            return [], r.status_code
        ns = {"a": "http://www.w3.org/2005/Atom"}
        out = []
        for entry in ET.fromstring(r.content).findall("a:entry", ns):
            content = entry.findtext("a:content", default="", namespaces=ns)
            content = re.sub(r"<[^>]+>", " ", content)
            link = entry.find("a:link", ns)
            out.append({"id": entry.findtext("a:id", default="", namespaces=ns),
                        "title": entry.findtext("a:title", default="", namespaces=ns),
                        "selftext": content, "permalink": link.get("href") if link is not None else "",
                        "created": entry.findtext("a:updated", default="", namespaces=ns)[:10],
                        "num_comments": 0, "author": ""})
        return out, 200


def reddit(rows):
    api = Reddit()
    seen, threads, errors = set(), [], {}
    for sub in SUBREDDITS:
        for q in UC_QUERIES + NEED_QUERIES:
            data, status = api.get(f"/r/{sub}/search", {"q": q, "restrict_sr": 1, "sort": "relevance",
                                                         "t": "year", "limit": 50})
            posts = []
            if data:
                for c in data.get("data", {}).get("children", []):
                    p = c["data"]
                    posts.append({"id": p["id"], "title": p.get("title", ""), "selftext": p.get("selftext", ""),
                                  "permalink": f"https://www.reddit.com{p['permalink']}",
                                  "created": datetime.fromtimestamp(p["created_utc"], timezone.utc).date().isoformat(),
                                  "num_comments": p.get("num_comments", 0), "author": p.get("author", ""),
                                  "api_path": p["permalink"]})
            else:
                errors[status] = errors.get(status, 0) + 1
                posts, st2 = api.search_rss(sub, q)
                if st2 != 200:
                    errors[f"rss {st2}"] = errors.get(f"rss {st2}", 0) + 1
            for p in posts:
                if p["id"] in seen:
                    continue
                seen.add(p["id"])
                kind = "uc_query" if q in UC_QUERIES else "need_query"
                rows.append({
                    "record_id": rid("RD", p["id"]), "source": f"reddit_post:{kind}",
                    "url": p["permalink"], "date": p["created"], "rating": "",
                    "author_hash": hashed(p["author"]), "thread": p["id"],
                    "text": scrub(f"[r/{sub}] {p['title']}. {p['selftext']}"),
                })
                if p.get("num_comments", 0) >= 3 and "api_path" in p:
                    threads.append(p)
        print(f"  r/{sub}: running total {len(seen)} posts")
    if errors:
        log("reddit", f"search errors by HTTP status: {errors}")
    threads = sorted(threads, key=lambda p: -p["num_comments"])[:MAX_THREADS_WITH_COMMENTS]
    comments = 0
    for p in threads:
        data, status = api.get(p["api_path"].rstrip("/"), {"limit": 100, "depth": 2})
        if not data or len(data) < 2:
            continue
        stack = list(data[1]["data"]["children"])
        while stack:
            c = stack.pop()
            if c.get("kind") != "t1":
                continue
            cd = c["data"]
            body = cd.get("body", "")
            if len(body) >= 40:
                comments += 1
                rows.append({
                    "record_id": rid("RC", cd["id"]), "source": "reddit_comment",
                    "url": f"https://www.reddit.com{cd.get('permalink', '')}",
                    "date": datetime.fromtimestamp(cd["created_utc"], timezone.utc).date().isoformat(),
                    "rating": "", "author_hash": hashed(cd.get("author", "")), "thread": p["id"],
                    "text": scrub(body),
                })
            replies = cd.get("replies")
            if isinstance(replies, dict):
                stack.extend(replies["data"]["children"])
    log("reddit", f"{len(seen)} posts, {comments} comments")


# ---------------- Manual sources ----------------
def manual(rows):
    path = "data/manual_sources.csv"
    if not os.path.exists(path):
        log("manual", "no data/manual_sources.csv, skipped")
        return
    df = pd.read_csv(path)
    for i, r in df.iterrows():
        text = str(r.get("text", ""))
        if len(text) < 25:
            continue
        rows.append({
            "record_id": rid("MN", f"{r.get('url', '')}{i}{text[:40]}"),
            "source": str(r.get("source", "manual")), "url": str(r.get("url", "")),
            "date": str(r.get("date", "")), "rating": r.get("rating", ""),
            "author_hash": "", "thread": "", "text": scrub(text),
        })
    log("manual", f"{len(df)} rows read")


def main():
    rows = []
    play_store(rows)
    app_store(rows)
    reddit(rows)
    manual(rows)
    df = pd.DataFrame(rows)
    if df.empty:
        print("\nNothing collected. Paste the lines above to get help.")
        return
    before = len(df)
    df["norm"] = df["text"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str[:200]
    df = df.drop_duplicates("norm").drop_duplicates("record_id").drop(columns="norm")
    df.to_csv("data/raw.csv", index=False)
    with open("data/collection_log.txt", "w") as f:
        f.write("\n".join(STATUS))
    print(f"\nSaved {len(df)} records ({before - len(df)} duplicates removed) to data/raw.csv")
    print(df["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
