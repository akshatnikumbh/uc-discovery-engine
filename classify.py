"""Steps 3 to 4: turn each record into a structured behaviour record with an LLM.

Run:  GEMINI_API_KEY=... python classify.py
Out:  data/classified.csv  (resumable: re-running skips records already done)

Checks built in:
- every label is validated against taxonomy.py; anything else becomes "unclear" or "none"
- evidence_quote must appear verbatim in the source text, else quote_verified = False
- a daily-quota error stops cleanly with progress saved
Optional: MAX_RECORDS=600 python classify.py labels a sample stratified by source.
"""
import json
import os
import re
import time

import pandas as pd

import llm
from taxonomy import (ACQUISITION, ALTERNATIVES, CATEGORIES, NEED_TRIGGERS, NOT_RETURN_REASONS,
                      POST_SERVICE_EXPECTATIONS, RECORD_TYPES, RETURN_SIGNALS)

BATCH = int(os.environ.get("BATCH_SIZE", 15))
SLEEP = int(os.environ.get("SLEEP_SECONDS", llm.DEFAULT_SLEEP[llm.PROVIDER]))
MAX_RECORDS = os.environ.get("MAX_RECORDS")
OUT = "data/classified.csv"

PROMPT = f"""You label public posts about home services in India (Urban Company and alternatives)
for product research on why customers do or do not book again after a completed service.
Return exactly ONE JSON object, no prose: {{"items": [ ... ]}} with one object per record, same order.

Fields per item:
- record_id: copy exactly
- relevant: true if the text is about getting a home service done (cleaning, repairs, AC, pest
  control, salon or spa at home, painting) or about Urban Company as a service provider
- record_type: one of {RECORD_TYPES}. need_post = the writer is asking how or where to get a
  home service done.
- personal_experience: true if the writer describes something that happened to them
- anchor_category: the main service discussed, one of {CATEGORIES}
- categories_mentioned: list from {CATEGORIES}
- mentions_uc: true if Urban Company or UrbanClap is named
- return_signal: one of {RETURN_SIGNALS}. Only what the text states.
- not_return_reasons: list from {NOT_RETURN_REASONS}, why they did not book UC again or chose
  something else. Empty list if not stated.
- need_triggers: list from {NEED_TRIGGERS}, what made the need arise
- alternatives_used: list from {ALTERNATIVES}, what they used or recommend instead of UC
- post_service_expectations: list from {POST_SERVICE_EXPECTATIONS}, what they expected after
  the job was done, only if stated
- acquisition_cue: one of {ACQUISITION}, how they came to UC, only if stated
- household_cues: household type, renting or owning, domestic help, ONLY if stated, else "unknown"
- city_cue: city ONLY if stated or the subreddit name shows it, else "unknown"
- theme: 2 to 5 words in your own words naming the main point
- evidence_quote: an exact span copied from the text, at most 25 words, that supports the labels
- confidence: "high", "medium" or "low"

Rules (v2 and v3, written after the golden-set reviews; see the Evidence Log):
1. return_signal "none" = the text does not talk about booking again, stopping, or using
   something else. Most single reviews are "none". "unclear" = the text DOES talk about it,
   but the outcome is ambiguous. Never use "unclear" just because the text is silent.
2. A complaint is not churn. Use "did_not_return" only if the writer says they stopped, will
   not book again, or deleted the app. Use "switched_to_alternative" only if they say they
   used or now use something else instead. Otherwise "none". Telling others to avoid Urban
   Company, or calling it a bad decision, is a complaint, not churn: "none".
3. not_return_reasons: only reasons the writer gives for not booking again, choosing something
   else, or saying they will stop. A complaint with no stated consequence gets an empty list.
   Do not add a reason that is not written in the text.
4. record_type (v3): "need_post" = the writer has a service need that is not met yet: asking
   where or how to get it, saying it is not available in their area, or asking UC to add it.
   This holds even if they mention a past booking. "service_experience" only if the text
   describes a specific booking that happened (a visit, a professional, a date, a job).
   Short verdicts with no specific booking ("useless app", "workers should be faster",
   "add more options") are "general_opinion".
5. Text starting with [r/name] comes from that subreddit; use it for city_cue only if the
   subreddit is a city.

Do not guess what is not in the text.

Records:
"""


def one(value, allowed, default):
    return value if value in allowed else default


def many(values, allowed):
    if not isinstance(values, list):
        values = [values]
    kept = [v for v in values if v in allowed]
    return "|".join(kept) if kept else "none"


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def label_batch(batch):
    payload = [{"record_id": r.record_id, "text": r.text[:2000]} for r in batch.itertuples()]
    for attempt in range(4):
        try:
            text = llm.complete(PROMPT + json.dumps(payload, ensure_ascii=False), json_mode=True)
            text = re.sub(r"^```(?:json)?|```$", "", text.strip()).strip()
            parsed = json.loads(text)
            items = parsed.get("items", parsed) if isinstance(parsed, dict) else parsed
            return items if isinstance(items, list) else [items]
        except Exception as exc:
            msg = str(exc)
            if "perday" in msg.lower().replace(" ", ""):
                print(f"\nDaily quota used up: {msg[:200]}\nProgress is saved; rerun after the reset.")
                raise SystemExit(1)
            wait = 20 * (attempt + 1)
            print(f"  call failed ({msg[:160]}); retrying in {wait}s")
            time.sleep(wait)
    print("  giving up on this batch; it will be retried on the next run")
    return []


def pick_sample(df, n):
    per = max(1, n // df["source"].nunique())
    sample = df.groupby("source", group_keys=False).apply(lambda g: g.sample(min(len(g), per), random_state=7))
    if len(sample) < n:
        rest = df.drop(sample.index)
        sample = pd.concat([sample, rest.sample(min(n - len(sample), len(rest)), random_state=7)])
    return sample


def main():
    raw = pd.read_csv("data/raw.csv")
    raw["text"] = raw["text"].fillna("").astype(str)
    if MAX_RECORDS:
        raw = pick_sample(raw, int(MAX_RECORDS))
        print(f"MAX_RECORDS: labelling a source-stratified sample of {len(raw)}. Note it in the Evidence Log.")
    done = set(pd.read_csv(OUT)["record_id"]) if os.path.exists(OUT) else set()
    todo = raw[~raw["record_id"].isin(done)]
    texts = dict(zip(raw["record_id"], raw["text"]))
    calls = -(-len(todo) // BATCH)
    print(f"{len(todo)} to label with {llm.PROVIDER}:{llm.MODEL_NAME}, {len(done)} done already")
    print(f"{calls} calls, about {round(calls * (SLEEP + 4) / 60, 1)} minutes")

    for start in range(0, len(todo), BATCH):
        out = []
        for item in label_batch(todo.iloc[start:start + BATCH]):
            if not isinstance(item, dict) or item.get("record_id") not in texts:
                continue
            r = item["record_id"]
            quote = str(item.get("evidence_quote", ""))
            out.append({
                "record_id": r,
                "relevant": bool(item.get("relevant", False)),
                "record_type": one(item.get("record_type"), RECORD_TYPES, "irrelevant"),
                "personal_experience": bool(item.get("personal_experience", False)),
                "anchor_category": one(item.get("anchor_category"), CATEGORIES, "unclear"),
                "categories_mentioned": many(item.get("categories_mentioned", []), CATEGORIES),
                "mentions_uc": bool(item.get("mentions_uc", False)),
                "return_signal": one(item.get("return_signal"), RETURN_SIGNALS, "unclear"),
                "not_return_reasons": many(item.get("not_return_reasons", []), NOT_RETURN_REASONS),
                "need_triggers": many(item.get("need_triggers", []), NEED_TRIGGERS),
                "alternatives_used": many(item.get("alternatives_used", []), ALTERNATIVES),
                "post_service_expectations": many(item.get("post_service_expectations", []), POST_SERVICE_EXPECTATIONS),
                "acquisition_cue": one(item.get("acquisition_cue"), ACQUISITION, "unknown"),
                "household_cues": str(item.get("household_cues", "unknown"))[:150],
                "city_cue": str(item.get("city_cue", "unknown"))[:40],
                "theme": str(item.get("theme", ""))[:60],
                "evidence_quote": quote[:300],
                "quote_verified": bool(quote) and norm(quote) in norm(texts[r]),
                "confidence": one(item.get("confidence"), {"high", "medium", "low"}, "low"),
                "model": f"{llm.PROVIDER}:{llm.MODEL_NAME}",
                "prompt_version": "v3",
            })
        if out:
            pd.DataFrame(out).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
        print(f"  {min(start + BATCH, len(todo))}/{len(todo)} processed, {len(out)} labelled")
        time.sleep(SLEEP)

    df = pd.read_csv(OUT)
    print(f"\nDone: {len(df)} labelled, {int(df['relevant'].sum())} relevant, "
          f"{int((~df['quote_verified']).sum())} quotes failed the verbatim check")


if __name__ == "__main__":
    main()
