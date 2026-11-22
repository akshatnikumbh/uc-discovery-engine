"""Steps 5 to 6: score opportunities, correct for the two biases, serve the evidence.

Run locally:  streamlit run app.py
Deploy:       Streamlit Community Cloud, with GEMINI_API_KEY in the app's Secrets.
"""
import math
import os
import re

import pandas as pd
import streamlit as st

from taxonomy import (ADDRESSABILITY, ALTERNATIVES, CATEGORIES, NON_RETURN_SIGNALS,
                      NOT_RETURN_REASONS, RECORD_TYPES, RETURN_SIGNALS)

st.set_page_config(page_title="Repeat Booking Discovery Engine", layout="wide")
try:
    for _k in ("GEMINI_API_KEY", "MISTRAL_API_KEY", "LLM_PROVIDER", "LLM_MODEL"):
        if _k in st.secrets:
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass

BOOL_COLS = ["relevant", "personal_experience", "mentions_uc", "quote_verified"]
LIST_COLS = ["categories_mentioned", "not_return_reasons", "need_triggers", "alternatives_used",
             "post_service_expectations"]


@st.cache_data
def load():
    raw = pd.read_csv("data/raw.csv")
    lab = pd.read_csv("data/classified.csv")
    df = raw.merge(lab, on="record_id", how="inner")
    for c in BOOL_COLS:
        df[c] = df[c].astype(str).str.lower().eq("true")
    for c in LIST_COLS:
        df[c] = df[c].fillna("none").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["family"] = df["source"].str.split(":").str[0].replace({"reddit_comment": "reddit_post"})
    df["lens"] = "other"
    df.loc[df["rating"].between(1, 2), "lens"] = "review 1-2 star"
    df.loc[df["rating"] == 3, "lens"] = "review 3 star"
    df.loc[df["rating"].between(4, 5), "lens"] = "review 4-5 star"
    df.loc[df["family"].str.startswith("reddit"), "lens"] = "reddit"
    return raw, df


def explode(frame, col):
    out = frame.assign(_v=frame[col].str.split("|")).explode("_v").reset_index(drop=True)
    return out[out["_v"] != "none"]


raw, df = load()
rel = df[df["relevant"]]

st.title("Repeat Booking Discovery Engine")
st.caption("Public conversations about home services, turned into traceable behaviour records. "
           "Counts show what people talk about, not how common something is among all customers.")

tabs = st.tabs(["Overview", "Opportunities", "Silent churners", "Segments", "Evidence",
                "Golden set", "Ask the corpus", "Company context"])

# ---------------- Overview ----------------
with tabs[0]:
    c = st.columns(5)
    c[0].metric("Collected", len(raw))
    c[1].metric("Labelled", len(df))
    c[2].metric("Relevant", int(df["relevant"].sum()))
    c[3].metric("Need posts", int((rel["record_type"] == "need_post").sum()))
    c[4].metric("Mention Urban Company", int(rel["mentions_uc"].sum()))
    left, right = st.columns(2)
    with left:
        st.subheader("Records by source")
        st.bar_chart(df["source"].value_counts())
    with right:
        st.subheader("Review star ratings collected")
        st.bar_chart(df["rating"].dropna().astype(int).value_counts().sort_index())
        st.caption("Play Store reviews are pulled in equal numbers per star rating on purpose, so "
                   "themes can be compared inside each star bucket instead of letting angry reviews dominate.")
    st.subheader("Quality signals")
    q = st.columns(3)
    q[0].metric("Quotes verified verbatim", f"{df['quote_verified'].mean():.0%}")
    q[1].metric("High-confidence labels", f"{(df['confidence'] == 'high').mean():.0%}")
    q[2].metric("City stated (not guessed)", f"{(rel['city_cue'] != 'unknown').mean():.0%}")

# ---------------- Opportunities ----------------
with tabs[1]:
    st.markdown(
        "**Score = √records × addressability × robustness.** "
        "*Addressability* (0 to 1) is how far a customer-side change with no discount, membership, "
        "price change or change to professionals could act on the reason. Price scores 0. "
        "*Robustness* is the share of the four lenses (1-2 star, 3 star, 4-5 star reviews, Reddit) in which "
        "the reason appears at least twice. A reason that only shows up in angry reviews is down-weighted: "
        "that is the correction for negativity bias.")
    ex = explode(rel, "not_return_reasons")
    lenses = ["review 1-2 star", "review 3 star", "review 4-5 star", "reddit"]
    rows = []
    for reason, g in ex.groupby("_v"):
        present = sum((g["lens"] == l).sum() >= 2 for l in lenses)
        robust = present / len(lenses)
        addr = ADDRESSABILITY.get(reason, 0.2)
        rows.append({
            "reason": reason, "records": len(g),
            **{l: int((g["lens"] == l).sum()) for l in lenses},
            "robustness": round(robust, 2), "addressability": addr,
            "score": round(math.sqrt(len(g)) * addr * robust, 2),
        })
    opp = pd.DataFrame(rows).sort_values("score", ascending=False) if rows else pd.DataFrame()
    st.dataframe(opp, width="stretch", hide_index=True)
    st.subheader("Star-balanced view")
    st.caption("Share of records in each lens that cite the reason. Read across a row: a reason with a "
               "similar share in 4-5 star reviews and on Reddit is not just complaint noise.")
    if not ex.empty:
        base = rel.groupby("lens").size()
        share = (ex.groupby(["_v", "lens"]).size() / base).unstack(fill_value=0).round(3)
        st.dataframe(share.reindex(columns=[l for l in lenses if l in share.columns]), width="stretch")
    st.subheader("What made the need arise")
    st.bar_chart(explode(rel, "need_triggers")["_v"].value_counts())
    st.subheader("What customers expected after the job")
    st.bar_chart(explode(rel, "post_service_expectations")["_v"].value_counts())

# ---------------- Silent churners ----------------
with tabs[2]:
    st.markdown(
        "Customers who never came back leave no review. Their **needs** still show up: someone asking their "
        "city subreddit for a plumber, an AC service or a cleaner. This tab measures how often Urban Company "
        "is even part of those conversations, and what people use instead.")
    need = rel[rel["source"].str.contains("need_query") | (rel["record_type"] == "need_post")]
    threads = set(need["thread"].dropna().astype(str)) - {""}
    convo = rel[rel["thread"].astype(str).isin(threads)]
    m = st.columns(3)
    m[0].metric("Need threads", len(threads))
    m[1].metric("Posts and replies in them", len(convo))
    m[2].metric("Of those, naming Urban Company", f"{convo['mentions_uc'].mean():.0%}" if len(convo) else "n/a")
    if len(convo):
        st.subheader("What people recommend or use instead")
        st.bar_chart(explode(convo, "alternatives_used")["_v"].value_counts())
        st.subheader("By category: how often Urban Company is named in need threads")
        cat = convo.groupby("anchor_category").agg(records=("record_id", "count"),
                                                   names_uc=("mentions_uc", "mean")).round(2)
        st.dataframe(cat.sort_values("records", ascending=False), width="stretch")
    st.caption("Limitation: Reddit users skew young, urban and English-speaking. Read these as directional.")

# ---------------- Segments ----------------
with tabs[3]:
    dim = st.selectbox("Compare by", ["anchor_category", "city_cue", "acquisition_cue", "lens"])
    seg = rel[rel[dim] != "unknown"]
    tab = pd.crosstab(seg[dim], seg["return_signal"])
    st.dataframe(tab, width="stretch")
    exs = explode(seg, "not_return_reasons")
    if not exs.empty:
        st.subheader(f"Top reasons by {dim}")
        top = exs.groupby([dim, "_v"]).size().reset_index(name="records")
        top = top.sort_values(["records"], ascending=False).groupby(dim).head(3)
        st.dataframe(top, width="stretch", hide_index=True)
    st.subheader("Household cues stated in the text")
    st.dataframe(rel[rel["household_cues"] != "unknown"]["household_cues"].value_counts().head(20),
                 width="stretch")

# ---------------- Evidence ----------------
with tabs[4]:
    f = st.columns(4)
    reason = f[0].selectbox("Reason", ["all"] + NOT_RETURN_REASONS)
    sig = f[1].selectbox("Return signal", ["all"] + RETURN_SIGNALS)
    rtype = f[2].selectbox("Record type", ["all"] + RECORD_TYPES)
    lens = f[3].selectbox("Lens", ["all", "review 1-2 star", "review 3 star", "review 4-5 star", "reddit"])
    v = rel
    if reason != "all":
        v = v[v["not_return_reasons"].str.contains(reason, regex=False)]
    if sig != "all":
        v = v[v["return_signal"] == sig]
    if rtype != "all":
        v = v[v["record_type"] == rtype]
    if lens != "all":
        v = v[v["lens"] == lens]
    st.write(f"{len(v)} records")
    st.dataframe(v[["record_id", "source", "rating", "date", "record_type", "return_signal",
                    "not_return_reasons", "alternatives_used", "evidence_quote", "quote_verified", "url"]],
                 width="stretch", hide_index=True,
                 column_config={"url": st.column_config.LinkColumn("source link")})

# ---------------- Golden set ----------------
GOLD_COLS = {"human_relevant": None, "human_record_type": RECORD_TYPES,
             "human_return_signal": RETURN_SIGNALS, "human_primary_reason": NOT_RETURN_REASONS}
with tabs[5]:
    st.markdown("**Step 1.** Download a blind sample of 20 (AI labels hidden). **Step 2.** Fill the four "
                "human columns using the exact codes listed below. **Step 3.** Upload to see agreement.")
    with st.expander("Valid codes"):
        st.write("human_relevant: True or False")
        st.write("human_record_type:", RECORD_TYPES)
        st.write("human_return_signal:", RETURN_SIGNALS)
        st.write("human_primary_reason:", NOT_RETURN_REASONS)
    picks = []
    for lens_name in ["review 1-2 star", "review 3 star", "review 4-5 star", "reddit"]:
        pool = df[df["lens"] == lens_name]
        picks.append(pool.sample(min(len(pool), 5), random_state=11))
    sample = pd.concat(picks).head(20)
    blind = sample[["record_id", "source", "rating", "url", "text"]].assign(**{k: "" for k in GOLD_COLS}, notes="")
    st.download_button("Download blind golden sample (CSV)", blind.to_csv(index=False),
                       "golden_blind.csv", "text/csv")
    up = st.file_uploader("Upload your labelled golden set", type="csv")
    if up is not None:
        try:
            gold = pd.read_csv(up)
        except UnicodeDecodeError:
            up.seek(0)
            gold = pd.read_csv(up, encoding="cp1252")  # files re-saved through Excel
        problems = {}
        for col, allowed in GOLD_COLS.items():
            if col not in gold.columns:
                problems[col] = "column missing"
            elif allowed:
                bad = sorted(set(gold[col].dropna().astype(str).str.strip()) - set(allowed))
                if bad:
                    problems[col] = bad
        if problems:
            st.error("This file's labels are not in code format, or a column is missing. "
                     "Agreement cannot be computed. You may have uploaded an older or reworded copy.")
            st.write(problems)
            st.stop()
        g = gold.merge(df[["record_id", "relevant", "record_type", "return_signal", "not_return_reasons"]],
                       on="record_id", how="left")
        g["h_rel"] = g["human_relevant"].astype(str).str.strip().str.lower().isin(["true", "yes", "1", "y"])
        g["ok_rel"] = g["h_rel"] == g["relevant"]
        g["ok_type"] = g["human_record_type"].astype(str).str.strip() == g["record_type"]
        g["ok_ret"] = g["human_return_signal"].astype(str).str.strip() == g["return_signal"]
        g["ok_reason"] = [str(h).strip() in str(a).split("|") or (str(h).strip() == "none" and str(a) == "none")
                          for h, a in zip(g["human_primary_reason"], g["not_return_reasons"])]
        tp = (g["h_rel"] & g["relevant"]).sum()
        fp = (~g["h_rel"] & g["relevant"]).sum()
        fn = (g["h_rel"] & ~g["relevant"]).sum()
        m = st.columns(6)
        m[0].metric("Items", len(g))
        m[1].metric("Relevance", f"{g['ok_rel'].sum()}/{len(g)}")
        m[2].metric("Precision / recall", f"{tp / max(tp + fp, 1):.0%} / {tp / max(tp + fn, 1):.0%}")
        m[3].metric("Record type", f"{g['ok_type'].sum()}/{len(g)}")
        m[4].metric("Return signal", f"{g['ok_ret'].sum()}/{len(g)}")
        m[5].metric("Primary reason found", f"{g['ok_reason'].sum()}/{len(g)}")
        st.caption("With about 20 items, one disagreement moves a rate by 5 points. Report counts, not just percentages.")
        st.subheader("Disagreements: candidates for the Evidence Log corrections")
        bad = g[~(g["ok_rel"] & g["ok_type"] & g["ok_ret"] & g["ok_reason"])]
        st.dataframe(bad[["record_id", "text", "human_relevant", "relevant", "human_record_type", "record_type",
                          "human_return_signal", "return_signal", "human_primary_reason", "not_return_reasons",
                          "notes"]], width="stretch", hide_index=True)

# ---------------- Ask the corpus ----------------
with tabs[6]:
    st.markdown("Answers come only from the labelled records and cite their IDs. "
                "If the records do not support an answer, it says so.")
    question = st.text_input("Ask about the evidence, e.g. what do people use instead of Urban Company for small repairs?")
    if question:
        words = [w for w in re.findall(r"[a-z]+", question.lower()) if len(w) > 3]
        scored = rel.assign(hits=rel["text"].str.lower().apply(lambda t: sum(w in t for w in words)))
        top = scored[scored["hits"] > 0].sort_values("hits", ascending=False).head(25)
        if len(top) < 5:
            st.warning("Not enough matching records to answer reliably. Try other words or the Evidence tab.")
        else:
            context = "\n".join(f"[{r.record_id}] {r.text[:600]}" for r in top.itertuples())
            prompt = ("Answer using ONLY the records below. Cite record IDs in square brackets after each claim "
                      "and say how many records support each point. If the records do not support an answer, "
                      "say so plainly.\n\n"
                      f"Question: {question}\n\nRecords:\n{context}")
            answer = ""
            try:
                import llm
                answer = llm.complete(prompt)
            except KeyError as exc:
                st.error(f"Missing secret or environment variable: {exc}")
            except Exception as exc:
                st.error(f"The model call failed: {str(exc)[:200]}")
            if answer:
                cited = set(re.findall(r"\[([A-Z]{2}-[0-9a-f]{8})\]", answer))
                unknown = cited - set(top["record_id"])
                st.write(answer)
                if unknown:
                    st.error(f"Check failed: cites IDs that were not provided: {', '.join(sorted(unknown))}")
                else:
                    st.success(f"Check passed: all {len(cited)} cited IDs exist in the retrieved records.")
                with st.expander("Records used"):
                    st.dataframe(top[["record_id", "source", "evidence_quote", "url"]], hide_index=True)

# ---------------- Company context ----------------
with tabs[7]:
    if os.path.exists("context_facts.md"):
        st.markdown(open("context_facts.md", encoding="utf-8").read())
