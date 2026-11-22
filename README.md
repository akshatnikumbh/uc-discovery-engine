# Repeat Booking Discovery Engine (Urban Company)

Turns public conversations about home services into traceable behaviour records, scores
opportunity areas against the 90-day repeat-booking metric, corrects for the two biases in
public data, and checks itself against a hand-labelled golden set.

## Run it (inside your venv)

```bash
python3 -m venv venv && source venv/bin/activate     # skip if you already have one active
pip install -r requirements.txt
python collect.py                 # no key needed; prints a row count and any error per source
export GEMINI_API_KEY=your_key
python classify.py                # resumable; stops cleanly if the daily quota runs out
streamlit run app.py
```

Send back `data/raw.csv`, `data/classified.csv` and `data/collection_log.txt`.

## If Reddit returns zero rows
Create a free "script" app at reddit.com/prefs/apps, then before `python collect.py`:
`export REDDIT_CLIENT_ID=...` and `export REDDIT_CLIENT_SECRET=...`

## Adding sources by hand
Trustpilot, consumer complaint forums, YouTube comments, social posts: put them in
`data/manual_sources.csv` with columns `source,url,date,rating,text`, then rerun collect.py.
Copy only public text. Never anything behind a login.

## How the two biases are handled
- Angry reviews dominate public data: Play Store reviews are pulled in equal numbers per
  star rating, and the Opportunities score down-weights any reason that appears in only one lens.
- Customers who never came back leave no trace: the Silent churners tab looks at city-subreddit
  threads where people ask for a home service, and measures how often Urban Company is even named
  and what people use instead.

## Deploy (Streamlit Community Cloud)
Push to a public GitHub repo including the three data files (never the key). On
share.streamlit.io create an app from `app.py`; under Settings, Secrets add
`GEMINI_API_KEY = "your_key"`. Open the link logged out to confirm it works.
