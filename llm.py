"""One place that talks to an LLM, so switching provider is a one-line change.

Pick the provider with an environment variable (default is gemini):
    LLM_PROVIDER=gemini   needs GEMINI_API_KEY   (Google AI Studio free tier)
    LLM_PROVIDER=mistral  needs MISTRAL_API_KEY  (free tier is very slow, about 2 requests a minute)
Override the model with LLM_MODEL, for example LLM_MODEL=gemini-3.8-flash
"""
import os

PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").lower()
DEFAULT_MODELS = {"gemini": "gemini-3.5-flash-lite", "mistral": "mistral-small-latest"}
DEFAULT_SLEEP = {"gemini": 7, "mistral": 35}   # seconds between calls that stay under the free limits
# gemini-2.5-flash and gemini-2.5-pro are blocked for new API keys (Google routes new
# projects to the 3.x family). gemini-3.8-flash exists too but its free daily quota is
# reportedly far lower than Flash-Lite's; Flash-Lite is the right default for a
# high-volume classification job like this one. Confirm current limits in your own
# project's quota console, Google doesn't publish exact numbers and they do change.

if PROVIDER not in DEFAULT_MODELS:
    raise SystemExit(f"LLM_PROVIDER must be one of {list(DEFAULT_MODELS)}, got '{PROVIDER}'")

MODEL_NAME = os.environ.get("LLM_MODEL", DEFAULT_MODELS[PROVIDER])
_client = None


def _get_client():
    global _client
    if _client is None:
        if PROVIDER == "gemini":
            from google import genai
            _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        else:
            from mistralai import Mistral
            _client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    return _client


def complete(prompt: str, json_mode: bool = False) -> str:
    """Send one prompt, return the model's text. json_mode asks for a single JSON object."""
    client = _get_client()
    if PROVIDER == "gemini":
        from google.genai import types
        config = types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json" if json_mode else None,
        )
        resp = client.models.generate_content(model=MODEL_NAME, contents=prompt, config=config)
        return resp.text or ""
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = client.chat.complete(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        **kwargs,
    )
    return resp.choices[0].message.content or ""
