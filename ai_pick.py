"""AI moment picking (Google Gemini, free tier) and learning from past performance.

The AI never sees the video, only the transcript and loudness/chat signals of the
candidate moments. It scores each one and writes an English title and headline.
Needs a free key from Google AI Studio in the GEMINI_API_KEY secret.
"""
import base64
import json
import os
import re
import time
from datetime import datetime, timezone

import requests

DEFAULT_MODELS = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-flash-latest", "gemini-flash-lite-latest",
                  "gemini-3.1-flash-lite", "gemini-2.5-flash"]
PERF_URL_DEFAULT = "https://raw.githubusercontent.com/cooldolphine/clipper-app/main/performance.json"


def mmss(t):
    t = int(max(0, t))
    return f"{t // 3600}:{t // 60 % 60:02}:{t % 60:02}" if t >= 3600 else f"{t // 60}:{t % 60:02}"


def _parse_time(s):
    d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def performance_notes(url=None):
    """A short summary of what worked / did not work on the channel, for the AI prompt."""
    url = url or os.environ.get("PERF_URL") or PERF_URL_DEFAULT
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return ""
        vids = r.json().get("videos", [])
    except Exception:
        return ""
    now, rows = datetime.now(timezone.utc), []
    for v in vids:
        if v.get("views") is None or not v.get("published_at"):
            continue
        try:
            age = (now - _parse_time(v["published_at"])).total_seconds() / 86400
        except Exception:
            continue
        if age < 1:
            continue  # too new to judge
        rows.append((v["views"] / age, v, age))
    if len(rows) < 4:
        return ""
    rows.sort(key=lambda r: r[0], reverse=True)

    def line(row):
        _, v, age = row
        m = v.get("meta") or {}
        return (f'- "{str(v.get("title", ""))[:70]}" | AI virality {m.get("virality", "?")} | headline "{m.get("hook", "")}" | '
                f'source {m.get("channel", "?")} | {int(v["views"])} views and {v.get("likes", 0)} likes in {age:.0f} days')

    return ("What worked and what did not on this channel so far:\nBEST:\n" + "\n".join(line(r) for r in rows[:4]) +
            "\nWEAKEST:\n" + "\n".join(line(r) for r in rows[-3:]) +
            "\nLearn from this: favour moments like the BEST ones and avoid the patterns of the WEAKEST.")


def _prompt(cands, title, channel, num, want_broll, notes):
    blocks = []
    for c in cands:
        blocks.append(f'[{c["id"]}] {mmss(c["start"])}-{mmss(c["end"])} | sound/chat energy {c.get("energy", 0):.1f}\n'
                      f'{(c.get("text") or "")[:900] or "(no clear speech)"}')
    broll = (',"broll":[{"at":<seconds from the clip start>,"seconds":<2 or 3>,"query":"<2-3 English words for stock footage>"}]'
             if want_broll else "")
    return f"""You are the editor of a gaming Shorts channel. Source video: "{title}" by {channel}.
Below are {len(cands)} candidate moments from it. The speech may be Hindi, Hinglish or English; understand it.
Many candidates also come with 2 still frames from the video. Use them: judge what is on screen (action,
glitches, crashes, crowds, funny facecam reactions), not only what is said.
Pick the {num} most clip-worthy ones. Score EVERY candidate from 0 to 100 on:
- hook_score: would it grab attention in the first 3 seconds
- engagement: emotion, humour, surprise, chaos
- value: is it entertaining or useful on its own, without context
- shareability: would people share it or comment
Also give virality (0-100, your overall judgement), a short reason (max 20 words), "visual" (max 15 words on what is
on screen, or "" if you saw no frames), an English Shorts title
(max 70 characters, no hashtags) and an ALL CAPS English headline (max 32 characters) for the first 3 seconds.
{notes}

CANDIDATES:
{chr(10).join(blocks)}

Return ONLY JSON in exactly this shape:
{{"moments":[{{"id":<int>,"hook_score":<int>,"engagement":<int>,"value":<int>,"shareability":<int>,"virality":<int>,"reason":"<text>","visual":"<text>","title":"<text>","headline":"<TEXT>"{broll}}}]}}"""


def _clamp(v, lo=0, hi=100):
    try:
        return max(lo, min(hi, int(round(float(v)))))
    except Exception:
        return lo


def _parse(text, cands):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    data = json.loads(text)
    ids = {c["id"] for c in cands}
    out = {}
    for m in data.get("moments", []):
        try:
            i = int(m["id"])
        except Exception:
            continue
        if i not in ids:
            continue
        detail = {k: _clamp(m.get(k)) for k in ("hook_score", "engagement", "value", "shareability")}
        virality = _clamp(m.get("virality", sum(detail.values()) / 4))
        broll = []
        for b in (m.get("broll") or [])[:2]:
            q = str(b.get("query", "")).strip()
            if q:
                broll.append({"at": float(b.get("at", 2)), "seconds": float(b.get("seconds", 2)), "query": q[:60]})
        out[i] = {"virality": virality, "detail": detail, "reason": str(m.get("reason", ""))[:160],
                  "visual": str(m.get("visual", ""))[:120],
                  "title": re.sub(r"[<>#]", "", str(m.get("title", "")))[:100].strip(),
                  "headline": re.sub(r"[<>]", "", str(m.get("headline", ""))).upper()[:36].strip(), "broll": broll}
    return out or None


def _err(r):
    try:
        return str(r.json()["error"]["message"])[:300]
    except Exception:
        return str(getattr(r, "text", ""))[:300]


OPENROUTER_MODELS = ["google/gemini-3.5-flash", "google/gemini-3.5-flash-lite"]


def _gemini(parts, parse, label, models, key, timeout):
    env_model = (os.environ.get("GEMINI_MODEL") or "").strip()
    models = models or ([env_model] if env_model else []) + DEFAULT_MODELS
    body = {"contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.4}}
    for model in models:
        for attempt in range(2):
            try:
                r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                                  headers={"x-goog-api-key": key, "Content-Type": "application/json"}, json=body, timeout=timeout)
            except Exception as e:
                print(f"  {label}: {model}: network error or timeout ({type(e).__name__})")
                break  # a slow model: try the next one instead of waiting again
            if r.status_code == 429:
                msg = _err(r)
                print(f"  {label}: {model} is over its limit: {msg[:160]}")
                if attempt == 0 and "limit: 0" not in msg and "per day" not in msg.lower() and "retry" in msg.lower():
                    time.sleep(20)
                    continue
                break
            if r.status_code in (500, 502, 503):
                print(f"  {label}: {model} is busy ({r.status_code})")
                time.sleep(10)
                continue
            if r.status_code in (400, 404):
                print(f"  {label}: {model} not usable ({r.status_code}): {_err(r)[:160]}")
                break
            if r.status_code in (401, 403):
                print(f"  {label}: the Gemini key was rejected ({r.status_code}): {_err(r)[:160]}")
                return None
            if r.status_code != 200:
                print(f"  {label}: {model} error {r.status_code}: {_err(r)[:160]}")
                break
            try:
                res = parse(r.json()["candidates"][0]["content"]["parts"][0]["text"])
            except Exception as e:
                print(f"  {label}: {model} gave an answer I could not read ({type(e).__name__}: {e})")
                break
            if res:
                print(f"  {label}: done with {model} (Gemini)")
                return res
            print(f"  {label}: {model} answered, but with nothing usable")
            break
    return None


def _to_openrouter(parts):
    content = []
    for p in parts:
        if "text" in p:
            content.append({"type": "text", "text": p["text"]})
        elif "inline_data" in p:
            mime, data = p["inline_data"]["mime_type"], p["inline_data"]["data"]
            kind = "video_url" if mime.startswith("video") else "image_url"
            content.append({"type": kind, kind: {"url": f"data:{mime};base64,{data}"}})
    return content


def _openrouter(parts, parse, label, models, key, timeout):
    env_model = (os.environ.get("OPENROUTER_MODEL") or "").strip()
    models = models or ([env_model] if env_model else []) + OPENROUTER_MODELS
    body_base = {"messages": [{"role": "user", "content": _to_openrouter(parts)}],
                 "response_format": {"type": "json_object"}, "temperature": 0.4}
    for model in models:
        for attempt in range(2):
            try:
                r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                                  headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                           "X-Title": "Auto clipper"},
                                  json={**body_base, "model": model}, timeout=timeout)
            except Exception as e:
                print(f"  {label}: OpenRouter {model}: network error or timeout ({type(e).__name__})")
                break
            if r.status_code == 429:
                print(f"  {label}: OpenRouter {model} is over its limit: {_err(r)[:160]}")
                if attempt == 0:
                    time.sleep(20)
                    continue
                break
            if r.status_code == 402:
                print(f"  {label}: OpenRouter says you are out of credits: {_err(r)[:160]}")
                return None
            if r.status_code in (401, 403):
                print(f"  {label}: the OpenRouter key was rejected ({r.status_code}): {_err(r)[:160]}")
                return None
            if r.status_code in (500, 502, 503):
                print(f"  {label}: OpenRouter {model} is busy ({r.status_code})")
                time.sleep(10)
                continue
            if r.status_code != 200:
                print(f"  {label}: OpenRouter {model} not usable ({r.status_code}): {_err(r)[:160]}")
                break
            try:
                res = parse(r.json()["choices"][0]["message"]["content"])
            except Exception as e:
                print(f"  {label}: OpenRouter {model} gave an answer I could not read ({type(e).__name__}: {e})")
                break
            if res:
                print(f"  {label}: done with {model} (OpenRouter)")
                return res
            print(f"  {label}: OpenRouter {model} answered, but with nothing usable")
            break
    return None


def generate(parts, parse, label="AI", models=None, key=None, timeout=120):
    """Ask the AI and return parse(answer_text), or None if no AI is available.
    Tries Gemini first (if GEMINI_API_KEY is set), then OpenRouter (if OPENROUTER_API_KEY is set)."""
    gkey = key or os.environ.get("GEMINI_API_KEY")
    okey = os.environ.get("OPENROUTER_API_KEY")
    if not gkey and not okey:
        print(f"  {label}: no GEMINI_API_KEY or OPENROUTER_API_KEY secret, skipping")
        return None
    if gkey:
        res = _gemini(parts, parse, label, models, gkey, timeout)
        if res or not okey:
            return res
        print(f"  {label}: Gemini did not work, trying OpenRouter")
    return _openrouter(parts, parse, label, None, okey, timeout)


def rank(cands, title, channel, num, want_broll=False, notes="", models=None, key=None):
    """cands: [{"id", "start", "end", "text", "energy"}] -> {id: scores} or None if the AI is not available."""
    parts = [{"text": _prompt(cands, title, channel, num, want_broll, notes)}]
    for c in cands:
        for img in c.get("images") or []:
            parts.append({"text": f"Frame from candidate [{c['id']}]:"})
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(img).decode()}})
    return generate(parts, lambda t: _parse(t, cands), "AI picking", models, key)
