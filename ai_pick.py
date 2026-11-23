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

DEFAULT_MODELS = ["gemini-3.5-flash", "gemini-flash-latest", "gemini-2.5-flash"]
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


def generate(parts, parse, label="AI", models=None, key=None, timeout=120):
    """Send parts to Gemini and return parse(answer_text), or None if the AI is not available."""
    key = key or os.environ.get("GEMINI_API_KEY")
    if not key:
        print(f"  {label}: no GEMINI_API_KEY secret, skipping")
        return None
    env_model = os.environ.get("GEMINI_MODEL")
    models = models or ([env_model] if env_model else DEFAULT_MODELS)
    body = {"contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.4}}
    for model in models:
        for attempt in range(3):
            try:
                r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                                  headers={"x-goog-api-key": key, "Content-Type": "application/json"}, json=body, timeout=timeout)
            except Exception as e:
                print(f"  {label}: network error ({e})")
                time.sleep(5)
                continue
            if r.status_code in (429, 500, 502, 503):
                time.sleep(15 * (attempt + 1))
                continue
            if r.status_code in (400, 404):
                print(f"  {label}: model {model} not usable ({r.status_code}), trying the next one")
                break
            if r.status_code in (401, 403):
                print(f"  {label}: the Gemini key was rejected")
                return None
            if r.status_code != 200:
                print(f"  {label}: error {r.status_code}")
                break
            try:
                res = parse(r.json()["candidates"][0]["content"]["parts"][0]["text"])
                if res:
                    print(f"  {label}: done with {model}")
                    return res
            except Exception as e:
                print(f"  {label}: could not read the answer ({e})")
            break
    return None


def rank(cands, title, channel, num, want_broll=False, notes="", models=None, key=None):
    """cands: [{"id", "start", "end", "text", "energy"}] -> {id: scores} or None if the AI is not available."""
    parts = [{"text": _prompt(cands, title, channel, num, want_broll, notes)}]
    for c in cands:
        for img in c.get("images") or []:
            parts.append({"text": f"Frame from candidate [{c['id']}]:"})
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(img).decode()}})
    return generate(parts, lambda t: _parse(t, cands), "AI picking", models, key)
