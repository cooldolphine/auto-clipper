"""The 'director pass': Gemini watches a short low-quality preview of each finalist (picture AND
sound) and returns the exact start/end, the funny peak, edit cues, and a title, description and
tags written for that exact clip."""
import base64
import re
import subprocess

import ai_pick

MAX_CLIP, MIN_CLIP = 59.0, 8.0


def make_preview(src, out, start, dur, headers=None, audio_wav=None, height=360):
    """A small mp4 (picture + sound) of src[start:start+dur]. Returns bytes or None."""
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.2f}", "-t", f"{dur:.2f}"]
    if headers:
        cmd += ["-headers", "".join(f"{k}: {v}\r\n" for k, v in headers.items())]
    cmd += ["-i", str(src)]
    if audio_wav:  # the stream we cut from has no sound (video-only), so use the audio we already have
        cmd += ["-ss", f"{start:.2f}", "-t", f"{dur:.2f}", "-i", str(audio_wav), "-map", "0:v:0", "-map", "1:a:0"]
    else:
        cmd += ["-map", "0:v:0", "-map", "0:a:0?"]
    cmd += ["-vf", f"scale=-2:{height},fps=12", "-c:v", "libx264", "-preset", "veryfast", "-crf", "33",
            "-c:a", "aac", "-b:a", "48k", "-ac", "1", "-movflags", "+faststart", "-shortest", str(out)]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=240)
        if r.returncode != 0:
            return None
        return open(out, "rb").read()
    except Exception:
        return None


def _prompt(batch, title, channel, want_broll, notes):
    blocks = []
    for f in batch:
        blocks.append(f'--- CLIP [{f["id"]}] (preview length {f["dur"]:.0f}s; all times below are seconds from the start of this preview)\n'
                      f'Speech with times:\n{f.get("lines") or "(no clear speech)"}')
    broll = (',"broll":[{"at":<sec>,"seconds":<2 or 3>,"query":"<2-3 English words for stock footage>"}]' if want_broll else "")
    return f"""You are the editor of a gaming channel that posts funny YouTube Shorts. Source: "{title}" by {channel}.
You get {len(batch)} preview videos (picture and sound, low quality). The speech may be Hindi, Hinglish or English.
For EACH clip decide how to cut the single funniest moment EXACTLY:
- keep: false if there is nothing funny or interesting in it.
- start / end: the cut. Start 1 second before the setup begins, end 1 second after the reaction or laughter finishes.
  The clip must be between 8 and 55 seconds long. Never cut a sentence in half.
- peak: the second of the punchline or the biggest funny event.
- zoom: {{"at": sec, "seconds": 1-2, "target": "face" or "gameplay"}} to punch in on the funny moment, or null.
- slowmo: {{"at": sec, "seconds": 1-2}} for a slow-motion replay feel on the key moment, or null.
- emphasis: up to 6 spoken words (exactly as written in the speech lines) that deserve to be bigger and highlighted.
- emoji: up to 3 {{"at": sec, "emoji": "<one emoji>"}} that match what is happening.
- funny and virality: 0 to 100.
- game: the game name if you can tell, else "".
- reason (max 20 words) and visual (max 15 words: what is on screen).
- headline: ALL CAPS English, max 4 words, for the first 3 seconds.
- titles: 5 different English YouTube Shorts titles for THIS clip (max 60 characters, no hashtags, no lies).
  Use different styles: curiosity, a quote, the reaction, the surprise, a short statement.
- description: 2 to 3 short English lines about exactly what happens in this clip (no hashtags, no links).
- tags: 10 YouTube tags specific to this clip and the game.{broll}
{notes}

{chr(10).join(blocks)}

Return ONLY JSON: {{"clips":[{{"id":<int>,"keep":<bool>,"funny":<int>,"virality":<int>,"start":<sec>,"end":<sec>,"peak":<sec>,"zoom":<obj or null>,"slowmo":<obj or null>,"emphasis":["word"],"emoji":[{{"at":<sec>,"emoji":"<emoji>"}}],"game":"<text>","reason":"<text>","visual":"<text>","headline":"<TEXT>","titles":["<text>"],"description":"<text>","tags":["<text>"]}}]}}"""


def _num(v, lo, hi, default):
    try:
        return max(lo, min(hi, float(v)))
    except Exception:
        return default


def _clean(text, n):
    return re.sub(r"[<>#]", "", str(text or "")).strip()[:n]


def _parse(text, batch):
    import json
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    data = json.loads(text)
    by_id = {f["id"]: f for f in batch}
    out = {}
    for c in data.get("clips", []):
        try:
            i = int(c["id"])
        except Exception:
            continue
        f = by_id.get(i)
        if not f:
            continue
        dur = f["dur"]
        start = _num(c.get("start"), 0, dur - MIN_CLIP, 0)
        end = _num(c.get("end"), start + MIN_CLIP, dur, dur)
        peak = _num(c.get("peak"), start, end, (start + end) / 2)
        if end - start > MAX_CLIP:
            end = start + MAX_CLIP
        if end - start < MIN_CLIP:  # too short: widen around the peak
            start, end = max(0.0, peak - MIN_CLIP / 2), min(dur, peak + MIN_CLIP / 2)

        def cue(o, maxsec=2.0):
            if not isinstance(o, dict):
                return None
            at = _num(o.get("at"), start, end - 0.5, None)
            return None if at is None else {"at": at, "seconds": _num(o.get("seconds"), 0.8, maxsec, 1.2),
                                            "target": "face" if str(o.get("target")) == "face" else "gameplay"}

        emoji = []
        for e in (c.get("emoji") or [])[:3]:
            ch = str(e.get("emoji", "")).strip() if isinstance(e, dict) else ""
            at = _num(e.get("at"), start, end - 1.0, None) if isinstance(e, dict) else None
            if ch and at is not None and len(ch) <= 8 and any(ord(x) > 0x2000 for x in ch):
                emoji.append({"at": at, "emoji": ch})
        broll = []
        for b in (c.get("broll") or [])[:2]:
            if isinstance(b, dict) and str(b.get("query", "")).strip():
                broll.append({"at": _num(b.get("at"), start + 1, end - 3, start + 2), "seconds": _num(b.get("seconds"), 2, 3.5, 2.5),
                              "query": str(b["query"]).strip()[:60]})
        titles = [t for t in (_clean(t, 100) for t in (c.get("titles") or [])) if t][:5]
        out[i] = {
            "keep": bool(c.get("keep", True)), "funny": int(_num(c.get("funny"), 0, 100, 50)),
            "virality": int(_num(c.get("virality"), 0, 100, 50)), "start": start, "end": end, "peak": peak,
            "zoom": cue(c.get("zoom")), "slowmo": cue(c.get("slowmo")),
            "emphasis": [str(w).strip() for w in (c.get("emphasis") or [])][:6], "emoji": emoji, "broll": broll,
            "game": _clean(c.get("game"), 60), "reason": _clean(c.get("reason"), 160), "visual": _clean(c.get("visual"), 120),
            "headline": re.sub(r"[<>]", "", str(c.get("headline", ""))).upper().strip()[:36],
            "titles": titles, "description": _clean(c.get("description"), 700),
            "tags": [t for t in (_clean(t, 40) for t in (c.get("tags") or [])) if t][:12],
        }
    return out or None


def review(finalists, title, channel, want_broll=False, notes="", batch_size=3):
    """finalists: [{"id", "dur", "preview": bytes, "lines": str}] -> {id: result} (may be partial)."""
    results = {}
    usable = [f for f in finalists if f.get("preview")]
    for i in range(0, len(usable), batch_size):
        batch = usable[i:i + batch_size]
        parts = [{"text": _prompt(batch, title, channel, want_broll, notes)}]
        for f in batch:
            parts.append({"text": f"Preview video of CLIP [{f['id']}]:"})
            parts.append({"inline_data": {"mime_type": "video/mp4", "data": base64.b64encode(f["preview"]).decode()}})
        got = ai_pick.generate(parts, lambda t, b=batch: _parse(t, b), f"Director pass {i // batch_size + 1}", timeout=240)
        if got:
            results.update(got)
    return results or None
