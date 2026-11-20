#!/usr/bin/env python3
"""Auto clipper: URL -> best moments -> vertical 9:16 clips with captions.

Usage:
    python clipper.py "https://youtube.com/watch?v=XXXX" --num 5 --length 30
"""
import argparse
import json
import re
import subprocess
import wave
from pathlib import Path

import numpy as np
import yt_dlp
from faster_whisper import WhisperModel

# Words that usually show up around funny / chaotic / glitchy moments
KEYWORDS = [
    "lol", "lmao", "omg", "oh my god", "no way", "what the", "wtf", "bro",
    "bug", "glitch", "broke", "broken", "crash", "died", "dead", "clip that",
    "insane", "crazy", "wait what", "how", "help", "haha", "scream", "fell",
    "explode", "exploded", "flying", "stuck", "npc",
]


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def download(url, workdir, cookies=None, height=720):
    opts = {
        "format": f"bv*[height<={height}]+ba/b[height<={height}]",
        "outtmpl": str(workdir / "video.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "noplaylist": True,
        "writesubtitles": True,
        "subtitleslangs": ["live_chat"],  # chat replay of streams (if any)
    }
    if cookies:
        opts["cookiefile"] = cookies
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    return workdir / "video.mp4", info


def extract_audio(video, wav):
    run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(wav)])


def loudness_per_second(wav):
    """Per-second RMS loudness, computed in chunks to keep memory low."""
    with wave.open(str(wav)) as w:
        sr, total = w.getframerate(), w.getnframes()
        out = []
        while True:
            raw = w.readframes(sr * 300)  # 5 minutes at a time
            if not raw:
                break
            d = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            n = len(d) // sr
            if n:
                out.append(np.sqrt((d[: n * sr].reshape(n, sr) ** 2).mean(axis=1)))
    return np.concatenate(out) if out else np.zeros(1, dtype=np.float32)


def transcribe(wav, model_size="base"):
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(wav), vad_filter=True)
    return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]


CHAT_WORDS = ["lmao", "lol", "omg", "wtf", "clip", "💀", "😂", "🤣", "kek", "lmfao", "no way", "bruh"]


def chat_scores(work, n_seconds):
    """Chat message rate + hype words per second, from yt-dlp live chat replay."""
    f = next(iter(work.glob("*.live_chat.json")), None)
    if not f:
        return None
    rate = np.zeros(n_seconds, dtype=np.float32)
    hype = np.zeros(n_seconds, dtype=np.float32)
    for line in f.open(encoding="utf-8", errors="ignore"):
        try:
            d = json.loads(line)["replayChatItemAction"]
            sec = int(d.get("videoOffsetTimeMsec", 0)) // 1000
            if not 0 <= sec < n_seconds:
                continue
            for act in d.get("actions", []):
                item = act.get("addChatItemAction", {}).get("item", {})
                msg = item.get("liveChatTextMessageRenderer", {}).get("message", {})
                if not msg:
                    continue
                rate[sec] += 1
                text = "".join(r.get("text", r.get("emoji", {}).get("emojiId", "")) for r in msg.get("runs", [])).lower()
                hype[sec] += sum(w in text for w in CHAT_WORDS)
        except (KeyError, ValueError, TypeError):
            continue
    rate = np.convolve(rate, np.ones(5), mode="same")  # smooth over 5s
    z = np.clip((rate - rate.mean()) / (rate.std() + 1e-6), 0, None)
    return z * 1.5 + hype * 0.5


def build_scores(n_seconds, rms, segments, info, chat=None):
    score = np.zeros(n_seconds, dtype=np.float32)

    # 1) Loudness spikes (screaming, laughing, explosions)
    z = (rms - rms.mean()) / (rms.std() + 1e-6)
    score[: len(z)] += np.clip(z, 0, None)[:n_seconds]

    # 2) Keyword hits in the transcript
    for seg in segments:
        text = seg["text"].lower()
        hits = sum(1 for k in KEYWORDS if k in text)
        if hits:
            for s in range(int(seg["start"]), min(int(seg["end"]) + 1, n_seconds)):
                score[s] += 1.5 * hits

    # 3) YouTube "most replayed" heatmap, when available
    for h in info.get("heatmap") or []:
        for s in range(int(h["start_time"]), min(int(h["end_time"]) + 1, n_seconds)):
            score[s] += 3 * h["value"]

    # 4) Live chat spikes (best signal for streams)
    if chat is not None:
        score += chat

    return score


def pick_windows(score, length, num, lead_in=3):
    csum = np.concatenate([[0], np.cumsum(score)])
    window_sums = csum[length:] - csum[:-length]
    taken, picks = np.zeros(len(score), dtype=bool), []
    for idx in np.argsort(window_sums)[::-1]:
        if taken[idx: idx + length].any():
            continue
        start = max(0, int(idx) - lead_in)
        picks.append({"start": start, "end": start + length, "score": float(window_sums[idx])})
        taken[max(0, idx - length): idx + length] = True
        if len(picks) == num:
            break
    return sorted(picks, key=lambda p: p["start"])


def srt_time(t):
    ms = int(t * 1000)
    return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"


def write_srt(path, segments, start, end):
    lines, i = [], 1
    for s in segments:
        if s["end"] < start or s["start"] > end:
            continue
        a, b = max(s["start"], start) - start, min(s["end"], end) - start
        lines.append(f"{i}\n{srt_time(a)} --> {srt_time(b)}\n{s['text']}\n")
        i += 1
    path.write_text("\n".join(lines), encoding="utf-8")
    return i > 1


def cut_clip(video, outdir, name, start, length, has_subs):
    vf = "crop=ih*9/16:ih,scale=1080:1920"
    if has_subs:
        style = "FontSize=08,Bold=0.5,Outline=1,Alignment=2,MarginV=25"
        vf += f",subtitles={name}.srt:force_style='{style}'"
    subprocess.run([
        "ffmpeg", "-y", "-ss", str(start), "-t", str(length), "-i", str(video.resolve()),
        "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k", f"{name}.mp4",
    ], cwd=outdir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--num", type=int, default=5, help="number of clips")
    ap.add_argument("--length", type=int, default=30, help="clip length in seconds")
    ap.add_argument("--out", default="output")
    ap.add_argument("--model", default="base", help="whisper model: tiny/base/small")
    ap.add_argument("--height", type=int, default=720, help="max video height")
    ap.add_argument("--cookies", default=None, help="cookies.txt for yt-dlp")
    a = ap.parse_args()

    out = Path(a.out)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)

    print("1/5 Downloading...")
    video, info = download(a.url, work, a.cookies, a.height)
    print("2/5 Extracting audio...")
    wav = work / "audio.wav"
    extract_audio(video, wav)
    rms = loudness_per_second(wav)
    print("3/5 Transcribing (slow on CPU)...")
    segments = transcribe(wav, a.model)
    print("4/5 Scoring moments...")
    chat = chat_scores(work, len(rms))
    print("  chat replay:", "found" if chat is not None else "none")
    score = build_scores(len(rms), rms, segments, info, chat)
    picks = pick_windows(score, a.length, a.num)
    print("5/5 Cutting clips...")
    manifest = []
    for n, p in enumerate(picks, 1):
        name = f"clip_{n:02}"
        has_subs = write_srt(out / f"{name}.srt", segments, p["start"], p["end"])
        cut_clip(video, out, name, p["start"], a.length, has_subs)
        manifest.append({
            "file": f"{name}.mp4", "source": a.url, "source_title": info.get("title"),
            "channel": info.get("uploader"), "start": p["start"], "end": p["end"],
            "score": p["score"], "approved": False,
        })
        print(f"  {name}.mp4  ({p['start']}s-{p['end']}s)")
    (out / "clips.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done. {len(manifest)} clips in {out}/")


if __name__ == "__main__":
    main()
