#!/usr/bin/env python3
"""Auto clipper: URL -> best moments -> vertical 9:16 clips with captions.

Usage:
    python clipper.py "https://youtube.com/watch?v=XXXX" --num 5 --length 30
"""
import argparse
import json
import re
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import yt_dlp
from faster_whisper import WhisperModel
from yt_dlp.utils import download_range_func

# Words that usually show up around funny / chaotic / glitchy moments
KEYWORDS = [
    "lol", "lmao", "omg", "oh my god", "no way", "what the", "wtf", "bro",
    "bug", "glitch", "broke", "broken", "crash", "died", "dead", "clip that",
    "insane", "crazy", "wait what", "how", "help", "haha", "scream", "fell",
    "explode", "exploded", "flying", "stuck", "npc",
    # Hindi (Devanagari)
    "हाहा", "अरे", "अबे", "पागल", "गज़ब", "गजब", "बग", "ग्लिच", "मर गया", "बचाओ", "क्या बात", "मज़ा", "मजा", "कमाल",
    # Hinglish (Roman letters)
    "arre", "arey", "abe", "pagal", "gazab", "kamaal", "zabardast", "mast", "maza", "kya baat", "mar gaya", "bachao",
]


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def base_opts(cookies):
    o = {"quiet": True, "noplaylist": True}
    if cookies:
        o["cookiefile"] = cookies
    return o


def probe(url, cookies, max_minutes):
    """Look at the video first, so we can fail early with a clear reason."""
    with yt_dlp.YoutubeDL({**base_opts(cookies), "skip_download": True}) as ydl:
        pre = ydl.extract_info(url, download=False)
    if pre.get("live_status") in ("is_live", "is_upcoming"):
        sys.exit("ERROR: this video is still live or has not started. Clip it after the stream ends.")
    dur = pre.get("duration") or 0
    if dur > max_minutes * 60:
        sys.exit(f"ERROR: this video is {dur // 3600}h {dur % 3600 // 60}m long; the limit is {max_minutes} min.")
    return pre


def download(url, workdir, cookies=None, height=720):
    """Short videos: download the whole video (+ live chat replay if there is one)."""
    opts = {
        **base_opts(cookies),
        "format": f"bv*[height<={height}]+ba/b[height<={height}]",
        "outtmpl": str(workdir / "video.%(ext)s"),
        "merge_output_format": "mp4",
        "writesubtitles": True,
        "subtitleslangs": ["live_chat"],  # chat replay of streams (if any)
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    return workdir / "video.mp4", info


AUDIO_EXT = {".m4a", ".webm", ".opus", ".mp3", ".mp4", ".mka", ".ogg"}


def download_audio(url, workdir, cookies=None):
    """Long videos: download only the audio (small) + live chat replay."""
    opts = {
        **base_opts(cookies),
        "format": "ba[abr<=130]/ba",
        "outtmpl": str(workdir / "audio_src.%(ext)s"),
        "writesubtitles": True,
        "subtitleslangs": ["live_chat"],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    src = next(p for p in workdir.glob("audio_src.*") if p.suffix in AUDIO_EXT and "live_chat" not in p.name)
    return src, info


def download_segment(url, workdir, n, start, end, cookies=None, height=720):
    """Long videos: download ONLY one section of the video."""
    opts = {
        **base_opts(cookies),
        "format": f"bv*[height<={height}]+ba/b[height<={height}]",
        "outtmpl": str(workdir / f"seg_{n:02}.%(ext)s"),
        "merge_output_format": "mp4",
        "download_ranges": download_range_func(None, [(start, end)]),
        "force_keyframes_at_cuts": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    files = [p for p in workdir.glob(f"seg_{n:02}*") if p.suffix in (".mp4", ".mkv", ".webm")]
    if not files:
        raise RuntimeError(f"section {start}-{end}s was not downloaded")
    return files[0]


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


def load_model(model_size="base"):
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def transcribe_wav(model, wav, language=None, offset=0.0):
    segments, _ = model.transcribe(str(wav), vad_filter=True, language=language)
    return [{"start": s.start + offset, "end": s.end + offset, "text": s.text.strip()} for s in segments]


def transcribe(wav, model_size="base", language=None):
    return transcribe_wav(load_model(model_size), wav, language)


def keyword_hits(text):
    t = text.lower()
    return sum(1 for k in KEYWORDS if k in t)


CHAT_WORDS = ["pagal", "gazab", "kamaal", "haha", "हाहा", "lmao", "lol", "omg", "wtf", "clip", "💀", "😂", "🤣", "kek", "lmfao", "no way", "bruh"]


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
        style = "FontName=Noto Sans,FontSize=08,Bold=0,Outline=1,Alignment=2,MarginV=25"
        vf += f",subtitles={name}.srt:force_style='{style}'"
    subprocess.run([
        "ffmpeg", "-y", "-ss", str(start), "-t", str(length), "-i", str(video.resolve()),
        "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k", f"{name}.mp4",
    ], cwd=outdir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def short_pipeline(a, out, work):
    print("1/5 Downloading...")
    video, info = download(a.url, work, a.cookies, a.height)
    print("2/5 Extracting audio...")
    wav = work / "audio.wav"
    extract_audio(video, wav)
    rms = loudness_per_second(wav)
    print("3/5 Transcribing (slow on CPU)...")
    segments = transcribe(wav, a.model, a.language)
    print("4/5 Scoring moments...")
    chat = chat_scores(work, len(rms))
    print("  chat replay:", "found" if chat is not None else "none")
    score = build_scores(len(rms), rms, segments, info, chat)
    picks = pick_windows(score, a.length, a.num)
    for p in picks:
        p["video"], p["ss"] = video, p["start"]
    return info, picks, segments


def long_pipeline(a, out, work):
    """For long streams: audio only -> shortlist -> transcribe the shortlist -> download only the best sections."""
    print("1/5 Long video: downloading audio only...")
    src, info = download_audio(a.url, work, a.cookies)
    print("2/5 Extracting audio and measuring loudness...")
    wav = work / "audio.wav"
    extract_audio(src, wav)
    src.unlink()
    rms = loudness_per_second(wav)
    chat = chat_scores(work, len(rms))
    print("  chat replay:", "found" if chat is not None else "none")
    base = build_scores(len(rms), rms, [], info, chat)
    cands = pick_windows(base, a.length, max(a.num * 3, a.num + 4))
    print(f"3/5 Transcribing the {len(cands)} best candidate moments only...")
    model = load_model(a.model)
    segments = []
    for i, c in enumerate(cands, 1):
        sl = work / "slice.wav"
        run(["ffmpeg", "-y", "-ss", str(c["start"]), "-t", str(a.length), "-i", str(wav), "-c", "copy", str(sl)])
        segs = transcribe_wav(model, sl, a.language, offset=c["start"])
        segments += segs
        c["score"] += 3 * sum(keyword_hits(s["text"]) for s in segs)
        print(f"  candidate {i}/{len(cands)} done")
    picks = sorted(sorted(cands, key=lambda c: c["score"], reverse=True)[: a.num], key=lambda c: c["start"])
    print(f"4/5 Downloading only the {len(picks)} chosen sections of the video...")
    for n, p in enumerate(picks, 1):
        p["video"] = download_segment(a.url, work, n, p["start"], p["end"] + 1, a.cookies, a.height)
        p["ss"] = 0
    return info, picks, segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--num", type=int, default=5, help="number of clips")
    ap.add_argument("--length", type=int, default=30, help="clip length in seconds")
    ap.add_argument("--out", default="output")
    ap.add_argument("--model", default="base", help="whisper model: tiny/base/small")
    ap.add_argument("--language", default=None, help="e.g. hi for Hindi, en for English (default: auto)")
    ap.add_argument("--height", type=int, default=720, help="max video height")
    ap.add_argument("--long-minutes", type=int, default=75, help="videos longer than this use the long-video mode")
    ap.add_argument("--max-minutes", type=int, default=720, help="refuse videos longer than this")
    ap.add_argument("--cookies", default=None, help="cookies.txt for yt-dlp")
    a = ap.parse_args()

    out = Path(a.out)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)

    pre = probe(a.url, a.cookies, a.max_minutes)
    long_video = (pre.get("duration") or 0) > a.long_minutes * 60
    info, picks, segments = (long_pipeline if long_video else short_pipeline)(a, out, work)

    print("5/5 Cutting clips...")
    manifest = []
    for n, p in enumerate(picks, 1):
        name = f"clip_{n:02}"
        has_subs = write_srt(out / f"{name}.srt", segments, p["start"], p["end"])
        cut_clip(p["video"], out, name, p["ss"], a.length, has_subs)
        manifest.append({
            "file": f"{name}.mp4", "source": a.url, "source_title": info.get("title"),
            "channel": info.get("uploader"), "source_duration": info.get("duration"),
            "start": p["start"], "end": p["end"], "score": p["score"], "approved": False,
        })
        print(f"  {name}.mp4  ({p['start']}s-{p['end']}s)")
    (out / "clips.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done. {len(manifest)} clips in {out}/")


if __name__ == "__main__":
    main()
