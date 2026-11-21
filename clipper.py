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
    segments, _ = model.transcribe(str(wav), vad_filter=True, language=language, word_timestamps=True)
    out = []
    for s in segments:
        words = [{"start": w.start + offset, "end": w.end + offset, "word": w.word.strip()}
                 for w in (s.words or []) if w.word.strip()]
        out.append({"start": s.start + offset, "end": s.end + offset, "text": s.text.strip(), "words": words})
    return out


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


ASS_HEAD = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Noto Sans,74,&H0000E5FF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,5,2,2,60,60,380,1
Style: Hook,Noto Sans,64,&H00FFFFFF,&H00FFFFFF,&H00141414,&H00141414,-1,0,0,0,100,100,0,0,3,14,0,8,70,70,250,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def ass_time(t):
    t = max(0.0, t)
    cs = int(round(t * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02}:{cs // 100 % 60:02}.{cs % 100:02}"


def ass_text(t):
    return str(t).replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ").strip()


def chunk_words(words, max_words=4, max_dur=1.8, gap=0.6):
    chunks, cur = [], []
    for w in words:
        if cur and (len(cur) >= max_words or w["start"] - cur[-1]["end"] > gap or w["end"] - cur[0]["start"] > max_dur):
            chunks.append(cur)
            cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    return chunks


def write_ass_captions(path, segments, start, end):
    """Word-by-word captions: the word being spoken lights up (karaoke style)."""
    words = [w for s in segments for w in s.get("words", []) if start - 0.05 <= w["start"] < end]
    if not words:
        return False
    lines = []
    for chunk in chunk_words(words):
        t0 = chunk[0]["start"] - start
        t1 = min(chunk[-1]["end"], end) - start
        parts = []
        for i, w in enumerate(chunk):
            nxt = chunk[i + 1]["start"] if i + 1 < len(chunk) else max(w["end"], w["start"] + 0.15)
            parts.append("{\\k%d}%s" % (max(1, round((nxt - w["start"]) * 100)), ass_text(w["word"])))
        lines.append(f"Dialogue: 0,{ass_time(t0)},{ass_time(t1 + 0.05)},Cap,,0,0,0,,{' '.join(parts)}")
    path.write_text(ASS_HEAD + "\n".join(lines) + "\n", encoding="utf-8")
    return True


def make_hook_text(title, topics=None):
    """A short headline from the source title (English letters only)."""
    t = re.sub(r"[\u0900-\u097F]+", " ", str(title or ""))
    t = re.sub(r"[^\w\s'!?&-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(re.sub(r"[^A-Za-z]", "", t)) < 4:
        return "WAIT FOR IT"
    if len(t) > 42:
        t = t[:42].rsplit(" ", 1)[0]
    return t.upper()


def write_hook_ass(path, text, seconds=3.0):
    path.write_text(
        ASS_HEAD + f"Dialogue: 1,{ass_time(0.2)},{ass_time(seconds)},Hook,,0,0,0,,{{\\fad(250,350)}}{ass_text(text)}\n",
        encoding="utf-8")
    return True


def face_center(video, ss, length):
    """Median x position of the largest face over the clip (None if no steady face)."""
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(video))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    try:
        det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    except Exception:
        return None
    if det.empty():
        print("    face detector data not found, using a center crop")
        return None
    xs, t = [], 0.0
    while t < length and W and H:
        cap.set(cv2.CAP_PROP_POS_MSEC, (ss + t) * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (640, max(1, int(H * 640 / W))))
        faces = det.detectMultiScale(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 1.1, 5, minSize=(40, 40))
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            xs.append((x + w / 2) / 640 * W)
        t += 1.0
    cap.release()
    if len(xs) < max(3, 0.2 * length):
        return None
    xs.sort()
    return xs[len(xs) // 2], W, H


def framing_chain(mode, video, ss, length):
    """ffmpeg filters that turn the landscape frame into 1080x1920."""
    if mode == "blur":  # whole frame in the middle, blurred copy behind it
        return ("split[a][b];[a]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=25:5[bg];"
                "[b]scale=1080:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2")
    if mode == "face":
        found = face_center(video, ss, length)
        if found:
            cx, W, H = found
            cw = int(H * 9 / 16) // 2 * 2
            x = int(max(0, min(cx - cw / 2, W - cw)))
            print(f"    face found, cropping around x={int(cx)}")
            return f"crop={cw}:{H}:{x}:0,scale=1080:1920"
        print("    no steady face, using a center crop")
    return "crop=ih*9/16:ih,scale=1080:1920"


CLASSIC_STYLE = "FontName=Noto Sans,FontSize=08,Bold=0.5,Outline=1,Alignment=2,MarginV=25"


def cut_clip(video, outdir, name, ss, length, vf, captions=None, hook=None):
    """captions = ("srt"|"ass", filename), hook = ass filename."""
    if captions:
        kind, fn = captions
        vf += f",subtitles={fn}" + (f":force_style='{CLASSIC_STYLE}'" if kind == "srt" else "")
    if hook:
        vf += f",subtitles={hook}"
    subprocess.run([
        "ffmpeg", "-y", "-ss", str(ss), "-t", str(length), "-i", str(Path(video).resolve()),
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
    ap.add_argument("--framing", default="center", choices=["center", "blur", "face"], help="how to make it vertical")
    ap.add_argument("--captions", default="karaoke", choices=["karaoke", "classic", "off"], help="caption style")
    ap.add_argument("--hook", default="auto", choices=["auto", "off"], help="headline in the first seconds")
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
        captions = None
        if a.captions == "karaoke" and write_ass_captions(out / f"{name}.ass", segments, p["start"], p["end"]):
            captions = ("ass", f"{name}.ass")
        elif a.captions in ("classic", "karaoke") and write_srt(out / f"{name}.srt", segments, p["start"], p["end"]):
            captions = ("srt", f"{name}.srt")  # classic style, or fallback when no word timings
        hook = None
        if a.hook == "auto" and write_hook_ass(out / f"{name}_hook.ass", make_hook_text(info.get("title"))):
            hook = f"{name}_hook.ass"
        vf = framing_chain(a.framing, p["video"], p["ss"], a.length)
        cut_clip(p["video"], out, name, p["ss"], a.length, vf, captions, hook)
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
