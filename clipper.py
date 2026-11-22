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

import ai_pick
import broll
import progress

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


def is_youtube(url):
    return bool(re.search(r"(youtube\.com|youtu\.be)", url))


def base_opts(cookies):
    o = {"quiet": True, "noplaylist": True}
    if cookies:
        o["cookiefile"] = cookies
    return o


def fmt_video(height):
    # last two choices are for sources that do not report a height (direct file links)
    return f"bv*[height<={height}]+ba/b[height<={height}]/bv*+ba/b"


def download_hook(stage, label):
    def hook(d):
        if d.get("status") != "downloading":
            return
        total = d.get("total_bytes") or d.get("total_bytes_estimate")
        if total:
            frac = d["downloaded_bytes"] / total
        elif d.get("fragment_count"):
            frac = (d.get("fragment_index") or 0) / d["fragment_count"]
        else:
            return
        stage(frac, f"{label} {int(frac * 100)}%")
    return hook


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


def download(url, workdir, cookies=None, height=1080, stage=None):
    """Short videos: download the whole video (+ live chat replay if it is a YouTube stream)."""
    opts = {
        **base_opts(cookies),
        "format": fmt_video(height),
        "outtmpl": str(workdir / "video.%(ext)s"),
        "merge_output_format": "mp4",
    }
    if is_youtube(url):
        opts.update(writesubtitles=True, subtitleslangs=["live_chat"])  # chat replay of streams (if any)
    if stage:
        opts["progress_hooks"] = [download_hook(stage, "Downloading video")]
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    files = [p for p in workdir.glob("video.*") if p.suffix in (".mp4", ".mkv", ".webm", ".mov")]
    return (workdir / "video.mp4" if (workdir / "video.mp4").exists() else files[0]), info


AUDIO_EXT = {".m4a", ".webm", ".opus", ".mp3", ".mp4", ".mka", ".ogg", ".aac", ".wav", ".mkv", ".mov"}


def download_audio(url, workdir, cookies=None, stage=None):
    """Long videos: download only the audio (small) + live chat replay."""
    opts = {
        **base_opts(cookies),
        "format": "ba[abr<=130]/ba/b",
        "outtmpl": str(workdir / "audio_src.%(ext)s"),
    }
    if is_youtube(url):
        opts.update(writesubtitles=True, subtitleslangs=["live_chat"])
    if stage:
        opts["progress_hooks"] = [download_hook(stage, "Downloading audio")]
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    src = next(p for p in workdir.glob("audio_src.*") if p.suffix in AUDIO_EXT and "live_chat" not in p.name)
    return src, info


def download_segment(url, workdir, n, start, end, cookies=None, height=1080):
    """Long videos: download ONLY one section of the video."""
    opts = {
        **base_opts(cookies),
        "format": fmt_video(height),
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


def transcribe_wav(model, wav, language=None, offset=0.0, duration=None, on_progress=None):
    segments, _ = model.transcribe(str(wav), vad_filter=True, language=language, word_timestamps=True)
    out = []
    for s in segments:
        words = [{"start": w.start + offset, "end": w.end + offset, "word": w.word.strip()}
                 for w in (s.words or []) if w.word.strip()]
        out.append({"start": s.start + offset, "end": s.end + offset, "text": s.text.strip(), "words": words})
        if on_progress and duration:
            on_progress(min(1.0, s.end / duration))
    return out


def transcribe(wav, model_size="base", language=None, duration=None, on_progress=None):
    return transcribe_wav(load_model(model_size), wav, language, 0.0, duration, on_progress)


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
Style: Cap,Noto Sans,74,&H0000E5FF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,5,2,2,60,60,@CAPM@,1
Style: Hook,Noto Sans,64,&H00FFFFFF,&H00FFFFFF,&H00141414,&H00141414,-1,0,0,0,100,100,0,0,3,14,0,8,70,70,250,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def ass_head(capm=230):
    return ASS_HEAD.replace("@CAPM@", str(int(capm)))


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


def write_ass_captions(path, segments, start, end, margin=230):
    """Word-by-word captions at the bottom: the word being spoken lights up (karaoke style)."""
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
    path.write_text(ass_head(margin) + "\n".join(lines) + "\n", encoding="utf-8")
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


def write_hook_ass(path, text, seconds=3.0, seam=False):
    """seam=True puts the headline on the line between the face and the gameplay."""
    tag = "{\\an5\\pos(540,640)\\fad(250,350)}" if seam else "{\\fad(250,350)}"
    path.write_text(ass_head() + f"Dialogue: 1,{ass_time(0.2)},{ass_time(seconds)},Hook,,0,0,0,,{tag}{ass_text(text)}\n",
                    encoding="utf-8")
    return True


YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


def yunet_model():
    path = Path.home() / ".cache" / "clipper" / "yunet.onnx"
    try:
        if path.exists() and path.stat().st_size > 10000:
            return str(path)
        import urllib.request
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(YUNET_URL, path)
        return str(path) if path.stat().st_size > 10000 else None
    except Exception:
        return None


def face_samples(video, ss, length, step=0.5):
    """Face position every `step` seconds: (W, H, [(t, (cx, cy, fw) or None), ...]) or None."""
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(video))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not (W and H):
        return None
    yn, haar = None, None
    model = yunet_model()
    if model and hasattr(cv2, "FaceDetectorYN"):
        try:
            yn = cv2.FaceDetectorYN.create(model, "", (320, 320), 0.7, 0.3, 50)
        except Exception:
            yn = None
    if yn is None:
        try:
            haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        except Exception:
            haar = None
        if haar is None or haar.empty():
            cap.release()
            print("    no face detector available")
            return None
    sw = 640
    sh = max(1, int(H * sw / W))
    scale = W / sw
    samples, t = [], 0.0
    while t < length:
        cap.set(cv2.CAP_PROP_POS_MSEC, (ss + t) * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (sw, sh))
        box = None
        if yn is not None:
            yn.setInputSize((sw, sh))
            _, faces = yn.detect(small)
            if faces is not None and len(faces):
                f = max(faces, key=lambda r: r[2] * r[3])
                box = (float(f[0]), float(f[1]), float(f[2]), float(f[3]))
        else:
            faces = haar.detectMultiScale(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 1.1, 5, minSize=(40, 40))
            if len(faces):
                f = max(faces, key=lambda r: r[2] * r[3])
                box = tuple(float(v) for v in f)
        samples.append((t, None if box is None else ((box[0] + box[2] / 2) * scale, (box[1] + box[3] / 2) * scale, box[2] * scale)))
        t += step
    cap.release()
    return W, H, samples


def face_track(video, ss, length):
    """The face's path over the clip: {W, H, fw, t, cx, cy} or None if there is no steady face."""
    got = face_samples(video, ss, length)
    if not got:
        return None
    W, H, samples = got
    hits = [s for s in samples if s[1]]
    if len(hits) < max(3, 0.2 * len(samples)):
        return None
    fw = sorted(s[1][2] for s in hits)[len(hits) // 2]
    last = hits[0][1]
    ts, cxs, cys = [], [], []
    for t, box in samples:  # hold the last known position while the face is briefly lost
        last = box or last
        ts.append(t)
        cxs.append(last[0])
        cys.append(last[1])
    return {"W": W, "H": H, "fw": fw, "t": ts, "cx": cxs, "cy": cys}


def follow(vals, band, speed):
    """Camera movement: stay still while the face moves a little, glide when it moves a lot."""
    cur, out = vals[0], []
    for v in vals:
        if abs(v - cur) > band:
            cur += max(-speed, min(speed, v - cur))
        out.append(cur)
    sm = []
    for i in range(len(out)):  # light smoothing
        w = out[max(0, i - 1): i + 2]
        sm.append(sum(w) / len(w))
    return sm


def path_expr(ts, vs, tol=2.0):
    """ffmpeg expression of time t that moves smoothly through the points."""
    pts = [(ts[0], vs[0])]
    for t, v in zip(ts[1:-1], vs[1:-1]):
        if abs(v - pts[-1][1]) > tol:
            pts.append((t, v))
    if len(ts) > 1:
        pts.append((ts[-1], vs[-1]))
    if len(pts) == 1 or max(v for _, v in pts) - min(v for _, v in pts) <= tol:
        return f"{round(sum(v for _, v in pts) / len(pts))}"
    expr = f"{pts[-1][1]:.1f}"
    for (t0, v0), (t1, v1) in reversed(list(zip(pts[:-1], pts[1:]))):
        seg = f"{v0:.1f}+({v1 - v0:.2f})*(t-{t0:.2f})/{max(t1 - t0, 0.05):.2f}"
        expr = f"if(lt(t,{t1:.2f}),{seg},{expr})"
    return expr


TOP_H, BOT_H = 640, 1280  # stacked layout: face panel on top, gameplay below (1080 wide)
CENTER = "[0:v]crop=ih*9/16:ih,scale=1080:1920[v0]"
BLUR = ("[0:v]split[a][b];[a]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=25:5[bg];"
        "[b]scale=1080:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2[v0]")


def layout_graph(mode, video, ss, length):
    """Returns (ffmpeg graph from [0:v] to [v0], headline-on-the-seam?, layout actually used)."""
    if mode == "blur":
        return BLUR, False, "blur"
    if mode in ("stack", "face"):
        tr = face_track(video, ss, length)
        if not tr:
            print("    no steady face found, using the " + ("blurred layout" if mode == "stack" else "center crop"))
            return (BLUR, False, "blur") if mode == "stack" else (CENTER, False, "center")
        W, H, fw, ts = tr["W"], tr["H"], tr["fw"], tr["t"]
        clamp = lambda v, lo, hi: max(lo, min(hi, v))
        if mode == "face":
            cw = int(H * 9 / 16) // 2 * 2
            cxs = follow(tr["cx"], 0.12 * cw, 0.3 * cw)
            xe = path_expr(ts, [clamp(c - cw / 2, 0, W - cw) for c in cxs])
            print("    face found, the crop follows it")
            return f"[0:v]crop={cw}:{H}:x='{xe}':y=0,scale=1080:1920[v0]", False, "face"
        pw = min(W, max(fw * 3.0, H * 0.35))  # the facecam is about 3 face-widths wide
        ph = pw * TOP_H / 1080
        if ph > H:
            ph, pw = H, H * 1080 / TOP_H
        pw, ph = int(pw) // 2 * 2, int(ph) // 2 * 2
        cxs = follow(tr["cx"], 0.12 * pw, 0.3 * pw)
        cys = follow(tr["cy"], 0.12 * ph, 0.3 * ph)
        xe = path_expr(ts, [clamp(c - pw / 2, 0, W - pw) for c in cxs])
        ye = path_expr(ts, [clamp(c - ph * 0.45, 0, H - ph) for c in cys])
        gw = min(W, int(H * 1080 / BOT_H)) // 2 * 2
        gh = int(gw * BOT_H / 1080) // 2 * 2
        gx = (W - gw) // 2
        print("    face found: face on top (the crop follows it), gameplay below")
        graph = (f"[0:v]split[t][g];[t]crop={pw}:{ph}:x='{xe}':y='{ye}',scale=1080:{TOP_H}[top];"
                 f"[g]crop={gw}:{gh}:{gx}:0,scale=1080:{BOT_H}[bot];[top][bot]vstack[v0]")
        return graph, True, "stack"
    return CENTER, False, "center"


CLASSIC_STYLE = "FontName=Noto Sans,FontSize=08,Bold=0.5,Outline=1,Alignment=2,MarginV=25"


XFADE = {
    "fade": ["fade"], "flash": ["fadewhite"], "slide": ["slideleft", "slideup"], "wipe": ["wipeleft", "wiperight"],
    "zoom": ["zoomin"], "circle": ["circleopen", "circleclose"],
    "mix": ["fade", "slideleft", "zoomin", "circleopen", "wipeleft", "fadewhite", "pixelize"],
}
TD = 0.4  # transition length in seconds


def plan_cutaways(items, length):
    """Keep only cutaways that fit (not too early, not overlapping, not at the very end)."""
    kept, last_end = [], 0.0
    for b in sorted(items, key=lambda b: b["at"]):
        at, dur = b["at"], b["seconds"]
        if at < TD + 0.6 or at < last_end + 2 * TD + 0.6 or at + dur + TD + 0.4 > length:
            continue
        kept.append(b)
        last_end = at + dur
    return kept


def hard_cut_parts(cur, kept):
    parts = []
    for k, b in enumerate(kept, 1):
        at, dur = b["at"], b["seconds"]
        parts.append(f"[{k}:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1,"
                     f"format=yuv420p,setpts=PTS-STARTPTS+{at}/TB[b{k}]")
        parts.append(f"[{cur}][b{k}]overlay=eof_action=pass:enable='between(t,{at},{at + dur})'[o{k}]")
        cur = f"o{k}"
    return parts, cur


def xfade_parts(cur, kept, length, style):
    """Stock footage cutaways with real transitions (slide, wipe, zoom, circle...) in and out."""
    n = len(kept)
    outs = "".join(f"[m{i}]" for i in range(n + 1))
    parts = [f"[{cur}]fps=30,format=yuv420p,settb=AVTB,setsar=1,split={n + 1}{outs}"]
    bounds = []
    for i in range(n + 1):
        st = 0.0 if i == 0 else kept[i - 1]["at"] + kept[i - 1]["seconds"] - TD
        en = length if i == n else kept[i]["at"] + TD
        parts.append(f"[m{i}]trim=start={st:.3f}:end={en:.3f},setpts=PTS-STARTPTS[p{i}]")
        bounds.append((st, en))
    for k, b in enumerate(kept, 1):
        parts.append(f"[{k}:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1,fps=30,"
                     f"format=yuv420p,settb=AVTB,trim=duration={b['seconds']:.3f},setpts=PTS-STARTPTS[b{k}]")
    names = XFADE.get(style, ["fade"])
    x, ln, j = "p0", bounds[0][1], 0
    for k, b in enumerate(kept, 1):
        parts.append(f"[{x}][b{k}]xfade=transition={names[j % len(names)]}:duration={TD}:offset={ln - TD:.3f}[x{k}a]")
        j += 1
        ln += b["seconds"] - TD
        parts.append(f"[x{k}a][p{k}]xfade=transition={names[j % len(names)]}:duration={TD}:offset={ln - TD:.3f}[x{k}]")
        j += 1
        ln += (bounds[k][1] - bounds[k][0]) - TD
        x = f"x{k}"
    parts.append(f"[{x}]tpad=stop_mode=clone:stop_duration=1,trim=duration={length}[xc]")
    return parts, "xc"


def render_clip(video, outdir, name, ss, length, graph, captions=None, hook=None, broll_items=(), transitions="off", crf=19):
    """captions = ("srt"|"ass", file), hook = ass file, broll_items = [{file, at, seconds}]."""
    kept = plan_cutaways(broll_items, length)
    inputs = ["-ss", str(ss), "-t", str(length), "-i", str(Path(video).resolve())]
    for b in kept:
        inputs += ["-t", str(b["seconds"] + 1.0), "-i", str(Path(b["file"]).resolve())]
    parts, cur = [graph], "v0"
    if kept:
        more, cur = (hard_cut_parts(cur, kept) if transitions == "off"
                     else xfade_parts(cur, kept, length, transitions))
        parts += more
    if transitions != "off":
        color = ":color=white" if transitions == "flash" else ""
        parts.append(f"[{cur}]fade=t=in:st=0:d=0.3{color},fade=t=out:st={length - 0.3:.2f}:d=0.3[vf]")
        cur = "vf"
    if captions:
        kind, fn = captions
        parts.append(f"[{cur}]subtitles={fn}" + (f":force_style='{CLASSIC_STYLE}'" if kind == "srt" else "") + "[vc]")
        cur = "vc"
    if hook:
        parts.append(f"[{cur}]subtitles={hook}[vh]")
        cur = "vh"
    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(parts), "-map", f"[{cur}]", "-map", "0:a?"]
    if transitions != "off":
        cmd += ["-af", f"afade=t=in:st=0:d=0.3,afade=t=out:st={length - 0.3:.2f}:d=0.3"]
    cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", f"{name}.mp4"]
    subprocess.run(cmd, cwd=outdir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def grab_frame(src, t, headers=None, width=448):
    """One still frame (JPEG bytes) at time t, or None."""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t:.2f}"]
    if headers:
        cmd += ["-headers", "".join(f"{k}: {v}\r\n" for k, v in headers.items())]
    cmd += ["-i", str(src), "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "6", "-f", "image2pipe", "-vcodec", "mjpeg", "-"]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=60)
    except Exception:
        return None
    return r.stdout if r.returncode == 0 and r.stdout else None


def remote_source(info, max_height=480):
    """A low-quality direct URL to pull single frames from, for long videos we did not download."""
    best = None
    for f in info.get("formats") or []:
        if not f.get("url") or f.get("vcodec") in (None, "none"):
            continue
        h = f.get("height") or 0
        if h > max_height:
            continue
        score = ((f.get("protocol") or "") in ("https", "http"), h)
        if best is None or score > best[0]:
            best = (score, f)
    if best:
        return best[1]["url"], best[1].get("http_headers")
    if info.get("url"):
        return info["url"], info.get("http_headers")
    return None


def add_frames(cands, length, frame_src):
    if not frame_src:
        return
    src, headers = frame_src
    got = 0
    for c in cands:
        imgs = [grab_frame(src, c["start"] + f * length, headers) for f in (0.25, 0.7)]
        c["images"] = [i for i in imgs if i]
        got += len(c["images"])
    print(f"  grabbed {got} still frames for the AI")


def attach_text(cands, segments):
    for c in cands:
        c["text"] = " ".join(s["text"] for s in segments if s["end"] >= c["start"] and s["start"] <= c["end"]).strip()


def choose(cands, a, info, frame_src=None):
    """Pick the final moments: AI scores (text + still frames) if available, otherwise sound/chat/keyword scores."""
    ranked = None
    if a.ai == "on":
        progress.report(a.ai_lo, "Looking at the best moments")
        add_frames(cands, a.length, frame_src)
        items = [{"id": i, "start": c["start"], "end": c["end"], "text": c.get("text", ""), "energy": c["score"],
                  "images": c.get("images", [])} for i, c in enumerate(cands)]
        progress.report(a.ai_lo + 0.03, "AI is scoring the moments")
        ranked = ai_pick.rank(items, info.get("title"), info.get("uploader"), a.num, a.broll == "auto",
                              ai_pick.performance_notes())
        if not ranked:
            print("  AI picking not available, using the sound/chat/keyword picks")
    if ranked:
        for i, c in enumerate(cands):
            r = ranked.get(i)
            if r:
                c.update(virality=r["virality"], detail=r["detail"], reason=r["reason"], ai_title=r["title"],
                         headline=r["headline"], broll=r["broll"], visual=r.get("visual", ""))
        top = sorted(cands, key=lambda c: (c.get("virality", -1), c["score"]), reverse=True)[: a.num]
    else:
        top = sorted(cands, key=lambda c: c["score"], reverse=True)[: a.num]
    for c in cands:
        c.pop("images", None)
    return sorted(top, key=lambda c: c["start"])


def short_pipeline(a, out, work):
    a.ai_lo, a.cut_lo = 0.66, 0.76
    print("1/5 Downloading...")
    progress.report(0.0, "Downloading video")
    video, info = download(a.url, work, a.cookies, a.height, stage=progress.stage(0.0, 0.30))
    print("2/5 Extracting audio...")
    progress.report(0.31, "Extracting audio")
    wav = work / "audio.wav"
    extract_audio(video, wav)
    rms = loudness_per_second(wav)
    print("3/5 Transcribing (slow on CPU)...")
    tstage = progress.stage(0.34, 0.65)
    segments = transcribe(wav, a.model, a.language, duration=len(rms), on_progress=lambda f: tstage(f, f"Transcribing {int(f * 100)}%"))
    print("4/5 Scoring moments...")
    chat = chat_scores(work, len(rms))
    print("  chat replay:", "found" if chat is not None else "none")
    score = build_scores(len(rms), rms, segments, info, chat)
    cands = pick_windows(score, a.length, max(a.num * 3, a.num + 4))
    attach_text(cands, segments)
    picks = choose(cands, a, info, (str(video), None))
    for p in picks:
        p["video"], p["ss"] = video, p["start"]
    return info, picks, segments


def long_pipeline(a, out, work):
    """For long streams: audio only -> shortlist -> transcribe the shortlist -> download only the best sections."""
    a.ai_lo, a.cut_lo = 0.50, 0.80
    print("1/5 Long video: downloading audio only...")
    progress.report(0.0, "Downloading the audio")
    src, info = download_audio(a.url, work, a.cookies, stage=progress.stage(0.0, 0.15))
    print("2/5 Extracting audio and measuring loudness...")
    progress.report(0.16, "Measuring loudness")
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
        progress.report(0.20 + 0.30 * (i - 1) / len(cands), f"Transcribing moment {i} of {len(cands)}")
        sl = work / "slice.wav"
        run(["ffmpeg", "-y", "-ss", str(c["start"]), "-t", str(a.length), "-i", str(wav), "-c", "copy", str(sl)])
        segs = transcribe_wav(model, sl, a.language, offset=c["start"])
        segments += segs
        c["score"] += 3 * sum(keyword_hits(s["text"]) for s in segs)
        print(f"  candidate {i}/{len(cands)} done")
    attach_text(cands, segments)
    picks = choose(cands, a, info, remote_source(info))
    print(f"4/5 Downloading only the {len(picks)} chosen sections of the video...")
    for n, p in enumerate(picks, 1):
        progress.report(0.62 + 0.17 * (n - 1) / max(1, len(picks)), f"Downloading section {n} of {len(picks)}")
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
    ap.add_argument("--framing", default="stack", choices=["stack", "blur", "face", "center"], help="how to make it vertical")
    ap.add_argument("--captions", default="karaoke", choices=["karaoke", "classic", "off"], help="caption style")
    ap.add_argument("--caption-margin", type=int, default=230, help="caption distance from the bottom edge (pixels of 1920)")
    ap.add_argument("--hook", default="auto", choices=["auto", "off"], help="headline in the first seconds")
    ap.add_argument("--ai", default="on", choices=["on", "off"], help="AI moment picking and virality scores")
    ap.add_argument("--broll", default="off", choices=["off", "auto"], help="stock footage cutaways (needs AI picking)")
    ap.add_argument("--transitions", default="off", choices=["off", "fade", "flash", "slide", "wipe", "zoom", "circle", "mix"],
                    help="fades at the ends, and the style used for stock footage cutaways")
    ap.add_argument("--height", type=int, default=1080, help="max source video height (720, 1080, 1440)")
    ap.add_argument("--long-minutes", type=int, default=75, help="videos longer than this use the long-video mode")
    ap.add_argument("--max-minutes", type=int, default=720, help="refuse videos longer than this")
    ap.add_argument("--cookies", default=None, help="cookies.txt for yt-dlp")
    a = ap.parse_args()

    out = Path(a.out)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)

    progress.report(0.0, "Checking the video", force=True)
    pre = probe(a.url, a.cookies, a.max_minutes)
    long_video = (pre.get("duration") or 0) > a.long_minutes * 60
    info, picks, segments = (long_pipeline if long_video else short_pipeline)(a, out, work)

    print("5/5 Cutting clips...")
    manifest = []
    for n, p in enumerate(picks, 1):
        progress.report(a.cut_lo + (1 - a.cut_lo) * (n - 1) / max(1, len(picks)), f"Cutting clip {n} of {len(picks)}")
        name = f"clip_{n:02}"
        graph, seam, used = layout_graph(a.framing, p["video"], p["ss"], a.length)
        captions = None
        if a.captions == "karaoke" and write_ass_captions(out / f"{name}.ass", segments, p["start"], p["end"], a.caption_margin):
            captions = ("ass", f"{name}.ass")
        elif a.captions in ("classic", "karaoke") and write_srt(out / f"{name}.srt", segments, p["start"], p["end"]):
            captions = ("srt", f"{name}.srt")  # classic style, or fallback when no word timings
        headline = p.get("headline") or make_hook_text(info.get("title"))
        hook = None
        if a.hook == "auto" and write_hook_ass(out / f"{name}_hook.ass", headline, seam=seam):
            hook = f"{name}_hook.ass"
        broll_items = []
        if a.broll == "auto":
            for k, b in enumerate(p.get("broll", [])[:2], 1):
                secs = max(2.0, min(3.5, b["seconds"]))
                at = max(1.0, min(a.length - secs - 1, b["at"]))
                f = broll.fetch(b["query"], work / f"broll_{n}_{k}.mp4", secs)
                if f:
                    broll_items.append({"file": f, "at": round(at, 2), "seconds": secs, "query": b["query"]})
        render_clip(p["video"], out, name, p["ss"], a.length, graph, captions, hook, broll_items, a.transitions)
        manifest.append({
            "file": f"{name}.mp4", "source": a.url, "source_title": info.get("title"),
            "channel": info.get("uploader"), "source_duration": info.get("duration"),
            "start": p["start"], "end": p["end"], "score": p["score"], "approved": False,
            "virality": p.get("virality"), "virality_detail": p.get("detail"), "reason": p.get("reason"),
            "visual": p.get("visual"), "ai_title": p.get("ai_title"), "hook": headline, "layout": used,
            "broll": [{"query": b["query"], "at": b["at"]} for b in broll_items],
        })
        print(f"  {name}.mp4  ({p['start']}s-{p['end']}s)" + (f"  virality {p['virality']}" if p.get("virality") is not None else ""))
    (out / "clips.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    progress.report(1.0, "Clips ready", force=True)
    print(f"Done. {len(manifest)} clips in {out}/")


if __name__ == "__main__":
    main()
