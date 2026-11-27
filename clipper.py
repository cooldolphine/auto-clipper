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
import director
import edit
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


def pick_formats(info, height):
    """Best video (<= height) and best audio stream URLs from yt-dlp's format list."""
    fm = [f for f in (info.get("formats") or []) if f.get("url")]
    is_v = lambda f: f.get("vcodec") not in (None, "none")
    is_a = lambda f: f.get("acodec") not in (None, "none")
    vids = [f for f in fm if is_v(f) and not is_a(f) and 0 < (f.get("height") or 0) <= height and (f.get("protocol") or "").startswith("http")]
    vids = vids or [f for f in fm if is_v(f) and 0 < (f.get("height") or 0) <= height]
    best_v = max(vids, key=lambda f: (f.get("height") or 0, str(f.get("vcodec", "")).startswith("avc1"), f.get("tbr") or 0), default=None)
    if best_v is None or is_a(best_v):
        return best_v, None
    auds = [f for f in fm if is_a(f) and not is_v(f)]
    best_a = max(auds, key=lambda f: (f.get("ext") == "m4a", f.get("abr") or f.get("tbr") or 0), default=None)
    return best_v, best_a


def fetch_section(info, out_path, start, end, height):
    """Cut [start, end] of the video straight from the stream URLs with ffmpeg. Returns the path or None."""
    v, au = pick_formats(info, height)
    if not v:
        return None
    dur = end - start

    def src(f):
        cmd = ["-ss", f"{start:.2f}", "-t", f"{dur:.2f}"]
        if f.get("http_headers"):
            cmd += ["-headers", "".join(f"{k}: {x}\r\n" for k, x in f["http_headers"].items())]
        return cmd + ["-i", f["url"]]

    cmd = ["ffmpeg", "-y", "-v", "error"] + src(v)
    if au:
        cmd += src(au) + ["-map", "0:v:0", "-map", "1:a:0"]
    else:
        cmd += ["-map", "0:v:0", "-map", "0:a:0?"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-shortest", str(out_path)]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=900)
    except Exception:
        return None
    p = Path(out_path)
    if r.returncode != 0 or not p.exists() or p.stat().st_size < 10000:
        print("    ffmpeg said:", r.stderr.decode(errors="ignore").strip()[-200:])
        return None
    return p


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


EMPH_TAG = "\\fscx120\\fscy120\\1c&H0050FF&"   # bigger and orange-red
EMPH_END = "{\\fscx100\\fscy100\\1c&H00E5FF&}"  # back to normal yellow


def write_ass_words(path, words, length, margin=230, emphasis=()):
    """Word-by-word captions at the bottom. words are clip-relative; emphasised words get bigger and red-orange."""
    words = [w for w in words if w["start"] < length]
    if not words:
        return False
    key = lambda t: re.sub(r"\W", "", str(t)).lower()
    emph = {key(e) for e in emphasis if key(e)}
    lines = []
    for chunk in chunk_words(words):
        t0 = chunk[0]["start"]
        t1 = min(chunk[-1]["end"], length)
        parts = []
        for i, w in enumerate(chunk):
            nxt = chunk[i + 1]["start"] if i + 1 < len(chunk) else max(w["end"], w["start"] + 0.15)
            hot = key(w["word"]) in emph
            parts.append("{\\k%d%s}%s%s" % (max(1, round((nxt - w["start"]) * 100)), EMPH_TAG if hot else "", ass_text(w["word"]),
                                           EMPH_END if hot else ""))
        lines.append(f"Dialogue: 0,{ass_time(t0)},{ass_time(t1 + 0.05)},Cap,,0,0,0,,{' '.join(parts)}")
    path.write_text(ass_head(margin) + "\n".join(lines) + "\n", encoding="utf-8")
    return True


def words_to_segments(words, per=6):
    segs = []
    for i in range(0, len(words), per):
        g = words[i:i + per]
        segs.append({"start": g[0]["start"], "end": g[-1]["end"], "text": " ".join(w["word"] for w in g)})
    return segs


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
    if path.exists() and path.stat().st_size > 10000:
        return str(path)
    import urllib.request
    for attempt in range(3):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(YUNET_URL, path)
            if path.stat().st_size > 10000:
                return str(path)
        except Exception as e:
            print(f"    could not download the face model (try {attempt + 1}): {type(e).__name__}")
    return None


class Faces:
    """Face detector: YuNet if its model can be downloaded, otherwise OpenCV's older Haar detector."""

    def __init__(self):
        self.kind, self.yn, self.haar = None, None, None
        try:
            import cv2
        except ImportError:
            print("    face detection is not installed")
            return
        self.cv2 = cv2
        model = yunet_model()
        if model and hasattr(cv2, "FaceDetectorYN"):
            try:
                self.yn = cv2.FaceDetectorYN.create(model, "", (320, 320), 0.6, 0.3, 200)
                self.kind = "YuNet"
            except Exception as e:
                print(f"    YuNet could not start: {e}")
        if self.yn is None:
            try:
                h = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                if not h.empty():
                    self.haar, self.kind = h, "Haar (less accurate)"
            except Exception:
                pass
        print(f"    face detector: {self.kind or 'none available'}")

    @property
    def ok(self):
        return self.kind is not None

    def detect(self, bgr, min_score=0.6):
        """Faces in a frame as [(cx, cy, w, h, score)], all as fractions of the frame (0..1)."""
        cv2 = self.cv2
        H, W = bgr.shape[:2]
        sw = min(W, 960)
        sh = max(1, int(H * sw / W))
        small = cv2.resize(bgr, (sw, sh)) if sw != W else bgr
        out = []
        if self.yn is not None:
            self.yn.setInputSize((sw, sh))
            _, faces = self.yn.detect(small)
            for f in (faces if faces is not None else []):
                if float(f[14]) >= min_score:
                    out.append(((f[0] + f[2] / 2) / sw, (f[1] + f[3] / 2) / sh, f[2] / sw, f[3] / sh, float(f[14])))
        elif self.haar is not None:
            for (x, y, w, h) in self.haar.detectMultiScale(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 1.1, 6, minSize=(int(0.03 * sw), int(0.03 * sw))):
                out.append(((x + w / 2) / sw, (y + h / 2) / sh, w / sw, h / sh, 1.0))
        return out


_FACES = []


def get_faces():
    if not _FACES:
        _FACES.append(Faces())
    return _FACES[0]


def video_size(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    w, h = r.stdout.strip().split(",")[:2]
    return int(w), int(h)


def grab_bgr(src, t, headers=None, width=960):
    jpg = grab_frame(src, t, headers, width)
    if not jpg:
        return None
    import cv2
    return cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)


def cluster_facecam(per_frame):
    """Find the facecam: a face that sits in the SAME place in many frames spread over the whole video.
    Game characters move around, so they never form such a group."""
    n = len(per_frame)
    if n < 4:
        return None
    clusters = []
    for fi, dets in enumerate(per_frame):
        for (cx, cy, w, h, _) in dets:
            for c in clusters:
                m = c["m"]
                if abs(cx - m[0]) < 0.06 and abs(cy - m[1]) < 0.08 and 0.6 < w / m[2] < 1.6:
                    c["boxes"].append((fi, cx, cy, w, h))
                    k = len(c["boxes"])
                    c["m"] = [m[i] + (v - m[i]) / k for i, v in enumerate((cx, cy, w, h))]
                    break
            else:
                clusters.append({"boxes": [(fi, cx, cy, w, h)], "m": [cx, cy, w, h]})
    best = None
    for c in clusters:
        frames = len({b[0] for b in c["boxes"]})
        med = lambda i: sorted(b[i] for b in c["boxes"])[len(c["boxes"]) // 2]
        cx, cy, w, h = med(1), med(2), med(3), med(4)
        edge = min(cx, 1 - cx) < 0.3 and min(cy, 1 - cy) < 0.4  # sits near a corner, like a facecam overlay
        score = frames * (1.35 if edge else 1.0)
        if frames >= max(3, 0.3 * n) and (best is None or score > best[0]):
            best = (score, {"cx": cx, "cy": cy, "w": w, "h": h, "frames": frames, "of": n, "edge": edge})
    return best[1] if best else None


def find_facecam(src, headers, duration, n=12):
    """Look at n frames spread over the video and find the streamer's facecam."""
    faces = get_faces()
    if not faces.ok or not src or not duration:
        return None
    per_frame = []
    for i in range(n):
        img = grab_bgr(src, duration * (0.06 + 0.88 * i / (n - 1)), headers)
        if img is not None:
            per_frame.append(faces.detect(img, 0.8))
    got = sum(1 for d in per_frame if d)
    cam = cluster_facecam(per_frame)
    print(f"    looked at {len(per_frame)} frames, faces seen in {got}; " +
          (f"facecam found at x={cam['cx']:.2f}, y={cam['cy']:.2f} (face width {cam['w']:.2f}), in {cam['frames']} frames"
           if cam else "no facecam found"))
    return cam


def facecam_check_image(path, src, headers, duration, cam):
    """A picture you can look at to see what was detected (saved next to the clips)."""
    try:
        import cv2
        img = grab_bgr(src, duration * 0.5, headers)
        if img is None:
            return
        H, W = img.shape[:2]
        if cam:
            x0, y0 = int((cam["cx"] - cam["w"] / 2) * W), int((cam["cy"] - cam["h"] / 2) * H)
            x1, y1 = int((cam["cx"] + cam["w"] / 2) * W), int((cam["cy"] + cam["h"] / 2) * H)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 3)
        cv2.putText(img, "facecam found" if cam else "no facecam found", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0) if cam else (0, 0, 255), 3)
        cv2.imwrite(str(path), img)
    except Exception:
        pass


def prepare_facecam(a, src, headers, duration, out):
    """Decide where the facecam is, once per video: by hand (--facecam tl/tr/bl/br/none) or by looking."""
    a.facecam_info = None
    if a.framing not in ("stack", "face", "auto"):
        return
    if a.facecam == "none":
        a.facecam_info = {"none": True}
    elif a.facecam in ("tl", "tr", "bl", "br"):
        a.facecam_info = {"preset": a.facecam, "size": a.facecam_size}
        print(f"    facecam set by hand: {a.facecam}")
    else:
        progress.report(a.ai_lo - 0.05, "Looking for the facecam")
        a.facecam_info = find_facecam(src, headers, duration)
        facecam_check_image(out / "facecam_check.jpg", src, headers, duration or 60, a.facecam_info)


def face_samples(video, ss, length, prior=None, step=0.5, follow=False):
    """Face position every `step` seconds: [(t, (cx, cy, face_width) in pixels or None)], W, H.
    With a prior (where the facecam is), only faces near it count, never game characters elsewhere."""
    faces = get_faces()
    if not faces.ok:
        return None
    import cv2
    cap = cv2.VideoCapture(str(video))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not (W and H):
        return None
    samples, t, prev, lost = [], 0.0, None, 0.0
    while t < length:
        cap.set(cv2.CAP_PROP_POS_MSEC, (ss + t) * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        dets = faces.detect(frame, 0.6)
        pick = None
        if dets and follow and prev is not None:  # keep following the same person, never jump to someone else
            d = min(dets, key=lambda d: (d[0] * W - prev[0]) ** 2 + (d[1] * H - prev[1]) ** 2)
            if abs(d[0] * W - prev[0]) < 0.3 * W and abs(d[1] * H - prev[1]) < 0.35 * H:
                pick = d
            elif lost >= 1.5:  # lost them for a while: start again with the most prominent face
                pick = max(dets, key=lambda d: d[2] * d[3])
        elif dets and prior:
            d = min(dets, key=lambda d: (d[0] * W - prior[0]) ** 2 + (d[1] * H - prior[1]) ** 2)
            if abs(d[0] * W - prior[0]) < 0.2 * W and abs(d[1] * H - prior[1]) < 0.25 * H:
                pick = d
        elif dets:
            pick = max(dets, key=lambda d: d[2] * d[3])
        if pick is not None:
            prev, lost = (pick[0] * W, pick[1] * H), 0.0
        else:
            lost += step
        samples.append((t, None if pick is None else (pick[0] * W, pick[1] * H, pick[2] * W)))
        t += step
    cap.release()
    return W, H, samples


def face_track(video, ss, length, prior=None, follow=False):
    """The face's path over the clip: {W, H, fw, t, cx, cy} or None if there is no steady face."""
    got = face_samples(video, ss, length, prior, follow=follow)
    if not got:
        return None
    W, H, samples = got
    hits = [s for s in samples if s[1]]
    print(f"    face tracking: face found in {len(hits)} of {len(samples)} moments")
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


def face_counts(video, ss, length, step=1.0):
    """How many people (clear faces) are in view, once per second. None if there is no face detector."""
    faces = get_faces()
    if not faces.ok:
        return None
    import cv2
    cap = cv2.VideoCapture(str(video))
    counts, t = [], 0.0
    while t < length:
        cap.set(cv2.CAP_PROP_POS_MSEC, (ss + t) * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        counts.append(sum(1 for d in faces.detect(frame, 0.7) if d[2] >= 0.04))
        t += step
    cap.release()
    return counts


def decide_layout(video, ss, length, facecam):
    """Pick the layout for this clip: face on top (facecam), follow the face (one person), or the whole
    frame on a blurred background (a crowd, or no clear person). Returns (mode, why)."""
    if facecam and facecam.get("preset"):
        return "stack", f"facecam set by hand ({facecam['preset']})"
    if facecam and "cx" in facecam and facecam["w"] < 0.12:
        return "stack", "a facecam overlay was found"
    counts = face_counts(video, ss, length)
    if not counts:
        return "blur", "no faces could be checked"
    n = len(counts)
    multi = sum(1 for c in counts if c >= 2) / n
    single = sum(1 for c in counts if c == 1) / n
    if multi >= 0.3:
        return "blur", f"several people are in view ({multi:.0%} of the clip)"
    if single >= 0.6:
        return "face", f"one person is in view ({single:.0%} of the clip)"
    return "blur", "no single person stays in view"


def with_zoom(graph, zoom):
    """Zoom the whole 1080x1920 frame (graphs end with [v0])."""
    if not zoom:
        return graph
    return graph[:-4] + "," + edit.zoom_filter(1080, 1920, zoom["at"], zoom["seconds"]) + "[v0]"


def game_x(W, gw, cam):
    """Where to crop the gameplay horizontally: the middle, but away from the facecam if it is in the way."""
    gx = (W - gw) // 2
    if not cam:
        return gx
    x0, y0, x1, y1 = cam
    if x1 <= gx or x0 >= gx + gw:
        return gx
    cand = min(W - gw, int(x1) + 10) if (x0 + x1) / 2 < W / 2 else max(0, int(x0) - gw - 10)
    return cand if 0 <= cand <= W - gw else gx


def layout_graph(mode, video, ss, length, zoom=None, facecam=None):
    """Returns (ffmpeg graph from [0:v] to [v0], headline-on-the-seam?, layout actually used).
    zoom = {at, seconds, target} punches in on the funny moment.
    facecam = where the streamer's facecam is: found by looking, set by hand, or {"none": True}."""
    if mode == "auto":
        mode, why = decide_layout(video, ss, length, facecam)
        print(f"    automatic layout: {mode} ({why})")
        if mode == "face" and facecam and facecam.get("none"):
            facecam = None
    if mode == "blur":
        return with_zoom(BLUR, zoom), False, "blur"
    if mode in ("stack", "face"):
        fallback = (with_zoom(BLUR, zoom), False, "blur") if mode == "stack" else (with_zoom(CENTER, zoom), False, "center")
        if facecam and facecam.get("none"):
            print("    no facecam in this video: using the " + fallback[2] + " layout")
            return fallback
        if facecam is None and mode == "stack":
            print("    no facecam found in this video: using the " + fallback[2] + " layout (you can set its corner in the app)")
            return fallback
        W, H = video_size(video)
        clamp = lambda v, lo, hi: max(lo, min(hi, v))
        if mode == "face" and facecam and "cx" in facecam and facecam["w"] < 0.12:
            print("    the face is a small facecam overlay, so this uses the face-on-top layout instead")
            mode = "stack"
        preset = facecam.get("preset") if facecam else None
        if preset and mode == "face":
            mode = "stack"
        tr = None
        if not preset:
            prior = (facecam["cx"] * W, facecam["cy"] * H) if facecam and "cx" in facecam else None
            follow_one = mode == "face" and prior is None  # no fixed facecam: follow one person
            if follow_one:
                print("    no fixed facecam: following the most prominent face")
            tr = face_track(video, ss, length, prior, follow=follow_one)
            if not tr and prior:  # not seen in this clip: stay where the facecam was found
                tr = {"W": W, "H": H, "fw": facecam["w"] * W, "t": [0.0, float(length)], "cx": [prior[0]] * 2, "cy": [prior[1]] * 2}
            if not tr:
                print("    no steady face found, using the " + fallback[2] + " layout")
                return fallback
        if mode == "face":
            cw = int(H * 9 / 16) // 2 * 2
            cxs = follow(tr["cx"], 0.12 * cw, 0.3 * cw)
            xe = path_expr(tr["t"], [clamp(c - cw / 2, 0, W - cw) for c in cxs])
            print("    face found, the crop follows it")
            return with_zoom(f"[0:v]crop={cw}:{H}:x='{xe}':y=0,scale=1080:1920[v0]", zoom), False, "face"
        if preset:  # a fixed rectangle in the chosen corner
            pw = int(W * facecam.get("size", 0.26)) // 2 * 2
            ph = int(pw * TOP_H / 1080) // 2 * 2
            m = int(0.01 * W)
            px = m if preset in ("tl", "bl") else W - pw - m
            py = m if preset in ("tl", "tr") else H - ph - m
            xe, ye, rect = str(px), str(py), (px, py, px + pw, py + ph)
            print(f"    facecam taken from the {preset} corner")
        else:
            pw = min(W, max(tr["fw"] * 3.0, H * 0.35))  # the facecam is about 3 face-widths wide
            ph = pw * TOP_H / 1080
            if ph > H:
                ph, pw = H, H * 1080 / TOP_H
            pw, ph = int(pw) // 2 * 2, int(ph) // 2 * 2
            cxs = follow(tr["cx"], 0.12 * pw, 0.3 * pw)
            cys = follow(tr["cy"], 0.12 * ph, 0.3 * ph)
            xs = [clamp(c - pw / 2, 0, W - pw) for c in cxs]
            ys = [clamp(c - ph * 0.45, 0, H - ph) for c in cys]
            xe, ye = path_expr(tr["t"], xs), path_expr(tr["t"], ys)
            rect = (min(xs), min(ys), max(xs) + pw, max(ys) + ph)
            print("    face found: face on top (the crop follows it), gameplay below")
        gw = min(W, int(H * 1080 / BOT_H)) // 2 * 2
        gh = int(gw * BOT_H / 1080) // 2 * 2
        gx = game_x(W, gw, rect)
        ztop = ("," + edit.zoom_filter(1080, TOP_H, zoom["at"], zoom["seconds"])) if zoom and zoom.get("target") == "face" else ""
        zbot = ("," + edit.zoom_filter(1080, BOT_H, zoom["at"], zoom["seconds"])) if zoom and zoom.get("target") != "face" else ""
        graph = (f"[0:v]split[t][g];[t]crop={pw}:{ph}:x='{xe}':y='{ye}',scale=1080:{TOP_H}{ztop}[top];"
                 f"[g]crop={gw}:{gh}:{gx}:0,scale=1080:{BOT_H}{zbot}[bot];[top][bot]vstack[v0]")
        return graph, True, "stack"
    return with_zoom(CENTER, zoom), False, "center"


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


def render_clip(video, outdir, name, ss, length, graph, captions=None, hook=None, broll_items=(), transitions="off", crf=19,
                emoji_items=(), margin=230):
    """captions = ("srt"|"ass", file), hook = ass file, broll_items = [{file, at, seconds}]."""
    kept = plan_cutaways(broll_items, length)
    inputs = ["-ss", str(ss), "-t", str(length), "-i", str(Path(video).resolve())]
    for b in kept:
        inputs += ["-t", str(b["seconds"] + 1.0), "-i", str(Path(b["file"]).resolve())]
    for em in emoji_items:
        inputs += ["-loop", "1", "-framerate", "30", "-t", str(em["seconds"] + 0.3), "-i", str(Path(em["file"]).resolve())]
    parts, cur = [graph], "v0"
    if kept:
        more, cur = (hard_cut_parts(cur, kept) if transitions == "off"
                     else xfade_parts(cur, kept, length, transitions))
        parts += more
    if transitions != "off":
        color = ":color=white" if transitions == "flash" else ""
        parts.append(f"[{cur}]fade=t=in:st=0:d=0.3{color},fade=t=out:st={length - 0.3:.2f}:d=0.3[vf]")
        cur = "vf"
    for j, em in enumerate(emoji_items):
        idx, at, sec = 1 + len(kept) + j, em["at"], em["seconds"]
        parts.append(f"[{idx}:v]scale=170:-1,format=rgba,fade=t=in:st=0:d=0.15:alpha=1,fade=t=out:st={sec - 0.2:.2f}:d=0.2:alpha=1,"
                     f"setpts=PTS-STARTPTS+{at:.3f}/TB[em{j}]")
        parts.append(f"[{cur}][em{j}]overlay=x=(W-w)/2:y=H-{margin + 330}:eof_action=pass:enable='between(t,{at:.3f},{at + sec:.3f})'[oe{j}]")
        cur = f"oe{j}"
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


def choose(cands, a, info, frame_src=None, k=None):
    """First pass: AI scores (text + still frames) if available, otherwise sound/chat/keyword scores. Returns k moments."""
    k = k or a.num
    ranked = None
    pool = sorted(cands, key=lambda c: c["score"], reverse=True)[:12]  # the AI judges the 12 most promising
    if a.ai == "on":
        progress.report(a.ai_lo, "Looking at the best moments")
        add_frames(pool, a.length, frame_src)
        items = [{"id": i, "start": c["start"], "end": c["end"], "text": c.get("text", ""), "energy": c["score"],
                  "images": c.get("images", [])} for i, c in enumerate(pool)]
        progress.report(a.ai_lo + 0.03, "AI is scoring the moments")
        ranked = ai_pick.rank(items, info.get("title"), info.get("uploader"), k, a.broll == "auto",
                              ai_pick.performance_notes())
        if not ranked:
            print("  AI picking not available, using the sound/chat/keyword picks")
    if ranked:
        for i, c in enumerate(pool):
            r = ranked.get(i)
            if r:
                c.update(virality=r["virality"], detail=r["detail"], reason=r["reason"], ai_title=r["title"],
                         headline=r["headline"], visual=r.get("visual", ""),
                         broll=[{"at": c["start"] + b["at"], "seconds": b["seconds"], "query": b["query"]} for b in r["broll"]])
        top = sorted(pool, key=lambda c: (c.get("virality", -1), c["score"]), reverse=True)[:k]
    else:
        top = sorted(cands, key=lambda c: c["score"], reverse=True)[:k]
    for c in cands:
        c.pop("images", None)
    return sorted(top, key=lambda c: c["start"])


def lines_for(segments, t0, t1):
    """Speech with times (seconds from t0) for the director."""
    words = [{"start": w["start"] - t0, "end": w["end"] - t0, "word": w["word"]}
             for s in segments for w in s.get("words", []) if t0 <= w["start"] < t1]
    return "\n".join(f"[{ch[0]['start']:.1f}] " + " ".join(w["word"] for w in ch) for ch in chunk_words(words, 6, 3.0, 0.8))


def direct(finalists, a, info, segments, work):
    """Second pass (the director): Gemini watches a preview of each finalist, then we apply its exact cut,
    edit cues and title/description/tags. Returns the final picks."""
    ctx = a.ctx
    total = info.get("duration") or 10 ** 9
    pad = 6.0
    items = []
    for i, f in enumerate(finalists):
        pv0 = max(0.0, f["start"] - pad)
        pv1 = min(float(total), f["end"] + pad)
        progress.report(a.ai_lo + 0.04 + 0.08 * i / max(1, len(finalists)), f"Making a preview of moment {i + 1} of {len(finalists)}")
        prev = director.make_preview(ctx["src"], work / f"preview_{i}.mp4", pv0, pv1 - pv0, ctx["headers"],
                                     ctx["wav"] if ctx["audio_sep"] else None)
        items.append({"id": i, "dur": pv1 - pv0, "pv0": pv0, "preview": prev, "lines": lines_for(segments, pv0, pv1)})
    print(f"  previews ready: {sum(1 for x in items if x['preview'])} of {len(items)}")
    progress.report(a.ai_lo + 0.13, "The AI director is watching the moments")
    res = director.review(items, info.get("title"), info.get("uploader"), a.broll == "auto", ai_pick.performance_notes()) or {}
    for it, f in zip(items, finalists):
        r = res.get(it["id"])
        if not r:
            continue
        pv0 = it["pv0"]
        f.update(start=pv0 + r["start"], end=pv0 + r["end"], peak=pv0 + r["peak"], director=True, keep=r["keep"],
                 funny=r["funny"], virality=r["virality"], reason=r["reason"] or f.get("reason"), visual=r["visual"] or f.get("visual"),
                 headline=r["headline"] or f.get("headline"), game=r["game"], titles=r["titles"], description=r["description"],
                 tags=r["tags"], emphasis=r["emphasis"], emoji=[{"at": pv0 + e["at"], "emoji": e["emoji"]} for e in r["emoji"]])
        f["zoom"] = r["zoom"] and {**r["zoom"], "at": pv0 + r["zoom"]["at"]}
        f["slowmo"] = r["slowmo"] and {**r["slowmo"], "at": pv0 + r["slowmo"]["at"]}
        if a.broll == "auto":
            f["broll"] = [{**b, "at": pv0 + b["at"]} for b in r["broll"]]
    pool = [f for f in finalists if f.get("keep", True)] or finalists
    rank_key = lambda f: (f.get("funny", f.get("virality", 0) or 0) * 0.6 + (f.get("virality") or 0) * 0.4, f["score"])
    top = sorted(pool, key=rank_key, reverse=True)[: a.num]
    if len(top) < a.num:  # not enough kept: fill with the best of the rest
        rest = [f for f in finalists if f not in top]
        top += sorted(rest, key=rank_key, reverse=True)[: a.num - len(top)]
    return sorted(top, key=lambda f: f["start"])


def short_pipeline(a, out, work):
    a.ai_lo, a.cut_lo = 0.66, 0.76
    print("1/5 Downloading...")
    progress.report(0.0, "Downloading video")
    video, info = download(a.url, work, a.cookies, a.height, stage=progress.stage(0.0, 0.30))
    print("2/5 Extracting audio...")
    progress.report(0.31, "Extracting audio")
    wav = work / "audio.wav"
    extract_audio(video, wav)
    a.wav = wav
    a.ctx = {"src": str(video), "headers": None, "wav": wav, "audio_sep": False}
    rms = loudness_per_second(wav)
    prepare_facecam(a, str(video), None, info.get("duration") or len(rms), out)
    print("3/5 Transcribing (slow on CPU)...")
    tstage = progress.stage(0.34, 0.65)
    segments = transcribe(wav, a.model, a.language, duration=len(rms), on_progress=lambda f: tstage(f, f"Transcribing {int(f * 100)}%"))
    print("4/5 Scoring moments...")
    chat = chat_scores(work, len(rms))
    print("  chat replay:", "found" if chat is not None else "none")
    score = build_scores(len(rms), rms, segments, info, chat)
    cands = pick_windows(score, a.length, max(a.num * 3, a.num + 4))
    attach_text(cands, segments)
    use_director = a.ai == "on" and a.director == "on"
    picks = choose(cands, a, info, (str(video), None), k=(a.num + 3) if use_director else None)
    if use_director:
        picks = direct(picks, a, info, segments, work)
    set_windows(picks, a, info)
    for p in picks:
        p["video"], p["t0"] = video, 0.0
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
    a.wav = wav
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
        run(["ffmpeg", "-y", "-ss", str(max(0, c["start"] - 6)), "-t", str(a.length + 12), "-i", str(wav), "-c", "copy", str(sl)])
        segs = transcribe_wav(model, sl, a.language, offset=max(0, c["start"] - 6))
        segments += segs
        c["score"] += 3 * sum(keyword_hits(s["text"]) for s in segs if s["end"] >= c["start"] and s["start"] <= c["end"])
        print(f"  candidate {i}/{len(cands)} done")
    attach_text(cands, segments)
    remote = remote_source(info)
    cam_src = remote_source(info, 720) or remote
    prepare_facecam(a, cam_src[0] if cam_src else None, cam_src[1] if cam_src else None, info.get("duration") or len(rms), out)
    a.ctx = {"src": remote[0] if remote else None, "headers": remote[1] if remote else None, "wav": wav, "audio_sep": True}
    use_director = a.ai == "on" and a.director == "on" and remote is not None
    picks = choose(cands, a, info, remote, k=(a.num + 3) if use_director else None)
    if use_director:
        picks = direct(picks, a, info, segments, work)
    set_windows(picks, a, info)
    print(f"4/5 Downloading only the {len(picks)} chosen sections of the video...")
    ok = []
    for n, p in enumerate(picks, 1):
        progress.report(0.62 + 0.17 * (n - 1) / max(1, len(picks)), f"Downloading section {n} of {len(picks)}")
        path = fetch_section(info, work / f"seg_{n:02}.mp4", p["win0"], p["win1"] + 1, a.height)
        if not path:
            print(f"  section {n}: direct download failed, trying yt-dlp")
            try:
                path = download_segment(a.url, work, n, p["win0"], p["win1"] + 1, a.cookies, a.height)
            except Exception as e:
                print(f"  section {n}: could not be downloaded ({str(e)[:160]}), skipping it")
                continue
        p["video"], p["t0"] = path, p["win0"]
        ok.append(p)
    if not ok:
        raise RuntimeError("none of the chosen sections could be downloaded")
    return info, ok, segments


def to_rel(cue, start):
    return cue and {**cue, "at": cue["at"] - start}


def set_windows(picks, a, info):
    """How much of the video each clip needs: its funny moment plus room to reach the target length."""
    total = float(info.get("duration") or 10 ** 9)
    target = min(float(a.length), 59.0)
    for p in picks:
        s, e = p["start"], p["end"]
        if a.length_mode == "target":
            span = min(total, target * 1.7 + 4)  # extra video, because cutting silence makes clips shorter
            need = max(0.0, span - (e - s))
            w0, w1 = max(0.0, s - need * 0.35), min(total, e + need * 0.65)
            short = span - (w1 - w0)
            if short > 0.1:  # one side ran out of video: use more of the other side
                if w0 <= 0:
                    w1 = min(total, w1 + short)
                else:
                    w0 = max(0.0, w0 - short)
        else:
            w0, w1 = s, e
        p["win0"], p["win1"] = w0, w1


def plan_clip(a, p, segments, s, e):
    """Plan the edits for the clip [s, e] (absolute seconds) and what its length will be afterwards."""
    length = e - s
    words = [{"start": w["start"] - s, "end": w["end"] - s, "word": w["word"]}
             for sg in segments for w in sg.get("words", []) if s - 0.05 <= w["start"] < e]
    rms, step = edit.rms_track(a.wav, s, e)
    zoom, slowmo = None, None
    if a.punch == "on":
        z, sm = to_rel(p.get("zoom"), s), to_rel(p.get("slowmo"), s)
        inside = lambda c: c and 0.3 <= c["at"] <= length - 1.0
        zoom, slowmo = (z if inside(z) else None), (sm if inside(sm) else None)
        if not zoom and not slowmo:  # no AI cue: use the loudest moment
            peak = edit.plan_peak(rms, step, length)
            zoom = {"at": max(0.5, peak - 0.3), "seconds": 1.6, "target": "gameplay"}
            slowmo = {"at": max(0.5, peak - 0.1), "seconds": 1.2}
    cuts = []
    if a.cut_silence == "on":
        protect = [(c["at"] - 0.3, c["at"] + c["seconds"] + 0.3) for c in (zoom, slowmo) if c]
        cuts = edit.plan_cuts(words, rms, step, length, protect)
    slows = []
    if slowmo:
        s0 = max(0.0, min(slowmo["at"], length - 1.0))
        slows = [(s0, min(length, s0 + slowmo["seconds"]), 0.5)]
    return {"length": length, "words": words, "rms": rms, "step": step, "zoom": zoom, "slowmo": slowmo,
            "cuts": cuts, "tl": edit.Timeline(length, cuts, slows)}


def choose_window(a, p, segments):
    """Final start/end so that the EDITED clip lasts about --length seconds (never more than 59)."""
    lo, hi = p["win0"], p["win1"]
    s0, e0 = p["start"], p["end"]
    if a.length_mode != "target":
        return s0, e0, plan_clip(a, p, segments, s0, e0)
    target = min(float(a.length), 59.0)
    peak = p.get("peak") or (s0 + e0) / 2
    L, best = min(target, hi - lo), None
    for _ in range(5):
        if L >= e0 - s0:  # keep the director's moment inside, with a bit more before than after
            s = s0 - (L - (e0 - s0)) * 0.35
        else:
            s = peak - L * 0.4
        s = max(lo, min(s, hi - L))
        pl = plan_clip(a, p, segments, s, s + L)
        err = pl["tl"].total - target
        if best is None or abs(err) < abs(best[3]):
            best = (s, s + L, pl, err)
        if abs(err) < 0.6:
            break
        newL = max(6.0, min(hi - lo, L - err))
        if abs(newL - L) < 0.2:
            break
        L = newL
    return best[0], best[1], best[2]


def clip_edits(a, p, segments, work, name, out):
    """Everything between 'we know the moment' and 'ready to style': length, cut silence, slow motion, cues.
    Returns (video, ss, length, timeline, words_new, cues)."""
    s, e, pl = choose_window(a, p, segments)
    p["start"], p["end"] = s, e
    ss = s - p["t0"]
    tl, length, words, rms, step, zoom = pl["tl"], pl["length"], pl["words"], pl["rms"], pl["step"], pl["zoom"]
    if tl.identity:
        video, vss, new_len = p["video"], ss, length
    else:
        raw = out / f"{name}_raw.mp4"
        edit.assemble(p["video"], raw, ss, length, tl)
        video, vss, new_len = raw, 0, tl.total
    mwords = [{"start": tl.map(w["start"]), "end": max(tl.map(w["end"]), tl.map(w["start"]) + 0.05), "word": w["word"]} for w in words]
    mzoom = None
    if zoom:
        t0 = tl.map(zoom["at"])
        mzoom = {"at": t0, "seconds": max(0.8, tl.map(min(length, zoom["at"] + zoom["seconds"])) - t0), "target": zoom.get("target", "gameplay")}
    emph, emoji = [], []
    if a.emoji == "on":
        emph = p.get("emphasis") or edit.loud_words(words, rms, step)
        for em in p.get("emoji") or []:
            at = em["at"] - s
            if 0.5 <= at <= length - 1.0:
                emoji.append({"at": tl.map(at), "emoji": em["emoji"], "seconds": 1.4})
    broll_items = []
    for br in p.get("broll") or []:
        at = br["at"] - s
        if 1.0 <= at <= length - 3.0:
            broll_items.append({"at": tl.map(at), "seconds": br["seconds"], "query": br["query"]})
    return video, vss, new_len, tl, mwords, {"zoom": mzoom, "emphasis": emph, "emoji": emoji, "broll": broll_items,
                                             "cuts": len(pl["cuts"]), "slowmo": bool(pl["slowmo"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--num", type=int, default=5, help="number of clips")
    ap.add_argument("--length", type=int, default=30, help="clip length in seconds (Shorts: 58)")
    ap.add_argument("--length-mode", default="target", choices=["target", "ai"],
                    help="target: every clip is as close to --length as possible; ai: the AI decides, up to --length")
    ap.add_argument("--out", default="output")
    ap.add_argument("--model", default="base", help="whisper model: tiny/base/small")
    ap.add_argument("--language", default=None, help="e.g. hi for Hindi, en for English (default: auto)")
    ap.add_argument("--framing", default="auto", choices=["auto", "stack", "blur", "face", "center"], help="how to make it vertical")
    ap.add_argument("--captions", default="karaoke", choices=["karaoke", "classic", "off"], help="caption style")
    ap.add_argument("--facecam", default="auto", choices=["auto", "tl", "tr", "bl", "br", "none"],
                    help="where the facecam is: auto-detect, a corner (top/bottom left/right), or none")
    ap.add_argument("--facecam-size", type=float, default=0.26, help="facecam width as a share of the video width (corner mode)")
    ap.add_argument("--caption-margin", type=int, default=230, help="caption distance from the bottom edge (pixels of 1920)")
    ap.add_argument("--hook", default="auto", choices=["auto", "off"], help="headline in the first seconds")
    ap.add_argument("--ai", default="on", choices=["on", "off"], help="AI moment picking and virality scores")
    ap.add_argument("--director", default="on", choices=["on", "off"], help="AI director: watches previews, cuts exactly, writes titles")
    ap.add_argument("--punch", default="on", choices=["on", "off"], help="zoom and slow motion on the funny moment")
    ap.add_argument("--cut-silence", default="on", choices=["on", "off"], help="remove silent gaps")
    ap.add_argument("--emoji", default="on", choices=["on", "off"], help="emoji pop-ups and highlighted words")
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
        progress.report(a.cut_lo + (1 - a.cut_lo) * (n - 1) / max(1, len(picks)), f"Editing clip {n} of {len(picks)}")
        name = f"clip_{n:02}"
        video, ss, length, tl, mwords, cues = clip_edits(a, p, segments, work, name, out)
        graph, seam, used = layout_graph(a.framing, video, ss, length, cues["zoom"], a.facecam_info)
        captions = None
        if a.captions == "karaoke" and write_ass_words(out / f"{name}.ass", mwords, length, a.caption_margin, cues["emphasis"]):
            captions = ("ass", f"{name}.ass")
        elif a.captions in ("classic", "karaoke") and mwords:
            (out / f"{name}.srt").write_text("", encoding="utf-8")
            if write_srt(out / f"{name}.srt", words_to_segments(mwords), 0, length):
                captions = ("srt", f"{name}.srt")  # classic style, or fallback when no word timings
        headline = p.get("headline") or make_hook_text(info.get("title"))
        hook = None
        if a.hook == "auto" and write_hook_ass(out / f"{name}_hook.ass", headline, seam=seam):
            hook = f"{name}_hook.ass"
        broll_items = []
        if a.broll == "auto":
            for k, b in enumerate(cues["broll"][:2], 1):
                f = broll.fetch(b["query"], work / f"broll_{n}_{k}.mp4", b["seconds"])
                if f:
                    broll_items.append({"file": f, "at": round(b["at"], 2), "seconds": b["seconds"], "query": b["query"]})
        emoji_items = []
        for e in cues["emoji"]:
            f = edit.emoji_file(e["emoji"], work)
            if f:
                emoji_items.append({"file": f, "at": e["at"], "seconds": e["seconds"]})
        render_clip(video, out, name, ss, length, graph, captions, hook, broll_items, a.transitions,
                    emoji_items=emoji_items, margin=a.caption_margin)
        (out / f"{name}_raw.mp4").unlink(missing_ok=True)
        manifest.append({
            "file": f"{name}.mp4", "source": a.url, "source_title": info.get("title"),
            "channel": info.get("uploader"), "source_duration": info.get("duration"),
            "start": p["start"], "end": p["end"], "length": round(length, 1), "score": p["score"], "approved": False,
            "virality": p.get("virality"), "virality_detail": p.get("detail"), "reason": p.get("reason"),
            "visual": p.get("visual"), "ai_title": p.get("ai_title"), "hook": headline, "layout": used,
            "ai_titles": p.get("titles") or ([p["ai_title"]] if p.get("ai_title") else []),
            "ai_description": p.get("description"), "ai_tags": p.get("tags") or [], "game": p.get("game"),
            "edits": {"cuts": cues["cuts"], "slowmo": cues["slowmo"], "zoom": bool(cues["zoom"]), "emoji": len(emoji_items),
                      "highlighted": len(cues["emphasis"])},
            "broll": [{"query": b["query"], "at": b["at"]} for b in broll_items],
        })
        print(f"  {name}.mp4  ({p['start']:.1f}s-{p['end']:.1f}s, {length:.1f}s)" + (f"  virality {p['virality']}" if p.get("virality") is not None else ""))
    (out / "clips.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    progress.report(1.0, "Clips ready", force=True)
    print(f"Done. {len(manifest)} clips in {out}/")


if __name__ == "__main__":
    main()
