"""Editing helpers: cut out silence, slow-motion on the key moment, a punch-in zoom,
highlighted words and emoji pop-ups. All times are seconds from the start of the clip."""
import subprocess
import wave
from pathlib import Path

import numpy as np


# ---------- timeline: cut pieces out / slow pieces down, and map old times to new times ----------
class Timeline:
    def __init__(self, length, cuts=(), slows=()):
        """cuts: [(start, end)] removed; slows: [(start, end, speed)] e.g. speed 0.5 = half speed."""
        pts = {0.0, float(length)}
        for s, e in cuts:
            pts |= {max(0.0, s), min(length, e)}
        for s, e, _ in slows:
            pts |= {max(0.0, s), min(length, e)}
        pts = sorted(pts)
        segs = []
        for a, b in zip(pts[:-1], pts[1:]):
            if b - a < 0.02:
                continue
            mid = (a + b) / 2
            if any(s <= mid < e for s, e in cuts):
                continue
            speed = next((sp for s, e, sp in slows if s <= mid < e), 1.0)
            if segs and segs[-1][2] == speed and abs(segs[-1][1] - a) < 1e-6:
                segs[-1] = (segs[-1][0], b, speed)
            else:
                segs.append((a, b, speed))
        self.segments = segs or [(0.0, float(length), 1.0)]
        self.total = sum((e - s) / sp for s, e, sp in self.segments)
        self.length = float(length)

    @property
    def identity(self):
        return len(self.segments) == 1 and self.segments[0][2] == 1.0 and abs(self.segments[0][1] - self.segments[0][0] - self.length) < 0.05

    def map(self, t):
        """Where an old time lands on the new timeline (times inside a removed piece land on its end)."""
        acc = 0.0
        for s, e, sp in self.segments:
            if t < s:
                return acc
            if t < e:
                return acc + (t - s) / sp
            acc += (e - s) / sp
        return acc

    def scale_at(self, t):
        for s, e, sp in self.segments:
            if s <= t < e:
                return 1.0 / sp
        return 1.0


# ---------- loudness over time ----------
def rms_track(wav_path, start, end, step=0.1):
    """Loudness of wav[start:end] in `step`-second pieces (16 kHz mono wav)."""
    with wave.open(str(wav_path)) as w:
        sr = w.getframerate()
        w.setpos(min(w.getnframes(), int(max(0, start) * sr)))
        raw = w.readframes(int((end - start) * sr))
    d = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    n = int(step * sr)
    k = len(d) // n
    if k == 0:
        return np.zeros(1, dtype=np.float32), step
    return np.sqrt((d[: k * n].reshape(k, n) ** 2).mean(axis=1)), step


# ---------- planning ----------
def plan_cuts(words, rms, step, length, protect=(), min_gap=0.9, keep=0.25, max_share=0.3):
    """Silent gaps to remove. A gap counts only if nobody speaks AND it is quiet (action sounds stay)."""
    if not words:
        return []
    ref = float(np.percentile(rms, 90)) if len(rms) else 0.0  # how loud the clip gets at its loudest

    def quiet(a, b):
        """Quiet = far below the loud parts. Constant music or game sound keeps a gap from being cut."""
        i, j = int(a / step), max(int(a / step) + 1, int(b / step))
        r = rms[i:j]
        inner = r[2:-2] if len(r) > 6 else r  # ignore the fade at the edges of the gap
        return len(rms) == 0 or int((inner > 0.18 * ref).sum()) <= 1

    def protected(a, b):
        return any(a < pe and b > ps for ps, pe in protect)

    gaps = []
    first, last = words[0], words[-1]
    if first["start"] > 1.2:
        gaps.append((0.0, first["start"] - 0.4))
    for w1, w2 in zip(words[:-1], words[1:]):
        if w2["start"] - w1["end"] >= min_gap:
            gaps.append((w1["end"] + keep, w2["start"] - keep))
    if length - last["end"] > 1.6:
        gaps.append((last["end"] + 0.7, length))
    cuts, removed = [], 0.0
    for a, b in gaps:
        if b - a < 0.4 or protected(a, b) or not quiet(a, b):
            continue
        if removed + (b - a) > max_share * length:
            break
        cuts.append((a, b))
        removed += b - a
    return cuts


def plan_peak(rms, step, length):
    """Fallback 'funny moment': the loudest second (not at the very start or end)."""
    if len(rms) < 5:
        return length / 2
    lo, hi = int(1.5 / step), max(int(1.5 / step) + 1, len(rms) - int(2.0 / step))
    sm = np.convolve(rms, np.ones(5) / 5, mode="same")
    return float((lo + int(np.argmax(sm[lo:hi]))) * step)


def loud_words(words, rms, step, share=0.12):
    """Fallback emphasis: the loudest words."""
    scored = []
    for w in words:
        if len(w["word"]) < 3:
            continue
        i, j = int(w["start"] / step), max(int(w["start"] / step) + 1, int(w["end"] / step))
        if i < len(rms):
            scored.append((float(rms[i:j].mean()), w["word"]))
    scored.sort(reverse=True)
    return [w for _, w in scored[: max(1, int(len(words) * share))]]


# ---------- ffmpeg: pass 1 (cut / slow down) ----------
def has_audio(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


def assemble(video, out_path, ss, length, tl):
    """Re-cut the clip following the timeline (silence removed, slow-motion). Returns the new file."""
    n = len(tl.segments)
    audio = has_audio(video)
    parts = [f"[0:v]fps=30,settb=AVTB,setsar=1,split={n}" + "".join(f"[s{i}]" for i in range(n))]
    if audio:
        parts.append(f"[0:a]aresample=48000,asplit={n}" + "".join(f"[t{i}]" for i in range(n)))
    for i, (s, e, sp) in enumerate(tl.segments):
        parts.append(f"[s{i}]trim=start={s:.3f}:end={e:.3f},setpts=(PTS-STARTPTS)/{sp}[v{i}]")
        if audio:
            parts.append(f"[t{i}]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS" + (f",atempo={sp}" if sp != 1 else "") + f"[a{i}]")
    inputs = "".join(f"[v{i}]" + (f"[a{i}]" if audio else "") for i in range(n))
    parts.append(f"{inputs}concat=n={n}:v=1:a={1 if audio else 0}[vo]" + ("[ao]" if audio else ""))
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", str(ss), "-t", str(length), "-i", str(Path(video).resolve()),
           "-filter_complex", ";".join(parts), "-map", "[vo]"] + (["-map", "[ao]"] if audio else []) + [
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p"] + (["-c:a", "aac", "-b:a", "192k"] if audio else []) + [str(out_path)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return Path(out_path)


# ---------- ffmpeg: pass 2 helpers (zoom, emoji) ----------
def zoom_expr(at, dur, amount=0.18, rise=0.25, fall=0.35):
    """Zoom factor over time t: eases in, holds, eases out."""
    t1 = max(at + 0.1, at + dur - fall)
    return f"(1+{amount}*clip((t-{at:.3f})/{rise},0,1)*(1-clip((t-{t1:.3f})/{fall},0,1)))"


def zoom_filter(w, h, at, dur):
    z = zoom_expr(at, dur)
    return f"scale=w='trunc({w}*{z}/2)*2':h='trunc({h}*{z}/2)*2':eval=frame,crop={w}:{h}"


TWEMOJI = "https://cdn.jsdelivr.net/gh/jdecked/twemoji@15.1.0/assets/72x72/{}.png"


def emoji_file(emoji, workdir):
    """Download the picture of an emoji (Twemoji). Returns a path or None."""
    import requests
    name = "-".join(f"{ord(c):x}" for c in emoji if ord(c) != 0xFE0F)
    dest = Path(workdir) / f"emoji_{name}.png"
    if dest.exists():
        return dest
    try:
        r = requests.get(TWEMOJI.format(name), timeout=20)
        if r.status_code == 200 and r.content[:4] == b"\x89PNG":
            dest.write_bytes(r.content)
            return dest
    except Exception:
        pass
    return None
