"""Forced alignment of a known lyric sheet onto audio, plus fuzzy text matching.

The audio path uses torchaudio's MMS_FA CTC aligner. It is a speech model, so confidence
scores on sung vocals are low (median ~0.2 is normal) - but the alignment PATH is still
correct, because CTC forced alignment is constrained to emit the words you gave it in the
order you gave them. Do not treat a low score as a failed alignment.

torchaudio.load is broken in this ComfyUI install (missing libtorchcodec DLLs), so nothing
here touches it - audio arrives as a ComfyUI AUDIO dict and is resampled in-memory.
"""

import json
import re
import unicodedata
from difflib import SequenceMatcher

import torch
import torchaudio

BUNDLE = torchaudio.pipelines.MMS_FA
SAMPLE_RATE = BUNDLE.sample_rate          # 16000
FRAME_STRIDE = 320                        # wav2vec2 downsampling factor
# MMS_FA's vocabulary: 26 latin letters plus the apostrophe. Anything else - digits,
# punctuation, accented characters - has no token and must be stripped before tokenizing.
ALPHABET = set("abcdefghijklmnopqrstuvwxyz'")

_MODELS = {}


def _model(device):
    """Load once per device and keep it. The checkpoint is 1.18 GB."""
    if device not in _MODELS:
        m = BUNDLE.get_model(with_star=False).to(device).eval()
        _MODELS[device] = (m, BUNDLE.get_tokenizer(), BUNDLE.get_aligner())
    return _MODELS[device]


def normalize(word):
    """Fold a lyric word down to the aligner's alphabet. May return ''."""
    w = unicodedata.normalize("NFKD", word).lower()
    w = w.replace("’", "'").replace("ʼ", "'")
    return "".join(c for c in w if c in ALPHABET)


def tokenize_lyrics(text):
    """Split a lyric sheet into lines and alignable words.

    Returns (lines, tokens) where lines is a list of the original line strings (blank lines
    kept, so section breaks survive) and tokens is a list of
    {"line": line_index, "word": original word, "norm": alignable form}.

    Words that normalize to nothing - a bare "3", "(x2)", an em dash - are dropped from the
    token stream but stay in the line text, so the output keeps them.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    tokens = []
    for i, line in enumerate(lines):
        for word in line.split():
            n = normalize(word)
            if n:
                tokens.append({"line": i, "word": word, "norm": n})
    return lines, tokens


def to_mono_16k(audio):
    """ComfyUI AUDIO dict -> 1-D float tensor at 16 kHz."""
    wav = audio["waveform"]
    if wav.dim() == 3:                    # (batch, channels, samples)
        wav = wav[0]
    if wav.dim() == 2:                    # (channels, samples)
        wav = wav.mean(0)
    wav = wav.float().cpu()
    sr = int(audio["sample_rate"])
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav


def emissions(wav, device, chunk_seconds=0.0, overlap_seconds=2.0):
    """Log-probs over time, plus the wall-clock start time of every emission frame.

    wav2vec2 attends over every frame at once, so a long track is quadratic in memory. Set
    chunk_seconds to process in windows instead; emission frames are frame-local, so the
    windows are simply concatenated with half the overlap trimmed off each inner edge.
    chunk_seconds=0 runs a single pass - fine up to a couple of minutes on a 24 GB card.

    The frame->time table is returned rather than derived from the total frame count,
    because trimming seam frames removes real time from the middle of the track. Dividing
    duration by the surviving frame count spreads that loss evenly over the whole song and
    drifts words near the seams by seconds - measured at 3.0 s on a 72 s track with 30 s
    chunks. Timing each part from its own sample offset makes chunked and single-pass agree.
    """
    model = _model(device)[0]
    total = wav.shape[0]

    if chunk_seconds <= 0 or total <= int(chunk_seconds * SAMPLE_RATE):
        with torch.inference_mode():
            em, _ = model(wav[None].to(device))
        em = em[0].cpu()
        # Single pass: spread the true duration over the frames the model actually returned,
        # which absorbs its edge padding.
        times = torch.arange(em.shape[0], dtype=torch.float64) * (total / SAMPLE_RATE / em.shape[0])
        return em, times

    chunk = int(chunk_seconds * SAMPLE_RATE)
    overlap = int(overlap_seconds * SAMPLE_RATE)
    step = chunk - overlap
    trim = overlap // 2 // FRAME_STRIDE           # frames to drop per inner edge
    parts, stamps, pos = [], [], 0
    while pos < total:
        piece = wav[pos:pos + chunk]
        with torch.inference_mode():
            em, _ = model(piece[None].to(device))
        em = em[0].cpu()
        head = trim if pos > 0 else 0
        tail = trim if pos + chunk < total else 0
        keep = em[head:em.shape[0] - tail] if tail else em[head:]
        # Frame f of this part covers samples starting at pos + f * stride within the piece.
        idx = torch.arange(head, head + keep.shape[0], dtype=torch.float64)
        stamps.append((pos + idx * FRAME_STRIDE) / SAMPLE_RATE)
        parts.append(keep)
        pos += step
    return torch.cat(parts, dim=0), torch.cat(stamps, dim=0)


def align(audio, lyrics, device="cuda", chunk_seconds=0.0):
    """Force-align a lyric sheet onto audio. Returns the alignment dict."""
    lines, tokens = tokenize_lyrics(lyrics)
    if not tokens:
        raise ValueError("No alignable words in the lyrics. MMS_FA only knows a-z and '.")

    device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
    wav = to_mono_16k(audio)
    duration = wav.shape[0] / SAMPLE_RATE
    em, frame_times = emissions(wav, device, chunk_seconds)

    _, tokenizer, aligner = _model(device)
    spans = aligner(em, tokenizer([t["norm"] for t in tokens]))

    last = len(frame_times) - 1

    def at(frame):
        return float(frame_times[min(int(frame), last)])

    words = []
    for i, (tok, span) in enumerate(zip(tokens, spans)):
        span_len = sum(s.end - s.start for s in span) or 1
        words.append({
            "i": i,
            "line": tok["line"],
            "word": tok["word"],
            "start": round(at(span[0].start), 3),
            "end": round(at(span[-1].end), 3),
            "score": round(sum(s.score * (s.end - s.start) for s in span) / span_len, 3),
        })

    line_rows = []
    for idx, text in enumerate(lines):
        mine = [w for w in words if w["line"] == idx]
        if not mine:
            continue                       # blank line or unalignable - no timing to give
        line_rows.append({"line": idx, "text": text.strip(),
                          "start": mine[0]["start"], "end": mine[-1]["end"],
                          "score": round(sum(w["score"] for w in mine) / len(mine), 3)})

    scores = sorted(w["score"] for w in words)
    return {
        "duration": round(duration, 3),
        "device": device,
        "words": words,
        "lines": line_rows,
        "median_score": round(scores[len(scores) // 2], 3),
    }


def to_lrc(rows, offset=0.0):
    """Line rows -> LRC text. offset is subtracted, for rebasing onto a clip."""
    out = []
    for r in rows:
        t = max(0.0, r["start"] - offset)
        out.append("[%02d:%05.2f]%s" % (int(t // 60), t % 60, r["text"]))
    return "\n".join(out)


# ------------------------------------------------------------------ text-only matching
def _words(s):
    return re.findall(r"[a-z']+", s.lower())


def patch(transcript, lyrics, min_score=55.0):
    """Locate a noisy transcript inside a lyric sheet and return the real lines.

    Two stages. rapidfuzz's partial_ratio_alignment finds the rough character span, then the
    span is narrowed to the lines that actually earned it: a line is kept only if at least one
    of its words survives a word-level diff against the transcript.

    The second stage is not optional. The character span routinely runs past the real ending -
    it stops wherever the edit distance stops improving, which can be several words into the
    next line - and expanding that to whole lines hands back a lyric the clip never sang. On
    the 18-30 s test clip that spuriously appended the chorus. Requiring word-level evidence
    per line trims both ends honestly.

    Repeated choruses are still genuinely ambiguous here - that is the cost of not using the
    audio. LyricForcedAlign + LyricSliceByTime has no such problem.
    """
    from rapidfuzz import fuzz

    a = fuzz.partial_ratio_alignment(transcript.lower(), lyrics.lower())
    if a is None or a.score < min_score:
        return transcript, {
            "matched": False,
            "score": round(a.score, 1) if a else 0.0,
            "min_score": min_score,
            "note": "below min_score - transcript returned unchanged",
        }

    # Stage 1: widen the character span to whole lines, as a generous candidate region.
    start = lyrics.rfind("\n", 0, a.dest_start) + 1
    end = lyrics.find("\n", a.dest_end)
    end = len(lyrics) if end < 0 else end
    lines = lyrics[:end].split("\n")
    first_line = lyrics[:start].count("\n")

    # Stage 2: keep only lines carrying a word the transcript actually matched.
    heard = _words(transcript)
    region, owner = [], []
    for li in range(first_line, len(lines)):
        for w in _words(lines[li]):
            region.append(w)
            owner.append(li)

    hit = set()
    for blk in SequenceMatcher(None, heard, region).get_matching_blocks():
        for k in range(blk.size):
            hit.add(owner[blk.b + k])

    if hit:
        lo, hi = min(hit), max(hit)     # contiguous: keep interior lines that were misheard
        keep = lines[lo:hi + 1]
    else:
        lo, hi = first_line, len(lines) - 1
        keep = lines[lo:hi + 1]
    matched = "\n".join(keep).strip()

    truth = _words(matched)
    diffs = []
    for op, i1, i2, j1, j2 in SequenceMatcher(None, heard, truth).get_opcodes():
        if op == "equal":
            continue
        diffs.append("%-8s heard %-40r truth %r"
                     % (op, " ".join(heard[i1:i2]), " ".join(truth[j1:j2])))

    return matched, {
        "matched": True,
        "score": round(a.score, 1),
        "lines": [lo, hi],
        "words_heard": len(heard),
        "words_truth": len(truth),
        "mismatches": len(diffs),
        "diff": diffs,
    }


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False, indent=1)


# ------------------------------------------------------------------ vocal presence
def rms_db(waveform):
    """RMS of a ComfyUI AUDIO waveform, in dBFS."""
    import math
    x = waveform
    if x.dim() == 3:
        x = x[0]
    v = float(x.float().pow(2).mean().sqrt())
    return 20.0 * math.log10(max(v, 1e-9))


def vocal_presence(audio, reference=None, threshold_db=-5.0, floor_db=-45.0):
    """Is anyone actually singing in this window?

    Measured, not transcribed. Whisper hallucinates confidently over instrumental passages - on
    this project's own test track it invented a sign-off line over the outro - so "the transcript
    came back non-empty" is not evidence of a vocal.

    Compare the isolated vocal stem against the instrumental stem rather than against an absolute
    level. Absolute level does not separate: measured on the test track a quiet sung intro sat at
    -21.9 dB while a non-sung window sat at -22.9 dB. The RATIO does separate, and widely -
    sung windows ran -1.0..+4.3 dB, non-sung -20.7..-8.7 dB, a 7.7 dB gap. Hence the -5.0 default,
    which is the midpoint.

    With no reference connected it falls back to an absolute floor, which is the weaker test.
    """
    v = rms_db(audio["waveform"])
    if reference is not None:
        r = rms_db(reference["waveform"])
        ratio = v - r
        has = ratio >= threshold_db
        info = ("vocal %.1f dB, instrumental %.1f dB, ratio %+.1f dB (threshold %+.1f) -> %s"
                % (v, r, ratio, threshold_db, "SINGING" if has else "INSTRUMENTAL"))
        return has, ratio, info
    has = v >= floor_db
    info = ("vocal %.1f dB, no reference connected so using the absolute floor %.1f dB -> %s\n"
            "connect the instrumental stem for the reliable ratio test"
            % (v, floor_db, "SINGING" if has else "INSTRUMENTAL"))
    return has, v, info
