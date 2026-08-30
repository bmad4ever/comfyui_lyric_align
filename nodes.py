"""ComfyUI nodes for lyric forced alignment and transcript patching."""

import json
import os

from . import aligner

CAT = "audio/lyrics"


class LyricsLoadFile:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "path": ("STRING", {"default": "lyrics.txt", "multiline": False,
                                "tooltip": "Path to a .txt lyric sheet. Relative paths "
                                           "resolve against ComfyUI/input/."}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics",)
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = "Read a lyric sheet from disk."

    @classmethod
    def IS_CHANGED(cls, path):
        p = cls._resolve(path)
        return os.path.getmtime(p) if os.path.exists(p) else float("nan")

    @staticmethod
    def _resolve(path):
        if os.path.isabs(path):
            return path
        import folder_paths
        return os.path.join(folder_paths.get_input_directory(), path)

    def run(self, path):
        p = self._resolve(path)
        if not os.path.exists(p):
            raise FileNotFoundError("Lyric sheet not found: %s" % p)
        with open(p, encoding="utf-8") as f:
            return (f.read(),)


class LyricForcedAlign:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "The FULL track, not a clip. Slice afterwards "
                                               "with Lyric Slice By Time."}),
                "lyrics": ("STRING", {"multiline": True, "default": "",
                                      "tooltip": "Ground-truth lyric sheet, one line per "
                                                 "lyric line. Blank lines are kept as "
                                                 "section breaks."}),
            },
            "optional": {
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "chunk_seconds": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 600.0, "step": 5.0,
                    "tooltip": "0 = one pass over the whole track (fine to ~2 min on 24 GB). "
                               "Set 60 for long tracks: wav2vec2 attention is quadratic in "
                               "length, so a full song can exhaust VRAM in a single pass."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("alignment", "lrc", "info")
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = ("Force-align a known lyric sheet onto audio with torchaudio MMS_FA. "
                   "Gives every ground-truth word a timestamp.")

    def run(self, audio, lyrics, device="cuda", chunk_seconds=0.0):
        a = aligner.align(audio, lyrics, device=device, chunk_seconds=chunk_seconds)
        info = ("aligned %d words / %d lines over %.2f s on %s\n"
                "median score %.3f  (low is NORMAL on singing - MMS_FA is a speech model, "
                "and CTC alignment is order-constrained, so the path is still right)"
                % (len(a["words"]), len(a["lines"]), a["duration"], a["device"],
                   a["median_score"]))
        return (aligner.dumps(a), aligner.to_lrc(a["lines"]), info)


class LyricSliceByTime:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "alignment": ("STRING", {"forceInput": True}),
                "start_index": ("FLOAT", {
                    "default": 0.0, "min": -100000.0, "max": 100000.0, "step": 0.01,
                    "tooltip": "Clip start in seconds. Negative counts back from the end - "
                               "same convention as TrimAudioDuration."}),
                "duration": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 100000.0,
                                       "step": 0.01}),
            },
            "optional": {
                "mode": (["overlap", "contained"], {
                    "default": "overlap",
                    "tooltip": "overlap: keep a line if enough of it falls in the window "
                               "(see min_overlap). contained: only lines that fit entirely "
                               "inside."}),
                "min_overlap": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 60.0, "step": 0.05,
                    "tooltip": "Seconds of a line that must sit inside the window for it to "
                               "count, in overlap mode. Without this a line starting 0.08 s "
                               "before the window ends is 'in' the clip while the audio holds "
                               "barely a syllable of it. 0 restores bare any-overlap."}),
                "rebase_lrc": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Make the LRC timestamps relative to the clip start rather "
                               "than the full track."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "lrc", "info")
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = ("Pull the ground-truth lyric lines covering a time window out of an "
                   "alignment. Repeated choruses are unambiguous here - time separates them.")

    def run(self, alignment, start_index, duration, mode="overlap", min_overlap=0.3,
            rebase_lrc=True):
        a = json.loads(alignment)
        start = start_index + a["duration"] if start_index < 0 else start_index
        end = start + duration
        if mode == "contained":
            rows = [r for r in a["lines"] if r["start"] >= start and r["end"] <= end]
        else:
            rows = []
            for r in a["lines"]:
                shared = min(end, r["end"]) - max(start, r["start"])
                # A line shorter than min_overlap can never clear the bar on its own, so let
                # it qualify by being wholly inside the window instead.
                if shared > 0 and (shared >= min_overlap
                                   or shared >= r["end"] - r["start"]):
                    rows.append(r)
        text = "\n".join(r["text"] for r in rows)
        lrc = aligner.to_lrc(rows, offset=start if rebase_lrc else 0.0)
        info = ("window %.2f - %.2f s of %.2f s | %d/%d lines (%s"
                % (start, end, a["duration"], len(rows), len(a["lines"]), mode))
        info += ", min_overlap %.2f s)" % min_overlap if mode == "overlap" else ")"
        return (text, lrc, info)


class LyricPatchTranscript:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "transcript": ("STRING", {"forceInput": True,
                                          "tooltip": "The noisy ASR output to repair."}),
                "lyrics": ("STRING", {"multiline": True, "default": "",
                                      "tooltip": "Ground-truth lyric sheet to match against."}),
            },
            "optional": {
                "min_score": ("FLOAT", {
                    "default": 55.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Below this partial-match score the transcript is passed "
                               "through untouched rather than replaced with a wrong section."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "report", "diff")
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = ("Text-only: find where a noisy transcript sits in a lyric sheet and "
                   "return the real lines, plus what was wrong. No audio needed - but a "
                   "repeated chorus can match the wrong copy.")

    def run(self, transcript, lyrics, min_score=55.0):
        text, rep = aligner.patch(transcript, lyrics, min_score=min_score)
        diff = "\n".join(rep.get("diff") or []) or "(no differences)"
        return (text, aligner.dumps({k: v for k, v in rep.items() if k != "diff"}), diff)


class LyricVocalPresence:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "The ISOLATED VOCAL stem, from VocalRemovalNode "
                                               "output 0."}),
            },
            "optional": {
                "reference": ("AUDIO", {
                    "tooltip": "The instrumental stem (VocalRemovalNode output 1). Connect it - "
                               "the vocal-to-instrumental ratio separates singing from "
                               "instrumental far more reliably than any absolute level."}),
                "threshold_db": ("FLOAT", {
                    "default": -5.0, "min": -60.0, "max": 30.0, "step": 0.5,
                    "tooltip": "vocal minus instrumental, in dB, at or above which the window "
                               "counts as sung. Measured on this project's track: sung windows "
                               "ran -1.0..+4.3 dB, non-sung -20.7..-8.7 dB. -5.0 is the midpoint "
                               "of that 7.7 dB gap."}),
                "floor_db": ("FLOAT", {
                    "default": -45.0, "min": -90.0, "max": 0.0, "step": 1.0,
                    "tooltip": "Absolute RMS fallback, used only when no reference is connected. "
                               "Weaker: a quiet sung passage and a noisy silent one overlap."}),
            },
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "STRING")
    RETURN_NAMES = ("has_vocal", "ratio_db", "info")
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = ("Is anyone singing in this window? Measures the vocal stem against the "
                   "instrumental stem. Use this rather than trusting a transcript - ASR "
                   "hallucinates words over instrumental passages.")

    def run(self, audio, reference=None, threshold_db=-5.0, floor_db=-45.0):
        has, ratio, info = aligner.vocal_presence(audio, reference, threshold_db, floor_db)
        return (bool(has), float(ratio), info)



NODE_CLASS_MAPPINGS = {
    "LyricsLoadFile": LyricsLoadFile,
    "LyricForcedAlign": LyricForcedAlign,
    "LyricSliceByTime": LyricSliceByTime,
    "LyricPatchTranscript": LyricPatchTranscript,
    "LyricVocalPresence": LyricVocalPresence,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LyricsLoadFile": "Lyrics Load File",
    "LyricForcedAlign": "Lyric Forced Align",
    "LyricSliceByTime": "Lyric Slice By Time",
    "LyricPatchTranscript": "Lyric Patch Transcript",
    "LyricVocalPresence": "Lyric Vocal Presence",
}
