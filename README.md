# comfyui-lyric-align

You have the real lyrics. Whisper's transcript of the song is wrong. These nodes fix it,
and tell you when nobody was singing at all.

## Nodes

| node | does |
|---|---|
| **Lyrics Load File** | read a `.txt` lyric sheet (relative paths → `ComfyUI/input/`) |
| **Lyric Forced Align** | audio + real lyrics → a timestamp for every ground-truth word |
| **Lyric Slice By Time** | alignment + start/duration → the real lines for that window |
| **Lyric Patch Transcript** | noisy transcript + lyrics → the real lines. No audio, instant |
| **Lyric Vocal Presence** | vocal stem vs instrumental stem → `has_vocal` BOOLEAN |

## Which patcher

**Align + Slice** if you can spare the audio pass — it locates lyrics by *time*, so four identical
chorus lines stop being ambiguous. **Patch Transcript** to clean a transcript you already have;
it matches on text alone, so a repeated chorus can match the wrong copy. Below `min_score` it
returns the transcript untouched rather than guessing.

## Don't ask the transcript whether anyone is singing

Whisper hallucinates over instrumental passages. **Lyric Vocal Presence** measures instead:
connect the vocal stem to `audio` and the instrumental to `reference`, then gate on `has_vocal`.
Use the ratio, not an absolute level — absolute level does not separate the two.

## Notes

- Confidence scores ~0.2 on singing are normal. CTC alignment can't reorder your words, only
  place them. Judge the LRC, not the scores.
- Only `a-z` and `'` align. Other characters survive in the text but get no timestamp.
- Set `chunk_seconds 60` for a full song; `0` (one pass) is fine to ~2 minutes.
- Needs `torchaudio` + `rapidfuzz`. First align downloads 1.18 GB.
