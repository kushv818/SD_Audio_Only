# audio_interaction_pipeline.py

Estimate **who's talking to whom, when, and for how long**, from multiple
close-talk audio recordings of a group conversation — using only the raw
waveforms. No transcription, no speech recognition, no language models.
Handles cases where two conversations are happening **at the same time**
(e.g. two side conversations in one room), and where **who's paired with
whom shifts over the course of a longer recording** (e.g. small groups
changing during a classroom session) — it doesn't assume one fixed pairing
holds for the entire file.

---

## The idea

Each person wears their own recorder (e.g. chest-worn, LENA-style). Every
recorder picks up *everyone* in the room, but the wearer is always the
loudest voice on their own channel (closest mic = highest signal), while
other speakers bleed through much more quietly. The whole pipeline is
built on that one physical fact:

1. **Align** the recordings onto one shared timeline (recorders are
   started by hand, so they rarely start at exactly the same millisecond).
2. **Detect who's speaking**, independently per channel, so two people in
   separate simultaneous conversations can both register as speaking at
   once — not forced into a single "loudest wins" decision.
3. **Segment** that into individual speech turns.
4. **Discover partnerships within time windows** (default every 3 minutes,
   not once for the whole recording), by combining three acoustic
   signals: bleed-through proximity, energy-envelope correlation, and
   turn-taking rhythm — plus a **confidence score** per window that tells
   you whether a pairing is a clear match or just the best of a weak
   field. Because discovery re-runs per window, pairings can change
   through the recording instead of being locked in from the start.
5. **Resolve each turn** to its likely addressee(s) — one person, or
   several at once if someone's addressing the group — blending local
   evidence with whichever window's partnership applies to that turn.
6. **Output** a per-turn timeline, merged conversation blocks, aggregate
   duration tables, and diagnostic plots/CSVs.

---

## Installation

```bash
pip install numpy scipy pandas matplotlib networkx
```

No audio-specific libraries (no `librosa`, no `ffmpeg`) — it reads plain
PCM `.wav` files directly.

**Optional** (only needed for `--use-voice-embeddings`):

```bash
pip install resemblyzer
```

---

## Quick start

```bash
python audio_interaction_pipeline.py --input-dir ./recordings --output-dir ./results
```

That's it for a first pass. Point `--input-dir` at a folder containing one
`.wav` file per person and it does everything else.

### File naming

Expects files named like `LENA_UNIT_U5_ALICIA_TRIAL1.wav`. The speaker
name is auto-detected as the last purely-alphabetic underscore-separated
token that isn't a known boilerplate word (`LENA`, `UNIT`, `TRIAL`, `DAY`,
`SESSION`, ...). Check the "Loaded ... -> speaker=" lines it prints at
startup to confirm all names parsed correctly. If your naming is
different, use `--name-index` to force which underscore-separated field
is the name (e.g. `--name-index 3`).

---

## Using this for children / classroom recordings

This works the same way for a classroom as for adults, with a few things
worth knowing:

- **One recorder per child is required.** The pipeline identifies who's
  bleeding into whom based on each channel belonging to a known person -
  if recordings instead come from one shared room microphone, this
  pipeline doesn't apply; that's a different problem (blind diarization +
  voice enrollment), not covered here.
- **Set `--partner-window-s` to match how often groups actually change**
  (e.g. `120`-`300` for centers/small-group rotations, larger for a longer
  single activity). The default (`180`s) is a reasonable starting point,
  not a validated recommendation for your specific classroom routine.
- **Pretrained voice-embedding models (`--use-voice-embeddings`) are
  trained mostly on adult voices.** Children's higher pitch and more
  variable voice quality may reduce its accuracy - worth checking
  `voice_similarity.csv` before trusting it, more so than with adults.
- **More movement and background noise than a quiet adult conversation**
  will affect every acoustic signal here, not just one - expect to spend
  more time tuning `--competitive-margin-db` and `--hangover-ms` than you
  would for a calmer recording.
- With many children rather than 4 people, partnership discovery still
  runs (max-weight matching handles any even number, leaving one person
  unmatched if the group is odd), but the "one stable partner" framing
  fits less naturally as group size grows past a handful. Confidence
  scores become more important to watch in that case, not less.

---

## What to check first in the output

1. **Console output**, near the top — confirm every file's speaker name
   parsed correctly.
2. **`Discovering partnerships in N time window(s)`** — the key result,
   printed per window. Each pair gets an affinity score and a confidence
   tier:
   - **high confidence** (≥0.6) — a clear, well-separated match
   - **medium confidence** (0.35–0.6) — reasonably supported
   - **LOW CONFIDENCE** (<0.35) — just the best option left over, not a
     validated pairing; treat with real skepticism
     If pairings differ between windows, that's the pipeline picking up a
     real shift in who's grouped with whom over time - check
     `partnership_windows.csv` for the full detail.
3. **`NOTE` warnings** — if a person correlates weakly with *everyone*
   else in the recording, the pipeline flags it. This usually means a
   hardware/positioning issue with that specific recorder (lower gain,
   farther placement, more clothing noise), not that the person lacks a
   real partner. Worth listening to that file directly.
4. **`partnership_tracks.png`** — visual check: does it show clean,
   continuous rows for each person's *most common* partner? (A summary
   across the whole recording — see `partnership_windows.csv` for how
   that changes over time.)
5. **`interaction_timeline.csv`** / printed blocks — the actual "who's
   talking to whom, when" detail.

---

## What it produces

| File                                             | What it is                                                                                                                                                  |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `segments.csv`                                 | Every detected speech turn: speaker, start, end, duration                                                                                                   |
| `speaking_time.csv`                            | Total speaking time / turn count per person                                                                                                                 |
| `interaction_timeline.csv`                     | **Main output** — every turn resolved to its addressee(s), with per-listener share                                                                   |
| `interaction_blocks.csv`                       | Consecutive turns between the same people merged into readable blocks (e.g.`00:00–00:30 Alicia <-> Kush`)                                                |
| `interaction_timeline.png`                     | Gantt-style chart of those blocks over time                                                                                                                 |
| `talk_duration_seconds.csv` / `_percent.csv` | Aggregate A→B seconds/percent matrix across the whole recording                                                                                            |
| `talk_duration_heatmap.png` / `_network.png` | Visualizations of the above                                                                                                                                 |
| `turn_taking_matrix.csv` + heatmap/network     | Simple turn-adjacency counts (who follows whom)                                                                                                             |
| `partnership_windows.csv`                      | **Time-varying pairings** — every discovered pair, per time window, with its affinity score and confidence                                           |
| `envelope_correlation.csv`                     | Raw pairwise loudness-correlation matrix (whole recording) — a quick, cheap sanity check independent of everything else                                    |
| `voice_similarity.csv`                         | *(only with `--use-voice-embeddings`)* Average voice-match (vocal timbre, not loudness) between each pair of channels                                   |
| `partnership_tracks.png`                       | One row per person's*most common* partner across all windows — a quick visual summary (see `partnership_windows.csv` for the real time-varying detail) |
| `speech_timeline.png`                          | Simple per-person "who's talking when" chart                                                                                                                |

---

## Key settings

Two modes, chosen with `--vad-mode`:

- **`independent`** (default) — each channel's speech is detected against
  its own noise floor. Use this whenever more than one person might talk
  at a time, including two separate simultaneous conversations.
- **`competitive`** — one "loudest wins" speaker per instant across all
  channels. Simpler, and fine if you're confident only one person ever
  talks at a time in your recordings.

### If results look off, in order of what to try first

| Symptom                                                                          | Try                                                                                                                                                                                                          |
| -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Timeline looks scattered / rapidly flips between wrong pairs                     | Check`envelope_correlation.csv` and the confidence tiers first — a low-confidence pair usually means the acoustic signal genuinely can't separate that person's real partner from bystanders, not a bug   |
| Implausible amounts of "simultaneous" speech everywhere                          | Lower`--competitive-margin-db` (default `12`) — this is the main knob for the independent-mode detector                                                                                                 |
| Real simultaneous speakers getting dropped/missed                                | Raise`--competitive-margin-db`                                                                                                                                                                             |
| Speech getting fragmented into tiny unusable pieces                              | Raise`--hangover-ms` (default `150`)                                                                                                                                                                     |
| Partnerships look wrong specifically when two parallel conversations are running | Lower`--affinity-turn-weight` (default `0.5`) — turn-taking timing gets noisier with multiple simultaneous conversations                                                                                |
| Pairings seem to flip-flop between windows that should be the same group         | Raise`--partner-window-s` so each window has more data to work with                                                                                                                                        |
| A real shift in groups partway through isn't showing up                          | Lower`--partner-window-s` so windows are short enough to catch it                                                                                                                                          |
| One person shows weak correlation with everyone                                  | Likely a recorder/mic issue for that person specifically — check the raw file                                                                                                                               |
| A pairing stays ambiguous even with everything above                             | Try`--use-voice-embeddings` (see below) — it's a different, more specific acoustic feature (vocal timbre) than loudness/energy, and can help exactly when energy-based signals have run out of resolution |
| You know only one person ever talks at a time                                    | Use`--vad-mode competitive`                                                                                                                                                                                |

### Optional: voice-embedding confirmation (`--use-voice-embeddings`)

Every signal above is ultimately about **loudness** — how much of a channel's
energy is bleed-through from a given speaker. `--use-voice-embeddings` adds
a different kind of acoustic evidence: it builds a "voiceprint" for each
person from their own clean speech (using a pretrained speaker-embedding
model), then checks whether the bleed-through picked up on another
channel actually **sounds like** that person's voice — pitch, timbre,
resonance — rather than just being loud. This can confirm genuine
bleed-through and reject same-time noise or other sources that happen to
be loud but don't actually match, which is exactly the kind of ambiguity
that pure energy-based signals can't resolve on their own.

This is still 100% acoustic — no words, no transcription, no language
model, just a different feature extracted from the waveform.

```bash
pip install resemblyzer
python audio_interaction_pipeline.py --input-dir ./recordings --output-dir ./results \
    --use-voice-embeddings
```

Notes:

- First use downloads pretrained model weights (one-time, needs internet);
  runs fully offline after that.
- Adds real runtime cost (an embedding is computed for every segment on
  every other channel, up to `--max-voice-segments` per speaker, default
  `60`). Lower that if it's too slow; raise `--voiceprint-seconds` (default
  `10`) if a speaker's voiceprint seems unreliable from too little audio.
- New output: `voice_similarity.csv` — a person's average voice-match to
  each other channel, over the whole recording.
- **I could not fully validate the real-world accuracy of this feature**
  (no internet access in my environment to install/download the model) —
  I verified the code runs correctly end-to-end with a stand-in mock, but
  the actual matching quality depends on the real pretrained model. Try it
  and check whether `voice_similarity.csv` and the resulting confidence
  scores look sensible before trusting it on an ambiguous pairing.

---

## Full CLI reference

Run `python audio_interaction_pipeline.py --help` for the complete, current
list with defaults. Grouped summary:

**Input/output**

- `--input-dir` (required) — folder of `.wav` files
- `--pattern` (default `*.wav`) — glob pattern to select files
- `--output-dir` (default `./interaction_results`)
- `--name-index` — force which underscore-separated filename field is the speaker name
- `--sr` (default `16000`) — target sample rate

**Speech detection**

- `--vad-mode {independent, competitive}` (default `independent`)
- `--own-margin-db` (default `8.0`) — absolute noise-floor gate (independent mode)
- `--competitive-margin-db` (default `12.0`) — **main tuning knob**, relative-SNR gate (independent mode)
- `--hangover-ms` (default `150`) — bridges short natural pauses within one turn
- `--min-speech-ms` (default `200`) — discard turns shorter than this
- `--frame-ms` / `--hop-ms` (default `30` / `10`) — analysis frame size
- `--silence-db` / `--margin-db` — competitive-mode-only thresholds

**Alignment**

- `--align-window` (default `60.0`s) — audio used to estimate cross-channel lag
- `--max-shift` (default `10.0`s) — max expected misalignment

**Partnership discovery**

- `--partner-window-s` (default `180.0`) — re-discover partnerships every N seconds instead of once for the whole recording; a value ≥ the recording length collapses to a single window
- `--envelope-weight` (default `1.5`) — weight of the loudness-correlation signal; set `0` to disable
- `--envelope-smooth-ms` (default `0`) — smoothing for that signal (0 = tested best)
- `--affinity-turn-weight` (default `0.5`) — weight of turn-taking rhythm relative to proximity
- `--use-voice-embeddings` — additionally use speaker-embedding voice matching (see above); requires `pip install resemblyzer`
- `--voiceprint-seconds` (default `10.0`) — audio used to build each speaker's reference voiceprint
- `--max-voice-segments` (default `60`) — cap on segments embedded per speaker (runtime control)
- `--voice-affinity-weight` (default `2.0`) — weight of voice similarity in partnership discovery
- `--voice-similarity-weight` (default `3.0`) — per-turn boost weight from voice similarity
- `--noise-percentile` (default `50`) — percentile used to estimate each channel's noise floor
- `--turn-gap` (default `2.0`s) — max gap between turns to count as a response

**Per-turn resolution**

- `--partner-boost` (default `6.0`) — how strongly a discovered stable partner is favored per turn (auto-scaled by confidence)
- `--no-partner-prior` — disable that boost, resolve every turn from local evidence only
- `--context-weight` (default `1.5`) — weight given to whoever spoke immediately before/after
- `--dominant-share` (default `0.55`) — share needed for a turn to resolve as directed 1:1
- `--group-share` (default `0.20`) — share needed to be included as a simultaneous addressee (group/circle talk)
- `--block-overlap-ratio` (default `0.6`) — how much consecutive turns must overlap in participants to merge into one block

---

## Honest limitations

- This is a lightweight acoustic heuristic, not full speaker diarization.
  It works well when there's *some* real acoustic difference between
  candidates (distance, orientation, mic gain) — if everyone is
  equidistant with identical recorders, the signal has a real ceiling.
- Confidence scores reflect genuine uncertainty in the acoustic data. A
  "low confidence" result is the pipeline being honest that it can't tell
  — not a bug to tune away.
- Partnerships are re-discovered per time window (`--partner-window-s`),
  not per turn — a real group change that happens to land mid-window
  will show up gradually rather than instantly at the exact moment it
  occurred. Shorter windows track change faster but with less data (and
  therefore lower confidence) per window; there's a real tradeoff here,
  not a free lunch.
- Requires one recorder per person; a shared room microphone is a
  different problem (blind diarization + voice enrollment) not covered
  by this pipeline.

See the comment block at the bottom of `audio_interaction_pipeline.py`
("EXTENDING THIS") for concrete ideas on pushing accuracy further while
staying fully acoustic (better VAD, cross-channel echo cancellation,
sound-source localization).
