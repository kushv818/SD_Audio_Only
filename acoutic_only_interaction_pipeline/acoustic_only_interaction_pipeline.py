#!/usr/bin/env python3
"""
audio_interaction_pipeline.py

Estimate who-talks-to-whom, over time, from N chest-worn ("close-talk")
recordings of a group conversation - including cases with SIMULTANEOUS
parallel conversations (e.g. two side conversations happening at once).

This version is PURELY ACOUSTIC: everything here is derived from the raw
waveforms alone (loudness, timing, cross-channel correlation) - no speech
transcription and no language model is used anywhere in the pipeline.

--------------------------------------------------------------------------
THE IDEA
--------------------------------------------------------------------------
Each person wears their own recorder on the chest. Every recorder picks up
EVERYONE in the room, but the wearer is always the loudest voice on their
own channel (closest mic = highest amplitude / SNR), while other speakers
bleed through much more quietly. That physical fact is what we exploit:

  1. ALIGN         - the N recordings almost certainly didn't start at the
                      exact same millisecond (recorders are started by
                      hand). GCC-PHAT cross-correlation finds the offset
                      between channels and shifts them onto one timeline.

  2. DETECT SPEECH  - per channel, INDEPENDENTLY (not "who's the single
                      loudest across everyone" - that assumption breaks
                      the moment two people talk at once in separate side
                      conversations). A channel counts as its wearer
                      speaking when its energy is (a) enough above its own
                      noise floor, AND (b) within a margin of the loudest
                      normalized SNR across all channels in that instant -
                      which correctly keeps multiple simultaneous genuine
                      speakers while rejecting mere bleed-through, without
                      needing a single brittle absolute volume cutoff.

  3. SEGMENT        - merge into per-person speech turns; a short "hangover"
                      bridges natural syllable-level volume dips so one
                      utterance doesn't fragment into unusable slivers.

  4. DISCOVER STABLE PARTNERSHIPS - using the WHOLE recording (not one
                      noisy turn), estimate how strongly every PAIR of
                      people acts like conversational partners, combining
                      three purely acoustic signals:
                        - proximity: while A talks, how much of A's voice
                          bleeds into B's channel - counted ONLY in frames
                          where B is independently silent, so B's own
                          voice never gets mistaken for bleed from A.
                        - envelope correlation: a simple, cheap complement -
                          whole-recording Pearson correlation between A's
                          and B's raw loudness-over-time curves. Real
                          partners' channels tend to share fine-grained
                          bleed-through co-modulation (the same underlying
                          voices at different attenuation) that unrelated
                          bystanders don't, and this needs no VAD, no
                          floors, no thresholds - just np.corrcoef.
                        - turn-taking: how often A's and B's turns
                          immediately follow each other, computed using
                          ONLY that pair's own two segment streams (doing
                          this on a globally-merged timeline across all N
                          channels is a trap: with parallel conversations,
                          a person's true partner's reply is often "hidden"
                          behind the other pair's interleaved turns,
                          making the wrong pair look more tightly coupled).
                      A max-weight matching over this affinity then finds
                      the best whole-recording pairing per person, along
                      with a CONFIDENCE score - a perfect matching always
                      assigns everyone a partner, even when the evidence is
                      weak, so the confidence score tells you whether a
                      pairing is a clear, validated match or just the best
                      of a weak field (see discover_partnerships).

  5. RESOLVE EACH TURN - for every individual turn, blend local proximity +
                      local context with a boost toward that speaker's
                      discovered stable partner, SCALED BY that partner's
                      confidence score - a clear, well-separated pairing
                      gets a strong, stabilizing boost; a low-confidence
                      pairing gets a much weaker one, so the output stays
                      honest about genuine uncertainty rather than being
                      forced to look clean. Still allows group/"circle"
                      addressing (one person, several simultaneous
                      listeners) when the evidence supports it.

  6. OUTPUT          - a per-turn timeline ("00:12 Alicia -> Kush"), merged
                      conversation blocks ("00:00-00:30 Alicia <-> Kush"),
                      an aggregate seconds/percent matrix, confidence and
                      component-breakdown CSVs, and plots - including a
                      multi-track view showing each discovered
                      partnership's own back-and-forth over the whole
                      recording, which is the clearest way to see two
                      parallel conversations running at once.

A "competitive" mode (single loudest-wins speaker per frame, simpler and
appropriate when only one person ever talks at a time) is also available
via --vad-mode competitive.

If a channel correlates weakly with EVERY other channel (a warning the
pipeline prints when this happens), that usually points at a hardware or
positioning difference for that specific recorder rather than that person
lacking a real conversational partner - worth checking that recording
directly.

This is a lightweight, dependency-light heuristic (not full speech
diarization). See "EXTENDING THIS" at the bottom of this file for ideas on
pushing accuracy further while staying acoustic (webrtcvad, pyannote,
sound-source localization, multi-mic beamforming, etc).

--------------------------------------------------------------------------
FILE NAMING
--------------------------------------------------------------------------
Expects files named like:  LENA_UNIT_U5_ALICIA_TRIAL1.wav
The speaker name is auto-detected as the last purely-alphabetic underscore-
separated token that isn't a known non-name token (LENA, UNIT, TRIAL, DAY,
SESSION, ...). Override with --name-index if your naming differs.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    python audio_interaction_pipeline.py --input-dir ./recordings --output-dir ./results

    # if the output shows implausible amounts of simultaneous speech,
    # tighten the competitive margin (the main knob for independent mode):
    python audio_interaction_pipeline.py --input-dir ./recordings --competitive-margin-db 8

    # only one person ever talks at a time? use the simpler competitive mode:
    python audio_interaction_pipeline.py --input-dir ./recordings --vad-mode competitive

Requires: numpy, scipy, pandas, matplotlib, networkx  (all pip-installable,
no audio-specific library like librosa/ffmpeg needed - reads plain PCM .wav)
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample_poly

import matplotlib
matplotlib.use("Agg")  # safe for headless/script use
import matplotlib.pyplot as plt
import networkx as nx


# ==========================================================================
# Config defaults (all overridable via CLI)
# ==========================================================================
DEFAULT_TARGET_SR = 16000     # Hz, all channels resampled to this
DEFAULT_FRAME_MS = 30         # analysis frame length
DEFAULT_HOP_MS = 10           # hop between frames
DEFAULT_SILENCE_DB = -45.0    # frames quieter than this (on ALL channels) = silence
DEFAULT_MARGIN_DB = 2.0       # loudest channel must beat 2nd loudest by this much
DEFAULT_MIN_SPEECH_MS = 200   # segments shorter than this are discarded as noise
DEFAULT_TURN_GAP_S = 2.0      # max silence gap between turns to count as a response
DEFAULT_ALIGN_WINDOW_S = 60.0 # seconds of audio used to estimate alignment lag
DEFAULT_MAX_SHIFT_S = 10.0    # search only +/- this many seconds for lag
DEFAULT_CONTEXT_WEIGHT = 1.5  # extra weight given to the prev/next speaker in a turn
DEFAULT_NOISE_PERCENTILE = 50 # percentile of each channel's silent-frame power used as its noise floor
DEFAULT_DOMINANT_SHARE = 0.55 # a listener needs >= this share of a turn to be its sole addressee
DEFAULT_GROUP_SHARE = 0.20    # otherwise, every listener with >= this share is an addressee
DEFAULT_BLOCK_OVERLAP_RATIO = 0.6  # symmetric overlap needed to merge consecutive turns into one block
DEFAULT_OWN_MARGIN_DB = 8.0   # a channel counts as "its own wearer speaking" this many dB above its floor
DEFAULT_COMPETITIVE_MARGIN_DB = 12.0  # must be within this many dB of the frame's loudest normalized SNR
DEFAULT_PARTNER_BOOST = 6.0   # multiplier applied to a speaker's discovered stable partner per turn
DEFAULT_MIN_CLEAN_FRAMES = 5  # min silent-on-listener frames needed to trust a bleed-through reading
DEFAULT_HANGOVER_MS = 150.0   # bridge silent gaps shorter than this within one channel's speech
DEFAULT_ENVELOPE_WEIGHT = 1.5  # weight of the raw energy-envelope correlation signal
DEFAULT_ENVELOPE_SMOOTH_MS = 0.0  # smoothing for envelope correlation - 0 (none) tested best
DEFAULT_AFFINITY_TURN_WEIGHT = 0.5  # weight of pairwise turn-taking relative to proximity (1.0)
DEFAULT_VOICEPRINT_SECONDS = 10.0  # seconds of a speaker's own audio used to build their reference voiceprint
DEFAULT_MAX_VOICE_SEGMENTS = 60    # cap on segments embedded per speaker (runtime control)
DEFAULT_MIN_VOICE_CLIP_S = 0.4     # shorter clips are too brief for a reliable voice embedding
DEFAULT_VOICE_AFFINITY_WEIGHT = 2.0    # weight in whole-recording partnership discovery
DEFAULT_VOICE_SIMILARITY_WEIGHT = 3.0  # per-turn boost weight
DEFAULT_PARTNER_WINDOW_S = 180.0  # re-discover partnerships every N seconds - who's paired with whom
                                   # can shift over a long recording (e.g. a classroom session);
                                   # a value >= the recording length collapses to one whole-recording
                                   # window, matching the previous fixed-pairing behavior

NON_NAME_TOKENS = {"LENA", "UNIT", "TRIAL", "DAY", "SESSION", "REC", "RECORDING"}


# ==========================================================================
# Data structures
# ==========================================================================
@dataclass
class Channel:
    name: str
    path: str
    sr: int = 0
    audio: np.ndarray = field(default=None, repr=False)  # float32, mono, [-1, 1]


@dataclass
class Segment:
    speaker: str
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class InteractionEvent:
    """One speech turn, resolved to who it was (likely) addressed to."""
    start_s: float
    end_s: float
    speaker: str
    addressees: List[str]           # one or more listeners, ranked by likelihood
    mode: str                       # "single" | "group" | "all"
    proportions: dict               # listener -> estimated share of this turn (0-1)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def label(self) -> str:
        if self.mode == "single":
            return f"{self.speaker} -> {self.addressees[0]}"
        tag = "ALL" if self.mode == "all" else "group"
        return f"{self.speaker} -> {', '.join(self.addressees)} ({tag})"


# ==========================================================================
# 1. Loading + naming
# ==========================================================================
def parse_speaker_name(filepath: str, name_index: Optional[int] = None) -> str:
    """Pull the speaker's name out of a filename like LENA_UNIT_U5_ALICIA_TRIAL1.wav"""
    stem = Path(filepath).stem
    parts = stem.split("_")

    if name_index is not None:
        if 0 <= name_index < len(parts):
            return parts[name_index]
        raise ValueError(f"--name-index {name_index} out of range for '{stem}'")

    # heuristic: alphabetic-only tokens that aren't known boilerplate.
    candidates = [p for p in parts if p.isalpha() and p.upper() not in NON_NAME_TOKENS]
    if candidates:
        # names typically appear right before the TRIAL/DAY token, so take the last one
        return candidates[-1].title()

    # fallback: just use the whole filename stem
    return stem


def load_wav_mono_float(path: str) -> Tuple[int, np.ndarray]:
    """Read a PCM wav file -> (sample_rate, float32 mono samples in [-1, 1])."""
    sr, data = wavfile.read(path)

    if data.ndim > 1:  # stereo/multi-channel -> average to mono
        data = data.mean(axis=1)

    if np.issubdtype(data.dtype, np.integer):
        max_val = float(np.iinfo(data.dtype).max)
        data = data.astype(np.float32) / max_val
    else:
        data = data.astype(np.float32)

    return sr, data


def resample_to(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return y
    frac = Fraction(target_sr, orig_sr).limit_denominator(1000)
    return resample_poly(y, frac.numerator, frac.denominator).astype(np.float32)


def load_channels(input_dir: str, pattern: str, target_sr: int,
                   name_index: Optional[int]) -> List[Channel]:
    paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not paths:
        raise FileNotFoundError(
            f"No files matched '{pattern}' in '{input_dir}'. "
            f"Check --input-dir / --pattern."
        )

    channels = []
    for p in paths:
        name = parse_speaker_name(p, name_index)
        sr, y = load_wav_mono_float(p)
        y = resample_to(y, sr, target_sr)
        channels.append(Channel(name=name, path=p, sr=target_sr, audio=y))
        print(f"Loaded {os.path.basename(p):45s} -> speaker='{name}', "
              f"{len(y) / target_sr:8.1f}s @ {target_sr} Hz")

    names = [c.name for c in channels]
    if len(set(names)) != len(names):
        print(f"WARNING: duplicate speaker names parsed {names}. "
              f"Use --name-index to disambiguate.")
    return channels


# ==========================================================================
# 2. Alignment (GCC-PHAT cross-correlation)
# ==========================================================================
def gcc_phat_lag(sig: np.ndarray, ref: np.ndarray, sr: int, max_shift_s: float) -> int:
    """
    Estimate the lag (in samples) that best aligns `sig` to `ref` using the
    phase transform (PHAT) weighted cross-correlation - robust to differing
    overall loudness between the two recordings.
    Positive return value means `sig` lags behind `ref` (sig started later).
    """
    n = len(sig) + len(ref)
    n_fft = 1 << (n - 1).bit_length()  # next power of 2

    SIG = np.fft.rfft(sig, n=n_fft)
    REF = np.fft.rfft(ref, n=n_fft)
    cross = SIG * np.conj(REF)
    denom = np.abs(cross)
    denom[denom < 1e-12] = 1e-12
    cross_phat = cross / denom

    cc = np.fft.irfft(cross_phat, n=n_fft)
    max_shift = min(int(max_shift_s * sr), n_fft // 2 - 1)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

    shift = np.argmax(np.abs(cc)) - max_shift
    return int(shift)


def align_channels(channels: List[Channel], align_window_s: float,
                    max_shift_s: float) -> List[Channel]:
    """
    Align all channels to channel[0] using a short window of audio (fast),
    then apply the resulting integer-sample shift to the FULL-length signal
    and trim everyone down to the shortest common length.
    """
    ref = channels[0]
    sr = ref.sr
    win = int(align_window_s * sr)
    ref_window = ref.audio[:win]

    shifts = [0]
    for ch in channels[1:]:
        sig_window = ch.audio[:win]
        lag = gcc_phat_lag(sig_window, ref_window, sr, max_shift_s)
        shifts.append(lag)
        print(f"  Estimated lag for '{ch.name}' relative to '{ref.name}': "
              f"{lag} samples ({lag / sr * 1000:.1f} ms)")

    aligned = []
    for ch, lag in zip(channels, shifts):
        if lag > 0:
            # ch started `lag` samples after ref -> drop lag samples from ch's start
            y = ch.audio[lag:]
        elif lag < 0:
            # ch started before ref -> pad ch's start with silence
            y = np.concatenate([np.zeros(-lag, dtype=np.float32), ch.audio])
        else:
            y = ch.audio
        aligned.append(Channel(name=ch.name, path=ch.path, sr=ch.sr, audio=y))

    min_len = min(len(c.audio) for c in aligned)
    for c in aligned:
        c.audio = c.audio[:min_len]

    print(f"  Aligned length: {min_len / sr:.1f}s across all {len(aligned)} channels")
    return aligned


# ==========================================================================
# 3. Frame energy + speech detection
# ==========================================================================
def frame_energy_db(y: np.ndarray, sr: int, frame_ms: int, hop_ms: int) -> np.ndarray:
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    n_frames = max(0, 1 + (len(y) - frame_len) // hop_len)
    if n_frames <= 0:
        return np.array([])

    # sliding-window view -> shape (n_frames, frame_len), no copy
    shape = (n_frames, frame_len)
    strides = (y.strides[0] * hop_len, y.strides[0])
    frames = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)

    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-10)


def build_energy_matrix(channels: List[Channel], frame_ms: int,
                         hop_ms: int) -> Tuple[np.ndarray, List[str]]:
    energies = [frame_energy_db(c.audio, c.sr, frame_ms, hop_ms) for c in channels]
    n_frames = min(len(e) for e in energies)
    energies = [e[:n_frames] for e in energies]
    return np.vstack(energies), [c.name for c in channels]  # shape (n_speakers, n_frames)


def dominant_speaker_per_frame(energy_db: np.ndarray, names: List[str],
                                silence_db: float, margin_db: float) -> List[Optional[str]]:
    """
    COMPETITIVE mode: for each frame, pick the loudest channel. If the
    loudest channel is below the silence floor -> nobody is speaking. If it
    doesn't clearly beat the 2nd loudest (within margin_db) -> ambiguous,
    treated as silence rather than guessed. Simple and effective when only
    one person ever talks at a time; see per_channel_speaking_masks for the
    default mode that also handles simultaneous/parallel conversations.
    """
    n_speakers, n_frames = energy_db.shape
    labels: List[Optional[str]] = []
    sorted_vals = np.sort(energy_db, axis=0)[::-1]  # descending per frame
    top_idx = np.argmax(energy_db, axis=0)

    for i in range(n_frames):
        top_val = sorted_vals[0, i]
        second_val = sorted_vals[1, i] if n_speakers > 1 else -np.inf
        if top_val < silence_db:
            labels.append(None)
        elif (top_val - second_val) < margin_db:
            labels.append(None)  # too close to call -> don't guess
        else:
            labels.append(names[top_idx[i]])
    return labels


def per_channel_speaking_masks(
    energy_db: np.ndarray, names: List[str], own_margin_db: float, noise_percentile: float,
    competitive_margin_db: float = DEFAULT_COMPETITIVE_MARGIN_DB,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    INDEPENDENT per-channel voice-activity detection (the default mode).

    dominant_speaker_per_frame() is "competitive": it picks ONE winner
    across all channels per frame. That's the right model for a single
    group conversation where only one person talks at a time - but it
    breaks down if there are two simultaneous side conversations happening
    at once (e.g. Alicia<->Kush chatting while, at the same moment,
    Riad<->Max are also chatting). In that case there can genuinely be 2
    people speaking simultaneously, and forcing a single "loudest wins"
    choice per frame just arbitrarily alternates between fragments of both
    conversations.

    A plain absolute dB threshold ("is this channel >= own_margin_db above
    its own floor?") turned out to be fragile in testing: a single fixed
    cutoff has to simultaneously (a) be low enough to catch genuine speech
    through natural volume dips, and (b) be high enough to reject bleed-
    through - and when the true own-voice-vs-bleed SNR gap is only modest
    (common once several people are genuinely close together), no single
    fixed number does both well, and results swing wildly with small
    threshold changes.

    This version instead combines two checks, and is self-calibrating
    rather than relying on one brittle constant:

      1. ABSOLUTE gate (own_margin_db): still requires SOME minimum SNR
         above the channel's own floor, to reject pure silence/noise.
      2. RELATIVE/competitive check (competitive_margin_db): a channel only
         counts as "speaking" if its own normalized SNR is within
         competitive_margin_db of the LOUDEST normalized SNR across all
         channels in that same frame. A true speaker's own channel is
         always near-maximal for its own voice, so this passes for anyone
         genuinely talking - including multiple people at once, since two
         unrelated simultaneous speakers are each near-maximal on their own
         channel independent of each other. Pure bleed-through, by
         contrast, is reliably far below the true speaker's SNR in that
         same frame, so it gets rejected regardless of the absolute dB
         scale of the recording.

    Returns:
      masks       : list of boolean arrays (one per channel), True = that
                    person's own voice is active in that frame.
      noise_floor : per-channel ambient noise floor (linear power), each
                    estimated from that channel's own non-speaking frames.
    """
    power = db_to_power(energy_db)
    n_speakers, n_frames = power.shape

    # first pass: rough per-channel floor from the low percentile of the whole channel
    rough_floor = np.array([np.percentile(power[s, :], min(noise_percentile, 20))
                             for s in range(n_speakers)])
    rough_floor_db = 10 * np.log10(rough_floor + 1e-12)
    rough_snr = energy_db - rough_floor_db[:, None]
    rough_speaking = rough_snr >= own_margin_db

    # second pass: refine each channel's floor using only ITS OWN quiet frames
    noise_floor = np.zeros(n_speakers)
    for s in range(n_speakers):
        quiet = ~rough_speaking[s]
        if quiet.sum() >= 20:
            noise_floor[s] = np.percentile(power[s, quiet], noise_percentile)
        else:
            noise_floor[s] = rough_floor[s]
    floor_db = 10 * np.log10(noise_floor + 1e-12)

    # final decision: absolute gate AND competitive (relative-to-frame-max) check
    snr = energy_db - floor_db[:, None]              # (n_speakers, n_frames)
    max_snr_per_frame = snr.max(axis=0)               # (n_frames,)
    abs_pass = snr >= own_margin_db
    competitive_pass = snr >= (max_snr_per_frame[None, :] - competitive_margin_db)
    speaking = abs_pass & competitive_pass

    masks = [speaking[s] for s in range(n_speakers)]
    return masks, noise_floor


def bridge_short_gaps(mask: np.ndarray, hop_ms: int, hangover_ms: float) -> np.ndarray:
    """
    'Hangover' smoothing: real speech has natural syllable-to-syllable energy
    dips (breaths, stop consonants, etc.), which can otherwise make a frame-
    level VAD flicker on/off many times within one continuous utterance,
    fragmenting it into pieces too short to survive min_speech_ms. This
    bridges any silent gap shorter than hangover_ms that's sandwiched
    between two speaking regions, back into one continuous turn.
    """
    hop_s = hop_ms / 1000.0
    max_gap_frames = int(round(hangover_ms / 1000.0 / hop_s))
    if max_gap_frames <= 0:
        return mask

    result = mask.copy()
    n = len(mask)
    i = 0
    while i < n:
        if not result[i]:
            j = i
            while j < n and not result[j]:
                j += 1
            gap_len = j - i
            if gap_len <= max_gap_frames and i > 0 and j < n:
                result[i:j] = True
            i = j
        else:
            i += 1
    return result


def masks_to_channel_segments(
    masks: List[np.ndarray], names: List[str], hop_ms: int, min_speech_ms: int,
    hangover_ms: float = 0.0,
) -> dict:
    """Convert each channel's independent boolean speaking mask into its own list of Segments."""
    channel_segments = {}
    for name, mask in zip(names, masks):
        if hangover_ms > 0:
            mask = bridge_short_gaps(mask, hop_ms, hangover_ms)
        labels = [name if m else None for m in mask]
        segs = filter_short_segments(frames_to_segments(labels, hop_ms), min_speech_ms)
        channel_segments[name] = segs
    return channel_segments


# ==========================================================================
# 4. Frames -> speech segments
# ==========================================================================
def frames_to_segments(labels: List[Optional[str]], hop_ms: int) -> List[Segment]:
    segments = []
    hop_s = hop_ms / 1000.0
    cur_speaker = None
    cur_start = 0

    for i, lab in enumerate(labels + [None]):  # sentinel to flush last segment
        if lab != cur_speaker:
            if cur_speaker is not None:
                segments.append(Segment(cur_speaker, cur_start * hop_s, i * hop_s))
            cur_speaker = lab
            cur_start = i
    return segments


def filter_short_segments(segments: List[Segment], min_ms: int) -> List[Segment]:
    min_s = min_ms / 1000.0
    return [s for s in segments if s.duration_s >= min_s]


# ==========================================================================
# 5. Whole-recording partnership discovery (proximity + envelope + turn-taking)
# ==========================================================================
def build_interaction_matrix(segments: List[Segment], names: List[str],
                              turn_gap_s: float) -> pd.DataFrame:
    matrix = pd.DataFrame(0, index=names, columns=names, dtype=int)
    for prev, curr in zip(segments, segments[1:]):
        gap = curr.start_s - prev.end_s
        if prev.speaker != curr.speaker and gap <= turn_gap_s:
            matrix.loc[prev.speaker, curr.speaker] += 1
    return matrix


def db_to_power(db: np.ndarray) -> np.ndarray:
    """Convert dB (20*log10(rms)) back to power (rms^2), for summing/averaging."""
    return 10.0 ** (db / 10.0)


def compute_envelope_correlation(
    energy_db: np.ndarray, names: List[str], smooth_ms: float = DEFAULT_ENVELOPE_SMOOTH_MS,
    hop_ms: float = 10.0,
) -> pd.DataFrame:
    """
    Whole-recording Pearson correlation between every pair of channels' raw
    loudness-over-time curves. This is a simple, cheap, model-free
    complement to the excess-energy proximity calculation: two mics that
    are both picking up (via bleed-through) the SAME nearby exchange tend
    to track each other's fine-grained syllable-rate amplitude modulation -
    the same underlying voices, just attenuated differently on each
    channel - producing real positive correlation over the whole
    recording. Two mics on an unrelated, simultaneous side conversation
    don't share that fine structure and correlate far less (often near
    zero or negative).

    Empirically, the RAW, unsmoothed energy curve discriminates real
    partners from bystanders better than any smoothed version tested -
    smoothing blurs away exactly the fine-grained co-modulation that makes
    this signal work, in favor of coarser patterns that are less
    informative and can even flip sign. smooth_ms is offered as a knob but
    defaults to 0 (no smoothing).
    """
    if smooth_ms and smooth_ms > 0:
        from scipy.ndimage import uniform_filter1d
        win = max(1, int(round(smooth_ms / hop_ms)))
        data = uniform_filter1d(energy_db, size=win, axis=1, mode="nearest") if win > 1 else energy_db
    else:
        data = energy_db
    corr = np.corrcoef(data)
    return pd.DataFrame(corr, index=names, columns=names)


def load_voice_encoder():
    """
    Lazily imports resemblyzer (optional dependency) and loads its pretrained
    speaker-embedding model. Fully local/offline after the one-time weight
    download - no API key, no gated model access. Still purely acoustic:
    the embedding captures vocal characteristics (pitch, timbre, resonance),
    not words - this is speaker VERIFICATION, not speech recognition.
    """
    try:
        from resemblyzer import VoiceEncoder
    except ImportError as e:
        raise RuntimeError(
            "--use-voice-embeddings requires the 'resemblyzer' package. "
            "Install with: pip install resemblyzer"
        ) from e
    return VoiceEncoder()


def build_speaker_voiceprints(
    channels: List["Channel"], channel_segments: dict, encoder,
    min_total_s: float = DEFAULT_VOICEPRINT_SECONDS,
) -> dict:
    """
    Build one reference voice embedding per speaker, from their OWN channel
    during their OWN detected segments (longest first, until min_total_s of
    audio is collected). This is the "known voiceprint" that bleed-through
    on other channels will later be compared against.
    """
    voiceprints = {}
    ch_by_name = {c.name: c for c in channels}
    for name, segs in channel_segments.items():
        ch = ch_by_name.get(name)
        if ch is None:
            continue
        segs_sorted = sorted(segs, key=lambda s: s.duration_s, reverse=True)
        chunks, total = [], 0.0
        for seg in segs_sorted:
            start = int(seg.start_s * ch.sr)
            end = int(seg.end_s * ch.sr)
            chunks.append(ch.audio[start:end])
            total += seg.duration_s
            if total >= min_total_s:
                break
        if not chunks:
            continue
        sample_audio = np.concatenate(chunks)
        try:
            voiceprints[name] = encoder.embed_utterance(sample_audio)
        except Exception as e:
            print(f"    WARNING: couldn't build a voiceprint for '{name}' ({e}); "
                  f"skipping voice-similarity for this person")
    return voiceprints


def compute_voice_similarity(
    channels: List["Channel"], channel_segments: dict, voiceprints: dict, encoder,
    max_segments_per_speaker: int = DEFAULT_MAX_VOICE_SEGMENTS,
    min_clip_s: float = DEFAULT_MIN_VOICE_CLIP_S,
) -> Tuple[pd.DataFrame, dict]:
    """
    For each of a speaker's segments (longest first, up to a cap for
    runtime), embed the BLEED-THROUGH audio on every OTHER channel during
    that same segment and compare it (cosine similarity) to that speaker's
    reference voiceprint. Unlike raw energy, this is largely loudness-
    invariant and checks WHOSE voice is actually present - so it can
    confirm genuine bleed-through and reject same-time noise or unrelated
    sources that just happen to be loud but don't actually sound like the
    speaker. This is what a "diarization-style" approach adds on top of
    pure energy-based proximity - still fully acoustic throughout.

    Returns:
      affinity_matrix : whole-recording (speaker x listener) average
                         similarity, symmetrized - for partnership discovery.
      per_segment      : {(speaker_start_s, listener_name): similarity} -
                         for per-turn resolution (see estimate_interactions).
    """
    ch_by_name = {c.name: c for c in channels}
    names = list(channel_segments.keys())
    sim_sum = pd.DataFrame(0.0, index=names, columns=names)
    sim_count = pd.DataFrame(0, index=names, columns=names, dtype=int)
    per_segment = {}

    for speaker, segs in channel_segments.items():
        if speaker not in voiceprints:
            continue
        segs_sample = sorted(segs, key=lambda s: s.duration_s, reverse=True)[:max_segments_per_speaker]
        for seg in segs_sample:
            if seg.duration_s < min_clip_s:
                continue
            for listener in names:
                if listener == speaker or listener not in voiceprints:
                    continue
                ch = ch_by_name[listener]
                start = int(seg.start_s * ch.sr)
                end = int(seg.end_s * ch.sr)
                clip = ch.audio[start:end]
                if len(clip) < int(min_clip_s * ch.sr):
                    continue
                try:
                    emb = encoder.embed_utterance(clip)
                except Exception:
                    continue
                denom = float(np.linalg.norm(emb) * np.linalg.norm(voiceprints[speaker]) + 1e-9)
                sim = float(np.dot(emb, voiceprints[speaker]) / denom)
                sim_sum.loc[speaker, listener] += sim
                sim_count.loc[speaker, listener] += 1
                per_segment[(round(seg.start_s, 3), listener)] = sim

    sim_avg = sim_sum / sim_count.replace(0, np.nan)
    sim_avg = sim_avg.fillna(0.0)
    affinity_matrix = (sim_avg + sim_avg.T) / 2
    return affinity_matrix, per_segment



def compute_pairwise_affinity(
    channel_segments: dict,
    energy_db: np.ndarray,
    names: List[str],
    hop_ms: int,
    noise_floor_power: np.ndarray,
    masks: List[np.ndarray],
    turn_gap_s: float,
    min_clean_frames: int = DEFAULT_MIN_CLEAN_FRAMES,
    turn_taking_weight: float = DEFAULT_AFFINITY_TURN_WEIGHT,
    envelope_correlation: Optional[pd.DataFrame] = None,
    envelope_weight: float = DEFAULT_ENVELOPE_WEIGHT,
    voice_affinity: Optional[pd.DataFrame] = None,
    voice_weight: float = DEFAULT_VOICE_AFFINITY_WEIGHT,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Estimate, using the WHOLE recording (not one noisy turn at a time), how
    strongly every pair of people acts like conversational partners, from
    three purely acoustic ingredients:

      - proximity: while A speaks, how much excess energy (above its own
        floor) shows up on B's channel - but ONLY counting frames where B
        is independently silent, so B's own voice never contaminates the
        reading of how much of A bled into B's mic. This is the primary,
        most statistically robust signal (averaged over the whole
        recording) and is weighted at full strength.
      - envelope correlation (optional): whole-recording Pearson
        correlation between A's and B's raw loudness-over-time curves -
        see compute_envelope_correlation. A simple, cheap complement to
        proximity: real partners' channels tend to share fine-grained
        bleed-through co-modulation that unrelated bystanders don't.
      - voice similarity (optional, --use-voice-embeddings): whole-
        recording average cosine similarity between A's voiceprint and
        the bleed-through audio on B's channel during A's turns - see
        compute_voice_similarity. Where proximity/envelope only measure
        HOW LOUD the bleed-through is, this checks WHOSE VOICE it
        actually is, which can confirm genuine bleed-through and reject
        same-time noise or unrelated sources that just happen to be loud.
      - turn-taking: how often A's turns and B's turns immediately follow
        one another, computed using ONLY that pair's own two segment
        streams. This helps, but even computed this way it can still pick
        up coincidental adjacency between two people who simply happen to
        pause/start near each other by chance (especially with two
        independent parallel conversations running at once) - so it's
        down-weighted (turn_taking_weight) rather than trusted equally.

    Returns (affinity, proximity_component, turn_taking_component), each an
    n x n DataFrame, symmetric, normalized to [0, 1] before combining.
    """
    hop_s = hop_ms / 1000.0
    name_to_idx = {n: i for i, n in enumerate(names)}
    power = db_to_power(energy_db)
    n_frames = power.shape[1]

    prox_sum = pd.DataFrame(0.0, index=names, columns=names)
    prox_weight = pd.DataFrame(0.0, index=names, columns=names)

    for speaker, segs in channel_segments.items():
        for seg in segs:
            f_start = max(0, int(round(seg.start_s / hop_s)))
            f_end = min(n_frames, max(f_start + 1, int(round(seg.end_s / hop_s))))
            n_seg_frames = f_end - f_start
            for listener in names:
                if listener == speaker:
                    continue
                l_idx = name_to_idx[listener]
                listener_quiet = ~masks[l_idx][f_start:f_end]
                if listener_quiet.sum() >= min_clean_frames:
                    clean_power = power[l_idx, f_start:f_end][listener_quiet]
                    weight = float(listener_quiet.sum()) * hop_s
                else:
                    # Listener was active for nearly all of this turn. Skipping
                    # the segment here ("too little clean signal to trust")
                    # would systematically discard exactly the segments where a
                    # real dialogue partner was backchanneling heavily -
                    # underweighting the correct partner relative to a
                    # bystander who happens to stay quiet (and thus never gets
                    # skipped) every time. Instead, fall back to the full
                    # segment, just weighted down since it's a noisier reading
                    # than a genuinely clean one.
                    clean_power = power[l_idx, f_start:f_end]
                    weight = float(n_seg_frames) * hop_s * 0.5
                excess = max(float(clean_power.mean()) - noise_floor_power[l_idx], 1e-12)
                prox_sum.loc[speaker, listener] += excess * weight
                prox_weight.loc[speaker, listener] += weight

    prox = prox_sum / prox_weight.replace(0, np.nan)
    prox = prox.fillna(0.0)
    prox_sym = (prox + prox.T) / 2  # average the two directions into one symmetric score

    # turn-taking: for EACH PAIR separately, take only that pair's own two
    # segment streams (ignoring everyone else) and count how often they
    # alternate. This must be done pair-by-pair rather than on one merged
    # timeline across all speakers: if two independent conversations are
    # running in parallel, a person's true partner's reply is often
    # "hidden" behind the OTHER conversation's interleaved turns in global
    # time, which would otherwise make turn-taking look strongest between
    # the wrong (unrelated) pair.
    turn_sym = pd.DataFrame(0, index=names, columns=names, dtype=int)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pair_segs = sorted(channel_segments.get(a, []) + channel_segments.get(b, []),
                                key=lambda s: s.start_s)
            count = 0
            for prev, curr in zip(pair_segs, pair_segs[1:]):
                gap = curr.start_s - prev.end_s
                if prev.speaker != curr.speaker and gap <= turn_gap_s:
                    count += 1
            turn_sym.loc[a, b] = count
            turn_sym.loc[b, a] = count

    def normalize01(df: pd.DataFrame) -> pd.DataFrame:
        vmax = df.values.max()
        return df / vmax if vmax > 0 else df

    prox_norm = normalize01(prox_sym)
    turn_norm = normalize01(turn_sym.astype(float))
    affinity = prox_norm + turn_taking_weight * turn_norm
    if envelope_correlation is not None:
        # only positive co-modulation counts as evidence of a shared exchange
        env_clipped = envelope_correlation.clip(lower=0.0)
        env_norm = normalize01(env_clipped)
        affinity = affinity + envelope_weight * env_norm
    if voice_affinity is not None:
        voice_clipped = voice_affinity.clip(lower=0.0)
        voice_norm = normalize01(voice_clipped)
        affinity = affinity + voice_weight * voice_norm
    affinity_arr = affinity.to_numpy(copy=True)
    np.fill_diagonal(affinity_arr, 0.0)
    affinity = pd.DataFrame(affinity_arr, index=names, columns=names)
    return affinity, prox_sym, turn_sym


def discover_partnerships(affinity: pd.DataFrame) -> Tuple[dict, List[Tuple[str, str, float]], dict]:
    """
    Find the best whole-recording pairing of speakers using max-weight
    matching on the affinity graph (networkx handles the odd-person-out
    case automatically by leaving them unmatched). This is what turns a
    noisy, turn-by-turn addressee guess into a stable, validated answer
    like "Alicia's partner is Kush for this whole recording".

    IMPORTANT: a perfect matching always assigns EVERYONE a partner, even
    if the evidence for a given pair is weak - it just picks the best
    AVAILABLE option, which isn't the same as a confident, validated
    answer. To surface that, this also returns a per-person confidence
    score: how much better the chosen partner's affinity is than that
    person's next-best alternative. A pair with high affinity but a close
    runner-up (or a person whose best score with anyone is low to begin
    with) gets a LOW confidence score, meaning "this is just what's left
    over" rather than "this is clearly who they talk to".

    Returns (partner_map, ranked_pairs, confidence) where partner_map[name]
    is that person's discovered partner (or None if unmatched), and
    confidence[name] is in [0, 1] (higher = more confidently a real pair,
    not just the best of a weak field).
    """
    G = nx.Graph()
    G.add_nodes_from(affinity.index)
    for i, a in enumerate(affinity.index):
        for b in affinity.columns[i + 1:]:
            w = affinity.loc[a, b]
            if w > 0:
                G.add_edge(a, b, weight=float(w))

    matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=False)

    partner_map = {name: None for name in affinity.index}
    pairs = []
    for a, b in matching:
        partner_map[a] = b
        partner_map[b] = a
        pairs.append((a, b, affinity.loc[a, b]))
    pairs.sort(key=lambda t: t[2], reverse=True)

    confidence = {}
    for name in affinity.index:
        partner = partner_map[name]
        if partner is None:
            confidence[name] = 0.0
            continue
        score = affinity.loc[name, partner]
        rivals = [affinity.loc[name, other] for other in affinity.index
                  if other not in (name, partner)]
        best_rival = max(rivals) if rivals else 0.0
        # margin, normalized so it's ~0 when the runner-up is nearly as good
        # (a toss-up) and ~1 when the chosen partner clearly dominates
        confidence[name] = score / (score + best_rival) if (score + best_rival) > 0 else 0.0

    return partner_map, pairs, confidence


def build_time_windowed_partnerships(
    channels: List["Channel"],
    channel_segments: dict,
    energy_db: np.ndarray,
    names: List[str],
    hop_ms: int,
    noise_floor_power: np.ndarray,
    masks: List[np.ndarray],
    turn_gap_s: float,
    window_s: float,
    turn_taking_weight: float,
    envelope_weight: float,
    envelope_smooth_ms: float,
    voice_affinity: Optional[pd.DataFrame] = None,
    voice_weight: float = DEFAULT_VOICE_AFFINITY_WEIGHT,
) -> List[dict]:
    """
    Discover partnerships independently within consecutive time windows,
    rather than one fixed pairing for the entire recording.

    A single whole-recording pairing assumes who-talks-to-whom stays fixed
    for the whole file - true enough for two adults having one
    conversation, but not for settings like a classroom, where children's
    partners/small groups routinely shift over the course of a session.
    Splitting into windows (default DEFAULT_PARTNER_WINDOW_S) and
    re-running discovery in each one lets the "stable partner" prior track
    those shifts instead of forcing one artificial pairing onto the whole
    recording.

    Proximity, envelope correlation, and turn-taking are all recomputed
    FRESH within each window (since who's physically near/interacting with
    whom is exactly what can change over time). Voice similarity, if
    provided, is reused as-is across every window: a person's vocal
    identity doesn't change through the recording, so there's no need to
    (expensively) re-embed audio per window for that signal.

    If window_s is <= 0 or >= the recording's duration, this collapses to
    a single window covering the whole recording - identical to the
    previous fixed, whole-recording-pairing behavior.

    Returns a list of window dicts (sorted by time), each with:
      start_s, end_s, affinity, partner_map, confidence, ranked_pairs
    """
    hop_s = hop_ms / 1000.0
    n_frames = energy_db.shape[1]
    total_duration_s = n_frames * hop_s

    if window_s <= 0 or window_s >= total_duration_s:
        edges = [0.0, total_duration_s]
    else:
        edges = list(np.arange(0.0, total_duration_s, window_s))
        if edges[-1] < total_duration_s:
            edges.append(total_duration_s)

    windows = []
    for start_s, end_s in zip(edges[:-1], edges[1:]):
        w_channel_segments = {
            name: [s for s in segs if start_s <= s.start_s < end_s]
            for name, segs in channel_segments.items()
        }

        w_envelope = None
        if envelope_weight > 0:
            f_start = int(round(start_s / hop_s))
            f_end = min(n_frames, max(f_start + 1, int(round(end_s / hop_s))))
            # a fresh, re-indexed (frame 0 = window start) slice - fine here since
            # envelope correlation only looks at relative alignment within this array
            w_envelope = compute_envelope_correlation(
                energy_db[:, f_start:f_end], names, smooth_ms=envelope_smooth_ms, hop_ms=hop_ms,
            )

        # IMPORTANT: pass the FULL (unsliced) energy_db/masks here, not a
        # windowed copy - segments carry ABSOLUTE timestamps, and
        # compute_pairwise_affinity converts those to frame indices via
        # seg.start_s / hop_s. Those indices only line up correctly against
        # the full, original array; a re-indexed (frame-0-at-window-start)
        # slice would silently misalign for every window after the first.
        w_affinity, _, _ = compute_pairwise_affinity(
            w_channel_segments, energy_db, names, hop_ms, noise_floor_power, masks,
            turn_gap_s, turn_taking_weight=turn_taking_weight,
            envelope_correlation=w_envelope, envelope_weight=envelope_weight,
            voice_affinity=voice_affinity, voice_weight=voice_weight,
        )
        partner_map, ranked_pairs, confidence = discover_partnerships(w_affinity)
        windows.append({
            "start_s": start_s, "end_s": end_s, "affinity": w_affinity,
            "partner_map": partner_map, "confidence": confidence, "ranked_pairs": ranked_pairs,
        })
    return windows


def find_window_for_time(t: float, windows: List[dict]) -> Optional[dict]:
    """Which window a given timestamp falls into (last window wins ties at the boundary)."""
    if not windows:
        return None
    for w in windows:
        if w["start_s"] <= t < w["end_s"]:
            return w
    return windows[-1]


def most_common_partner_map(windows: List[dict], names: List[str]) -> dict:
    """
    A single representative partner_map for visualization purposes only
    (e.g. plot_partnership_tracks needs one fixed grouping to lay out
    rows) - picks, for each person, whichever partner they were matched
    with in the most windows. The real, time-varying assignment used for
    actually resolving turns lives in the per-window data, not this.
    """
    from collections import Counter
    counters = {name: Counter() for name in names}
    for w in windows:
        for name, partner in w["partner_map"].items():
            if partner is not None:
                counters[name][partner] += 1
    return {name: (counters[name].most_common(1)[0][0] if counters[name] else None)
            for name in names}


def estimate_noise_floor_power(energy_db: np.ndarray, labels: List[Optional[str]],
                                percentile: float) -> np.ndarray:
    """
    Per-channel ambient noise floor, in power units, estimated from frames
    where NOBODY was attributed as speaking (labels[i] is None). Falls back
    to the low percentile of all frames if there aren't enough silent ones.
    """
    power = db_to_power(energy_db)  # shape (n_speakers, n_frames)
    silent_idx = [i for i, lab in enumerate(labels) if lab is None]

    n_speakers = power.shape[0]
    floor = np.zeros(n_speakers)
    for s in range(n_speakers):
        if len(silent_idx) >= 20:
            floor[s] = np.percentile(power[s, silent_idx], percentile)
        else:
            floor[s] = np.percentile(power[s, :], min(percentile, 10))
    return floor


# ==========================================================================
# 6. Per-turn addressee resolution
# ==========================================================================
def determine_addressees(
    combined: dict, dominant_share: float, group_share: float
) -> Tuple[List[str], str, dict]:
    """
    Turn a {listener: combined_score} dict into a resolved set of addressees.

      - If one listener clearly dominates (>= dominant_share of the total),
        this turn is treated as directed 1:1 -> mode "single".
      - Otherwise, every listener holding at least group_share of the total
        is included as a simultaneous addressee -> mode "group" (or "all"
        if that ends up being literally everyone). This is what lets the
        model represent "person 1 talking to 3 people in a circle" instead
        of forcing every turn into a pair.

    Returns (addressees_ranked, mode, proportions).
    """
    total = sum(combined.values()) or 1e-12
    proportions = {k: v / total for k, v in combined.items()}
    ranked = sorted(proportions.items(), key=lambda kv: kv[1], reverse=True)

    top_name, top_prop = ranked[0]
    if top_prop >= dominant_share:
        return [top_name], "single", proportions

    addressees = [name for name, p in ranked if p >= group_share] or [top_name]
    mode = "all" if len(addressees) == len(proportions) else "group"
    return addressees, mode, proportions


def estimate_interactions(
    segments: List[Segment],
    energy_db: np.ndarray,
    names: List[str],
    hop_ms: int,
    noise_floor_power: np.ndarray,
    turn_gap_s: float,
    context_weight: float,
    dominant_share: float,
    group_share: float,
    listener_masks: Optional[List[np.ndarray]] = None,
    windows: Optional[List[dict]] = None,
    partner_boost_scale: float = 0.0,
    voice_similarity: Optional[dict] = None,
    voice_similarity_weight: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[InteractionEvent]]:
    """
    Core "who talked to whom, when, for how long" estimator.

    For every speech segment (speaker A, duration D) this combines signals
    to figure out who A was talking to - possibly more than one person at
    once (e.g. addressing the whole group):

      - proximity: while A is speaking, whatever energy shows up on
        listener L's channel is bleed-through of A's voice, attenuated by
        distance/orientation - measured from L's genuinely-quiet frames
        when there are any. If L was independently active for basically
        all of A's turn, this falls back to the full-turn average rather
        than zeroing L's score out - zeroing would treat ordinary
        backchanneling ("mhm"/"yeah" while A is still talking) as
        "contamination" and actively punish the correct partner for
        behaving like a real dialogue partner.
      - context: a boost for whoever spoke immediately before or after A's
        turn (within turn_gap_s).
      - stable partner prior: `windows` (see build_time_windowed_partnerships)
        gives a time-varying partner_map/confidence - for the window
        containing this turn, if it says L is A's discovered partner for
        THAT period, L's score gets an extra boost, scaled by
        `partner_boost_scale` times that window's confidence for A (so
        low-confidence pairings get a smaller, more honest boost - see
        discover_partnerships). This is what keeps a single noisy turn
        from flipping the addressee away from an established pairing,
        WITHOUT forcing the same false confidence onto a pairing the
        evidence doesn't support, AND without assuming that pairing holds
        for the entire recording rather than just the current period.
      - voice similarity (optional, --use-voice-embeddings): if this
        specific turn's bleed-through on L's channel was checked against
        A's voiceprint (see compute_voice_similarity), a high match gets
        an extra boost - this is turn-level confirmation of WHOSE voice
        the bleed-through actually is, not just how loud it was.

    Returns:
      talk_seconds : DataFrame, talk_seconds.loc[A, B] = total seconds
                     estimated that A spent talking to B (soft/weighted
                     aggregate across all turns).
      talk_pct     : same, row-normalized to % of A's total talk time.
      events       : List[InteractionEvent], one per speech turn, in time
                     order - the actual "at time X, A is talking to B
                     (and maybe C)" timeline.
    """
    hop_s = hop_ms / 1000.0
    name_to_idx = {n: i for i, n in enumerate(names)}
    power = db_to_power(energy_db)  # (n_speakers, n_frames)
    n_frames = power.shape[1]

    talk_seconds = pd.DataFrame(0.0, index=names, columns=names)
    events: List[InteractionEvent] = []
    segments_sorted = sorted(segments, key=lambda s: s.start_s)

    for i, seg in enumerate(segments_sorted):
        listeners = [n for n in names if n != seg.speaker]
        if not listeners:
            continue

        f_start = max(0, int(round(seg.start_s / hop_s)))
        f_end = min(n_frames, max(f_start + 1, int(round(seg.end_s / hop_s))))

        # --- signal (a): proximity via bleed-through excess energy ---
        excess = {}
        for L in listeners:
            L_idx = name_to_idx[L]
            if listener_masks is not None:
                clean = ~listener_masks[L_idx][f_start:f_end]
                if clean.sum() > 0:
                    # prefer L's genuinely-quiet frames for an uncontaminated read
                    mean_power = float(power[L_idx, f_start:f_end][clean].mean())
                else:
                    # L was active for this ENTIRE turn - no clean frame exists;
                    # fall back to the full-turn average rather than zeroing
                    mean_power = float(power[L_idx, f_start:f_end].mean())
            else:
                mean_power = float(np.mean(power[L_idx, f_start:f_end]))
            excess[L] = max(mean_power - noise_floor_power[L_idx], 1e-12)

        # --- signal (b): conversational context (adjacent turns) ---
        prev_seg = segments_sorted[i - 1] if i > 0 else None
        if prev_seg is not None and not (
            seg.start_s - prev_seg.end_s <= turn_gap_s and prev_seg.speaker != seg.speaker
        ):
            prev_seg = None

        next_seg = segments_sorted[i + 1] if i < len(segments_sorted) - 1 else None
        if next_seg is not None and not (
            next_seg.start_s - seg.end_s <= turn_gap_s and next_seg.speaker != seg.speaker
        ):
            next_seg = None

        prev_speaker = prev_seg.speaker if prev_seg else None
        next_speaker = next_seg.speaker if next_seg else None

        current_window = find_window_for_time(seg.start_s, windows) if windows else None
        stable_partner = (current_window["partner_map"].get(seg.speaker)
                          if current_window else None)
        window_confidence = (current_window["confidence"].get(seg.speaker, 0.0)
                             if current_window else 0.0)

        combined = {}
        for L in listeners:
            boost = 1.0
            if L == prev_speaker:
                boost += context_weight
            if L == next_speaker:
                boost += context_weight
            if L == stable_partner:
                boost += partner_boost_scale * window_confidence
            if voice_similarity is not None:
                key = (round(seg.start_s, 3), L)
                if key in voice_similarity:
                    boost += voice_similarity_weight * max(voice_similarity[key], 0.0)
            combined[L] = excess[L] * boost

        addressees, mode, proportions = determine_addressees(
            combined, dominant_share, group_share
        )
        events.append(InteractionEvent(
            start_s=seg.start_s, end_s=seg.end_s, speaker=seg.speaker,
            addressees=addressees, mode=mode, proportions=proportions,
        ))

        for L in listeners:
            talk_seconds.loc[seg.speaker, L] += seg.duration_s * proportions[L]

    row_sums = talk_seconds.sum(axis=1).replace(0, np.nan)
    talk_pct = talk_seconds.div(row_sums, axis=0).fillna(0) * 100
    return talk_seconds.round(2), talk_pct.round(1), events


def fmt_time(seconds: float) -> str:
    """Seconds -> 'MM:SS', or 'H:MM:SS' once past an hour."""
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def merge_into_blocks(events: List[InteractionEvent], gap_s: float,
                       overlap_ratio: float = DEFAULT_BLOCK_OVERLAP_RATIO) -> List[dict]:
    """
    Merge consecutive turns into readable "conversation blocks" - e.g. a
    back-and-forth exchange between the same 2-3 people collapses into a
    single "00:00-00:30 Alicia <-> Bob" block instead of listing every
    individual turn.

    A turn merges into the current block only if the gap since the block's
    MOST RECENT turn is small, and the two turns' (speaker + addressees)
    sets overlap symmetrically by at least `overlap_ratio` in both
    directions. Comparing against just the last turn (not the whole
    block's accumulated membership) is important: otherwise, once a block
    has touched everyone, an unrelated later side-conversation between
    just 2 of those people would wrongly look like "still overlapping"
    and get absorbed into it.
    """
    blocks: List[dict] = []
    for ev in events:
        cluster = frozenset([ev.speaker] + ev.addressees)
        if blocks:
            last = blocks[-1]
            gap = ev.start_s - last["end_s"]
            shared = len(cluster & last["last_cluster"])
            ratio_a = shared / len(cluster)
            ratio_b = shared / len(last["last_cluster"])
            if gap <= gap_s and ratio_a >= overlap_ratio and ratio_b >= overlap_ratio:
                last["end_s"] = ev.end_s
                last["events"].append(ev)
                last["last_cluster"] = cluster
                continue
        blocks.append({"start_s": ev.start_s, "end_s": ev.end_s,
                        "events": [ev], "last_cluster": cluster})

    result = []
    for b in blocks:
        participants = set()
        for e in b["events"]:
            participants |= set([e.speaker] + e.addressees)
        participants = sorted(participants)
        label = (f"{participants[0]} <-> {participants[1]}" if len(participants) == 2
                 else ", ".join(participants) + " (group)")
        result.append({
            "start_s": b["start_s"], "end_s": b["end_s"],
            "duration_s": round(b["end_s"] - b["start_s"], 2),
            "participants": "|".join(participants),
            "label": label,
            "num_turns": len(b["events"]),
        })
    return result


def speaking_time_summary(segments: List[Segment], names: List[str]) -> pd.DataFrame:
    rows = []
    total_dur = sum(s.duration_s for s in segments) or 1.0
    for name in names:
        segs = [s for s in segments if s.speaker == name]
        total = sum(s.duration_s for s in segs)
        rows.append({
            "speaker": name,
            "num_turns": len(segs),
            "total_speaking_s": round(total, 2),
            "pct_of_speech": round(100 * total / total_dur, 1),
            "avg_turn_s": round(total / len(segs), 2) if segs else 0.0,
        })
    return pd.DataFrame(rows).sort_values("total_speaking_s", ascending=False)


# ==========================================================================
# 7. Plots
# ==========================================================================
def plot_heatmap(matrix: pd.DataFrame, out_path: str, title: str,
                  cbar_label: str, value_fmt: str = "{:d}"):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix.values, cmap="viridis")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Listener (B)")
    ax.set_ylabel("Speaker (A)")
    ax.set_title(title)

    vmax = matrix.values.max() or 1
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.values[i, j]
            ax.text(j, i, value_fmt.format(val), ha="center", va="center",
                    color="white" if val < vmax / 2 else "black")

    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_network(matrix: pd.DataFrame, speaking_time: pd.DataFrame, out_path: str,
                  title: str, edge_label_fmt: str = "{:d}"):
    G = nx.DiGraph()
    for name in matrix.index:
        G.add_node(name)
    for a in matrix.index:
        for b in matrix.columns:
            w = matrix.loc[a, b]
            if w > 0:
                G.add_edge(a, b, weight=float(w))

    fig, ax = plt.subplots(figsize=(6, 6))
    pos = nx.circular_layout(G)

    sizes_lookup = speaking_time.set_index("speaker")["total_speaking_s"].to_dict()
    node_sizes = [800 + 60 * sizes_lookup.get(n, 0) for n in G.nodes()]

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(weights) if weights else 1
    widths = [1 + 4 * (w / max_w) for w in weights]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="#6fa8dc", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(
        G, pos, width=widths, edge_color="#444444",
        arrowstyle="-|>", arrowsize=18, connectionstyle="arc3,rad=0.12", ax=ax,
    )
    edge_labels = {(u, v): edge_label_fmt.format(d["weight"]) for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_interaction_timeline(blocks: List[dict], out_path: str):
    """Gantt-style chart: one bar per conversation block, labeled with participants."""
    fig, ax = plt.subplots(figsize=(13, 2.5))
    cmap = plt.get_cmap("tab20")
    all_participant_sets = sorted({b["participants"] for b in blocks})
    color_lookup = {p: cmap(i % 20) for i, p in enumerate(all_participant_sets)}

    for b in blocks:
        ax.broken_barh(
            [(b["start_s"], b["duration_s"])], (0, 1),
            facecolors=color_lookup[b["participants"]], edgecolors="white",
        )
        mid = b["start_s"] + b["duration_s"] / 2
        if b["duration_s"] > 1.5:  # skip label if bar too thin to read
            ax.text(mid, 0.5, b["label"], ha="center", va="center",
                     fontsize=8, rotation=0, clip_on=True)

    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    ax.set_title("Interaction timeline (who's talking to whom, over time)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_partnership_tracks(channel_segments: dict, partner_map: dict,
                             names: List[str], out_path: str):
    """
    One horizontal row per person's MOST COMMON discovered partner across
    all time windows (see most_common_partner_map), each showing that
    pair's own speech turns over the full recording - a quick visual
    summary. Since partnerships can now shift over time (see
    partnership_windows.csv for the actual time-varying detail), this is a
    representative snapshot, not a guarantee that the pairing held for the
    entire recording.
    """
    # group into partnership rows: pairs first, then any unmatched singles
    seen = set()
    rows = []
    for name in names:
        if name in seen:
            continue
        partner = partner_map.get(name)
        if partner and partner not in seen:
            rows.append((name, partner))
            seen.update([name, partner])
        elif not partner:
            rows.append((name, None))
            seen.add(name)

    fig, ax = plt.subplots(figsize=(13, 0.8 * len(rows) + 1.5))
    cmap = plt.get_cmap("tab10", len(names))
    color_map = {name: cmap(i) for i, name in enumerate(names)}

    for row_i, (a, b) in enumerate(rows):
        for seg in channel_segments.get(a, []):
            ax.broken_barh([(seg.start_s, seg.duration_s)], (row_i - 0.4, 0.35),
                           facecolors=color_map[a])
        if b:
            for seg in channel_segments.get(b, []):
                ax.broken_barh([(seg.start_s, seg.duration_s)], (row_i + 0.05, 0.35),
                               facecolors=color_map[b])

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{a} <-> {b}" if b else f"{a} (no stable partner)" for a, b in rows])
    ax.set_xlabel("Time (s)")
    ax.set_title("Most common partnership per person - each row = one pair\n"
                 "(top/bottom strip = each person's own turns; see partnership_windows.csv "
                 "for how this changes over time)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_timeline(segments: List[Segment], names: List[str], out_path: str):
    fig, ax = plt.subplots(figsize=(12, 0.6 * len(names) + 1.5))
    colors = plt.get_cmap("tab10", len(names))
    color_map = {name: colors(i) for i, name in enumerate(names)}
    y_pos = {name: i for i, name in enumerate(names)}

    for seg in segments:
        ax.broken_barh(
            [(seg.start_s, seg.duration_s)],
            (y_pos[seg.speaker] - 0.4, 0.8),
            facecolors=color_map[seg.speaker],
        )

    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels(list(y_pos.keys()))
    ax.set_xlabel("Time (s)")
    ax.set_title("Speech timeline per speaker")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ==========================================================================
# Main pipeline
# ==========================================================================
def run_pipeline(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print("=== 1. Loading audio files ===")
    channels = load_channels(args.input_dir, args.pattern, args.sr, args.name_index)
    names = [c.name for c in channels]

    print("\n=== 2. Aligning channels to a common timeline ===")
    channels = align_channels(channels, args.align_window, args.max_shift)

    print(f"\n=== 3. Detecting who's speaking when (vad-mode={args.vad_mode}) ===")
    energy_db, names = build_energy_matrix(channels, args.frame_ms, args.hop_ms)

    windows = None
    listener_masks = None
    voice_similarity_lookup = None

    if args.vad_mode == "independent":
        print("  Using INDEPENDENT per-channel VAD - each person's own channel is "
              "checked against its own noise floor, so simultaneous/parallel\n"
              "  conversations (e.g. two side conversations at once) can be detected "
              "correctly instead of forcing one 'loudest wins' speaker per instant.")
        masks, noise_floor_power = per_channel_speaking_masks(
            energy_db, names, args.own_margin_db, args.noise_percentile,
            args.competitive_margin_db,
        )
        listener_masks = masks
        channel_segments = masks_to_channel_segments(masks, names, args.hop_ms, args.min_speech_ms,
                                                      args.hangover_ms)
        segments = sorted(
            (seg for segs in channel_segments.values() for seg in segs),
            key=lambda s: s.start_s,
        )
        print(f"  {len(segments)} speech turns retained across all channels "
              f"(min duration {args.min_speech_ms} ms)")

        voice_affinity = None
        if args.use_voice_embeddings:
            print(f"\n=== 3b. Building voice embeddings for cross-channel confirmation ===")
            encoder = load_voice_encoder()
            voiceprints = build_speaker_voiceprints(
                channels, channel_segments, encoder, min_total_s=args.voiceprint_seconds,
            )
            print(f"  Built voiceprints for {len(voiceprints)}/{len(names)} speakers "
                  f"(from up to {args.voiceprint_seconds:.0f}s of their own speech each)")
            voice_affinity, voice_similarity_lookup = compute_voice_similarity(
                channels, channel_segments, voiceprints, encoder,
                max_segments_per_speaker=args.max_voice_segments,
            )
            voice_affinity.to_csv(os.path.join(args.output_dir, "voice_similarity.csv"))
            print(f"  Computed voice-similarity for {len(voice_similarity_lookup)} "
                  f"(segment, listener) pairs")

        # global envelope correlation, purely for the diagnostic "weak
        # correlation with everyone" check below - partnership DISCOVERY
        # itself uses a fresh envelope computed within each time window
        envelope_corr = None
        if args.envelope_weight > 0:
            envelope_corr = compute_envelope_correlation(
                energy_db, names, smooth_ms=args.envelope_smooth_ms, hop_ms=args.hop_ms,
            )
            envelope_corr.to_csv(os.path.join(args.output_dir, "envelope_correlation.csv"))

        recording_duration_s = energy_db.shape[1] * (args.hop_ms / 1000.0)
        n_expected_windows = max(1, int(np.ceil(recording_duration_s / args.partner_window_s))
                                  if args.partner_window_s > 0 else 1)
        print(f"\n=== 4. Discovering partnerships in {n_expected_windows} time window(s) "
              f"of up to {args.partner_window_s:.0f}s each ===")
        windows = build_time_windowed_partnerships(
            channels, channel_segments, energy_db, names, args.hop_ms, noise_floor_power,
            masks, args.turn_gap, args.partner_window_s, args.affinity_turn_weight,
            args.envelope_weight, args.envelope_smooth_ms,
            voice_affinity=voice_affinity, voice_weight=args.voice_affinity_weight,
        )
        signal_desc = "bleed-through + envelope correlation"
        if args.use_voice_embeddings:
            signal_desc += " + voice similarity"
        signal_desc += " + turn-taking"
        for w in windows:
            if not w["ranked_pairs"]:
                continue
            print(f"  [{fmt_time(w['start_s'])}-{fmt_time(w['end_s'])}] "
                  f"partnerships ({signal_desc}):")
            for a, b, score in w["ranked_pairs"]:
                conf = w["confidence"][a]  # same for both a and b
                if conf >= 0.6:
                    tier = "high confidence"
                elif conf >= 0.35:
                    tier = "medium confidence"
                else:
                    tier = "LOW CONFIDENCE - best available match, not a clear pairing"
                print(f"      {a} <-> {b}   (affinity: {score:.2f}, "
                      f"confidence: {conf:.2f} - {tier})")
            unmatched = [n for n, p in w["partner_map"].items() if p is None]
            if unmatched:
                print(f"      No stable partner this window for: {', '.join(unmatched)}")

        # diagnostic: does any channel correlate weakly with EVERYONE (not just
        # its assigned partner), across the whole recording? That points at a
        # mic/recording issue for that person rather than a real absence of a partner.
        if envelope_corr is not None:
            for name in names:
                others = envelope_corr.loc[name].drop(name)
                if others.max() < 0.15:
                    print(f"  NOTE: '{name}' has weak envelope correlation with EVERY "
                          f"other channel (max {others.max():.3f}). This usually means "
                          f"a mic/positioning difference for that recorder (lower gain, "
                          f"farther placement, more noise) rather than that person "
                          f"lacking a real partner - worth checking that recording directly.")

        windows_df = pd.DataFrame([
            {"window_start_s": w["start_s"], "window_end_s": w["end_s"],
             "person_a": a, "person_b": b, "affinity_score": round(float(score), 3),
             "confidence": round(float(w["confidence"][a]), 3)}
            for w in windows for a, b, score in w["ranked_pairs"]
        ])
        windows_df.to_csv(os.path.join(args.output_dir, "partnership_windows.csv"), index=False)
    else:
        print("  Using COMPETITIVE per-frame VAD (single winner across all channels "
              "per frame) - best when only one person ever talks at a time.")
        labels = dominant_speaker_per_frame(energy_db, names, args.silence_db, args.margin_db)
        segments = filter_short_segments(frames_to_segments(labels, args.hop_ms), args.min_speech_ms)
        noise_floor_power = estimate_noise_floor_power(energy_db, labels, args.noise_percentile)
        print(f"  {len(segments)} speech segments retained "
              f"(min duration {args.min_speech_ms} ms)")

    print("\n=== 5. Building turn-taking matrix (conversational context signal) ===")
    turn_matrix = build_interaction_matrix(segments, names, args.turn_gap)
    speaking_time = speaking_time_summary(segments, names)

    print("\n=== 6. Noise floor per channel ===")
    for name, nf in zip(names, noise_floor_power):
        print(f"  {name}: noise floor power={nf:.2e} "
              f"({10*np.log10(nf+1e-12):.1f} dB)")

    print("\n=== 7. Estimating who talked to whom, when, and for how long "
          "(proximity + context" +
          (" + confidence-scaled stable-partner prior" if windows else "") + ") ===")
    active_windows = None if (args.no_partner_prior or not windows) else windows
    partner_boost_scale = 0.0 if active_windows is None else args.partner_boost

    talk_seconds, talk_pct, events = estimate_interactions(
        segments, energy_db, names, args.hop_ms, noise_floor_power,
        args.turn_gap, args.context_weight, args.dominant_share, args.group_share,
        listener_masks=listener_masks, windows=active_windows, partner_boost_scale=partner_boost_scale,
        voice_similarity=voice_similarity_lookup, voice_similarity_weight=args.voice_similarity_weight,
    )
    blocks = merge_into_blocks(events, args.turn_gap, args.block_overlap_ratio)

    print("\nSpeaking time summary:")
    print(speaking_time.to_string(index=False))

    print("\nTalk duration matrix in seconds (rows = speaker A, cols = listener B; "
          "soft-weighted aggregate across all turns):")
    print(talk_seconds.to_string())

    print(f"\nInteraction timeline ({len(blocks)} conversation blocks; "
          f"full per-turn detail in interaction_timeline.csv):")
    for b in blocks:
        print(f"  [{fmt_time(b['start_s'])}-{fmt_time(b['end_s'])}]  "
              f"{b['label']:35s} ({b['num_turns']} turn(s), {b['duration_s']:.1f}s)")

    print("\n=== 8. Saving outputs ===")
    seg_df = pd.DataFrame([(s.speaker, round(s.start_s, 3), round(s.end_s, 3),
                             round(s.duration_s, 3)) for s in segments],
                          columns=["speaker", "start_s", "end_s", "duration_s"])
    seg_df.to_csv(os.path.join(args.output_dir, "segments.csv"), index=False)
    speaking_time.to_csv(os.path.join(args.output_dir, "speaking_time.csv"), index=False)

    turn_matrix.to_csv(os.path.join(args.output_dir, "turn_taking_matrix.csv"))
    talk_seconds.to_csv(os.path.join(args.output_dir, "talk_duration_seconds.csv"))
    talk_pct.to_csv(os.path.join(args.output_dir, "talk_duration_percent.csv"))

    # per-turn timeline: THE main "who's talking to whom, when" deliverable
    timeline_rows = []
    for ev in events:
        row = {
            "start_s": round(ev.start_s, 3), "start_hms": fmt_time(ev.start_s),
            "end_s": round(ev.end_s, 3), "end_hms": fmt_time(ev.end_s),
            "duration_s": round(ev.duration_s, 3),
            "speaker": ev.speaker,
            "addressees": "|".join(ev.addressees),
            "mode": ev.mode,
            "description": ev.label(),
        }
        for n in names:
            row[f"share_{n}"] = round(ev.proportions.get(n, 0.0), 3)
        timeline_rows.append(row)
    timeline_df = pd.DataFrame(timeline_rows)
    timeline_df.to_csv(os.path.join(args.output_dir, "interaction_timeline.csv"), index=False)

    # merged conversation blocks: the coarser "00:00-00:30 A <-> B" view
    blocks_df = pd.DataFrame([{
        "start_s": round(b["start_s"], 3), "start_hms": fmt_time(b["start_s"]),
        "end_s": round(b["end_s"], 3), "end_hms": fmt_time(b["end_s"]),
        "duration_s": b["duration_s"], "participants": b["participants"].replace("|", ", "),
        "label": b["label"], "num_turns": b["num_turns"],
    } for b in blocks])
    blocks_df.to_csv(os.path.join(args.output_dir, "interaction_blocks.csv"), index=False)

    plot_heatmap(turn_matrix, os.path.join(args.output_dir, "turn_taking_heatmap.png"),
                 title="Turn-taking transition counts (A -> B)",
                 cbar_label="# of A -> B turn transitions", value_fmt="{:d}")
    plot_network(turn_matrix, speaking_time,
                 os.path.join(args.output_dir, "turn_taking_network.png"),
                 title="Turn-taking transitions\n(edge = # transitions, node size = speaking time)",
                 edge_label_fmt="{:.0f}")

    plot_heatmap(talk_seconds, os.path.join(args.output_dir, "talk_duration_heatmap.png"),
                 title="Estimated talk duration, A -> B (seconds)\n(proximity + context)",
                 cbar_label="seconds", value_fmt="{:.1f}")
    plot_network(talk_seconds, speaking_time,
                 os.path.join(args.output_dir, "talk_duration_network.png"),
                 title="Who talks to whom, and for how long\n"
                       "(edge = seconds, node size = total speaking time)",
                 edge_label_fmt="{:.1f}s")

    plot_timeline(segments, names, os.path.join(args.output_dir, "speech_timeline.png"))
    plot_interaction_timeline(blocks, os.path.join(args.output_dir, "interaction_timeline.png"))

    wrote_extra = ""
    if windows is not None:
        viz_partner_map = most_common_partner_map(windows, names)
        plot_partnership_tracks(channel_segments, viz_partner_map, names,
                                 os.path.join(args.output_dir, "partnership_tracks.png"))
        wrote_extra = "\n  Wrote partnership_windows.csv, partnership_tracks.png"
        if args.envelope_weight > 0:
            wrote_extra += "\n  Wrote envelope_correlation.csv"
        if args.use_voice_embeddings:
            wrote_extra += "\n  Wrote voice_similarity.csv"

    print(f"  Wrote segments.csv, speaking_time.csv")
    print(f"  Wrote turn_taking_matrix.csv, turn_taking_heatmap.png, turn_taking_network.png")
    print(f"  Wrote talk_duration_seconds.csv, talk_duration_percent.csv, "
          f"talk_duration_heatmap.png, talk_duration_network.png")
    print(f"  Wrote interaction_timeline.csv (per-turn), interaction_blocks.csv (merged), "
          f"interaction_timeline.png")
    print(f"  Wrote speech_timeline.png{wrote_extra}")
    print(f"\nAll outputs in: {os.path.abspath(args.output_dir)}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Estimate who-talks-to-whom from multiple chest-worn recordings "
                    "(purely acoustic - no transcription or language models).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir", required=True, help="Folder containing the .wav files")
    p.add_argument("--pattern", default="*.wav", help="Glob pattern to select files")
    p.add_argument("--output-dir", default="./interaction_results", help="Where to write outputs")
    p.add_argument("--name-index", type=int, default=None,
                   help="Underscore-split index to use as speaker name "
                        "(overrides auto-detection), e.g. 3 for "
                        "LENA_UNIT_U5_ALICIA_TRIAL1 -> 'ALICIA'")
    p.add_argument("--sr", type=int, default=DEFAULT_TARGET_SR, help="Target sample rate (Hz)")
    p.add_argument("--frame-ms", type=int, default=DEFAULT_FRAME_MS, help="Analysis frame length (ms)")
    p.add_argument("--hop-ms", type=int, default=DEFAULT_HOP_MS, help="Hop between frames (ms)")
    p.add_argument("--silence-db", type=float, default=DEFAULT_SILENCE_DB,
                   help="[competitive mode] frames quieter than this (dBFS) count as silence")
    p.add_argument("--margin-db", type=float, default=DEFAULT_MARGIN_DB,
                   help="[competitive mode] loudest channel must beat 2nd-loudest by this many dB")
    p.add_argument("--min-speech-ms", type=int, default=DEFAULT_MIN_SPEECH_MS,
                   help="Discard speech segments shorter than this (ms)")
    p.add_argument("--turn-gap", type=float, default=DEFAULT_TURN_GAP_S,
                   help="Max silence gap (s) between two turns to count as a response")
    p.add_argument("--align-window", type=float, default=DEFAULT_ALIGN_WINDOW_S,
                   help="Seconds of audio used to estimate cross-channel alignment lag")
    p.add_argument("--max-shift", type=float, default=DEFAULT_MAX_SHIFT_S,
                   help="Max expected misalignment between recorders (s)")
    p.add_argument("--context-weight", type=float, default=DEFAULT_CONTEXT_WEIGHT,
                   help="Extra weight given to whoever spoke immediately before/after "
                        "a turn (conversational-context signal) relative to proximity alone")
    p.add_argument("--noise-percentile", type=float, default=DEFAULT_NOISE_PERCENTILE,
                   help="Percentile of each channel's silent-frame power used as its "
                        "ambient noise floor when measuring bleed-through")
    p.add_argument("--dominant-share", type=float, default=DEFAULT_DOMINANT_SHARE,
                   help="If one listener holds at least this share of a turn's combined "
                        "score, the turn is resolved as directed 1:1 to them")
    p.add_argument("--group-share", type=float, default=DEFAULT_GROUP_SHARE,
                   help="When no listener dominates, every listener holding at least this "
                        "share is included as a simultaneous addressee (group/'circle' talk)")
    p.add_argument("--block-overlap-ratio", type=float, default=DEFAULT_BLOCK_OVERLAP_RATIO,
                   help="How much two consecutive turns' participant sets must overlap "
                        "(symmetric ratio) to be merged into the same conversation block")
    p.add_argument("--vad-mode", choices=["independent", "competitive"], default="independent",
                   help="'independent': each channel's own speech is detected against its own "
                        "noise floor (handles simultaneous/parallel side conversations). "
                        "'competitive': one 'loudest wins' speaker per frame across all "
                        "channels (simpler, but assumes only one person ever talks at a time)")
    p.add_argument("--own-margin-db", type=float, default=DEFAULT_OWN_MARGIN_DB,
                   help="[independent mode] absolute gate: minimum dB above a channel's own "
                        "floor to even be considered (mainly rejects pure silence/noise)")
    p.add_argument("--competitive-margin-db", type=float, default=DEFAULT_COMPETITIVE_MARGIN_DB,
                   help="[independent mode] MAIN TUNING KNOB: a channel counts as genuinely "
                        "speaking only if its own normalized SNR is within this many dB of the "
                        "frame's loudest normalized SNR. Lower = stricter (rejects more "
                        "bleed-through, but may also miss quieter simultaneous speakers); "
                        "raise if real simultaneous speakers are being dropped")
    p.add_argument("--hangover-ms", type=float, default=DEFAULT_HANGOVER_MS,
                   help="[independent mode] bridge silent gaps shorter than this within one "
                        "channel's speech, so natural syllable-level energy dips don't "
                        "fragment a continuous turn into unusable sub-min_speech_ms pieces")
    p.add_argument("--partner-boost", type=float, default=DEFAULT_PARTNER_BOOST,
                   help="[independent mode] multiplier boost applied to a speaker's discovered "
                        "whole-recording partner when resolving each individual turn (scaled "
                        "by that pairing's confidence - see discover_partnerships), so one "
                        "noisy turn can't flip away from an established stable pairing")
    p.add_argument("--no-partner-prior", action="store_true",
                   help="[independent mode] disable the stable-partner boost above and resolve "
                        "every turn from local evidence only")
    p.add_argument("--affinity-turn-weight", type=float, default=DEFAULT_AFFINITY_TURN_WEIGHT,
                   help="[independent mode] how much the pairwise turn-taking signal counts "
                        "relative to proximity (1.0) when discovering stable partnerships. "
                        "Lower this if partnerships look wrong with parallel conversations - "
                        "turn-taking is noisier than proximity when multiple pairs run at once")
    p.add_argument("--partner-window-s", type=float, default=DEFAULT_PARTNER_WINDOW_S,
                   help="[independent mode] re-discover partnerships independently every N "
                        "seconds, instead of assuming one fixed pairing for the whole "
                        "recording - who's paired with whom can shift over a long session "
                        "(e.g. children in a classroom). A value >= the recording's length "
                        "collapses to a single whole-recording window (the old behavior)")
    p.add_argument("--envelope-weight", type=float, default=DEFAULT_ENVELOPE_WEIGHT,
                   help="[independent mode] weight of the raw energy-envelope correlation "
                        "signal (how similarly two channels' loudness rises/falls over the "
                        "whole recording) relative to proximity (1.0) in whole-recording "
                        "partnership discovery. Cheap, model-free, and empirically strong. "
                        "Set to 0 to disable")
    p.add_argument("--envelope-smooth-ms", type=float, default=DEFAULT_ENVELOPE_SMOOTH_MS,
                   help="[--envelope-weight] smoothing window for envelope correlation. "
                        "Default 0 (no smoothing) tested best - smoothing blurs away the "
                        "fine-grained co-modulation that makes this signal discriminative")
    p.add_argument("--use-voice-embeddings", action="store_true",
                   help="[independent mode] build a voice embedding ('voiceprint') per "
                        "speaker and use it to check WHOSE VOICE bleed-through actually is, "
                        "not just how loud it is - can confirm genuine bleed-through and "
                        "reject same-time noise/unrelated sources. Still fully acoustic (no "
                        "words involved), just a different acoustic feature (vocal timbre) "
                        "than energy. Requires: pip install resemblyzer. Adds real runtime cost")
    p.add_argument("--voiceprint-seconds", type=float, default=DEFAULT_VOICEPRINT_SECONDS,
                   help="[--use-voice-embeddings] seconds of a speaker's own speech used to "
                        "build their reference voiceprint")
    p.add_argument("--max-voice-segments", type=int, default=DEFAULT_MAX_VOICE_SEGMENTS,
                   help="[--use-voice-embeddings] cap on segments embedded per speaker "
                        "(runtime control - lower this if it's too slow)")
    p.add_argument("--voice-affinity-weight", type=float, default=DEFAULT_VOICE_AFFINITY_WEIGHT,
                   help="[--use-voice-embeddings] weight of voice similarity relative to "
                        "proximity (1.0) in whole-recording partnership discovery")
    p.add_argument("--voice-similarity-weight", type=float, default=DEFAULT_VOICE_SIMILARITY_WEIGHT,
                   help="[--use-voice-embeddings] per-turn boost weight from voice similarity")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_pipeline(args)


# ==========================================================================
# EXTENDING THIS (all ideas below are acoustic - no transcription/LLM)
# ==========================================================================
# This script uses a fast energy-based heuristic tuned for close-talk /
# chest-mic setups. If interaction estimates look noisy, consider:
#
#   1. Better VAD per channel: swap frame_energy_db + per_channel_speaking_masks
#      for `webrtcvad` (pip install webrtcvad) or Silero VAD - these are
#      trained specifically to reject noise/breath/rustle, independent
#      per channel just like the current approach.
#
#   2. Cross-channel suppression: before computing energy, subtract a
#      scaled, delay-matched copy of each *other* channel from a channel
#      (simple acoustic echo cancellation) to reduce bleed-through further,
#      which should sharpen both the proximity and envelope-correlation
#      signals.
#
#   3. Real diarization: run pyannote.audio's speaker-diarization pipeline
#      independently on each channel, then fuse per-channel diarization
#      with the dominant-energy channel to confirm identity.
#
#   4. Sound-source localization / beamforming: if you know the physical
#      mic layout, TDOA (time-difference-of-arrival) between channels can
#      give an actual bearing/position estimate for each utterance, a much
#      stronger proximity signal than energy-based bleed-through alone.
#
#   5. Confidence-weighted alignment: if recorders drift in clock rate over
#      long sessions, do the GCC-PHAT alignment in multiple windows spread
#      across the recording (start/middle/end) and check the lag estimate
#      is consistent; if it drifts, resample one channel to match the
#      other's effective clock rate.
#
#   6. Overlapping/sliding windows: build_time_windowed_partnerships currently
#      uses consecutive, non-overlapping windows (--partner-window-s), so a
#      real group change is only reflected at the next window boundary
#      rather than the instant it happens. Sliding, overlapping windows
#      (e.g. re-discover every 30s over a 3-minute lookback) would track
#      changes more smoothly, at higher computational cost.
# ==========================================================================
