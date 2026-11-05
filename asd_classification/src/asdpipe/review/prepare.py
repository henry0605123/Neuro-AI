"""Build the review package for the human checks the audit requires.

Everything goes under ``outputs/review/`` (gitignored):

* ``items.json`` -- what the browser may see: coded clip names and, for the boundary task,
  the boundaries to check. Nothing that identifies a child, a session's group, or what the
  transcript says (except the syllables being aligned, which the check needs).
* ``key.json`` -- what the scorer needs: group, case, turn, the transcript's own text and
  speaker label. Never served.
* ``clips/`` -- 16 kHz mono WAV per clip, plus a spectrogram PNG for the boundary task.

Tasks (``docs/特徵品質稽核清單.md``):

========== ====== =========================================================================
mfa        P1-1   40 utterances, one judgement each -- roughly right, clearly wrong, or
                  inaudible -- with the failure rate compared between groups. The audit's
                  millisecond check (median boundary error < 20 ms) is not done: D13
speaker    P0-2   20 turns: who is speaking, judged without seeing the label (>= 95%)
echo       P0-1   10 high-echo child turns transcribed blind: did the transcript copy the
                  examiner's line?
a12        P0-F5  is a child's loudness change real, or a change of distance?
transcribe P0-1   blind second transcription. Off (transcribe_per_group = 0): transcription
                  reliability is reported as not assessed -- D13
========== ====== =========================================================================

B4 (P0-F6) is not here: it needs the clinician's annotation guideline and a second,
blind annotator before anyone labels anything.

Only the transcribed ASD sessions are used; the untranscribed ones stay untouched (D10).
The person doing these checks knows the diagnoses, while the audit asks for a blind
annotator. Items are coded and -- except the session-ordered transcription task --
shuffled across groups, which lowers that bias but does not remove it.
"""

from __future__ import annotations

import json
import random
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..contracts import Speaker, SyllableSpan, Turn

__all__ = ["prepare_package", "merge_boundaries", "PUBLIC_FIELDS", "public_view",
           "spread_pick", "syllable_labels", "refresh_mfa_display"]

GROUPS = ("TD", "ASD")
_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

# What each task may show in the browser. Anything else stays in key.json.
PUBLIC_FIELDS: dict[str, tuple[str, ...]] = {
    "mfa": ("id", "clip", "png", "dur", "boundaries", "syllables"),
    "speaker": ("id", "clip", "dur"),
    "echo": ("id", "clip", "dur"),
    "a12": ("id", "clip", "dur", "ref", "ref_dur"),
    "transcribe": ("id", "clip", "dur", "ctx"),
}

# Selection sizes. mfa: 20 per group sits inside the audit's 30-50 and lets the failure
# rates be compared between groups. transcribe: the audit asks for 20-25% of sessions
# (here 7 sessions, 2,551 sentences); what it is after is a stable agreement rate in both
# groups, and 150 sentences per group spread round-robin over every session estimates
# that with better coverage of children and rooms at about an eighth of the work (D13).
SIZES = {"mfa_per_group": 20, "speaker_per_group": 10, "echo_total": 10,
         "a12_per_group": 5, "transcribe_per_group": 0}
_CTX_S = 4.0             # how much of what came before a sentence the transcriber may hear
_PAD_S = 0.25            # context either side of a clip


@dataclass
class _Session:
    group: str
    case: str
    audio: np.ndarray
    sr: int
    turns: list[Turn]
    spans: dict[int, tuple[SyllableSpan, ...]] = field(default_factory=dict)
    turn_db: dict[int, float] = field(default_factory=dict)


# ------------------------------------------------------------------------------ pure helpers
def merge_boundaries(spans: Sequence[SyllableSpan], t0: float, tol_s: float = 1e-3
                     ) -> tuple[list[float], list[dict]]:
    """Syllable spans -> (unique boundary times relative to t0, syllables as index pairs).

    Adjacent syllables usually share a boundary; it is one line to drag, not two.
    """
    times: list[float] = []

    def index_of(t: float) -> int:
        for k, x in enumerate(times):
            if abs(x - t) <= tol_s:
                return k
        times.append(t)
        return len(times) - 1

    raw = [(index_of(s.t_start - t0), index_of(s.t_end - t0), s.text) for s in spans]
    order = sorted(range(len(times)), key=lambda k: times[k])
    rank = {k: r for r, k in enumerate(order)}
    bounds = [round(times[k], 4) for k in order]
    syllables = [{"a": rank[a], "b": rank[b], "text": text} for a, b, text in raw]
    return bounds, syllables


def syllable_labels(text: str, spans: Sequence[SyllableSpan]) -> list[str]:
    """The character each syllable spells, when the transcript's Han characters line up one
    to one with the aligned syllables; otherwise the aligner's own phone string. MFA labels
    syllables with phones (tɕow, pej, ...), which nobody checking a child's speech can read."""
    chars = re.findall("[一-鿿]", text or "")
    if len(chars) == len(spans):
        return chars
    return [s.text for s in spans]


def public_view(task: str, item: Mapping) -> dict:
    return {k: item[k] for k in PUBLIC_FIELDS[task] if k in item}


def spread_pick(by_case: Mapping[str, Sequence], n: int, rng: random.Random) -> list:
    """Take n items round-robin over cases, so every child contributes before any child
    contributes twice."""
    pools = {c: list(v) for c, v in by_case.items() if v}
    for v in pools.values():
        rng.shuffle(v)
    cases = sorted(pools)
    rng.shuffle(cases)
    out: list = []
    while len(out) < n and any(pools[c] for c in cases):
        for c in cases:
            if pools[c] and len(out) < n:
                out.append(pools[c].pop())
    return out


def _codes(rng: random.Random, n: int, taken: set[str]) -> list[str]:
    out = []
    while len(out) < n:
        c = "".join(rng.choice(_ALPHABET) for _ in range(6))
        if c not in taken:
            taken.add(c)
            out.append(c)
    return out


# ------------------------------------------------------------------------------ audio out
def _write_clip(path: Path, audio: np.ndarray, sr: int, t0: float, t1: float) -> float:
    import soundfile as sf

    a = max(0, int(round(t0 * sr)))
    b = min(len(audio), int(round(t1 * sr)))
    sf.write(path, np.asarray(audio[a:b], dtype=np.float32), sr, subtype="PCM_16")
    return (b - a) / sr


def _spectrogram_png(path: Path, wav: Path, px_per_s: int = 500, height: int = 240) -> None:
    """Grey spectrogram, 0-8 kHz, no margins: x maps linearly onto the clip's duration.

    2 ms per pixel, so the zoomed view in the browser crops a sharp picture instead of
    stretching an overview -- the STFT hop is 4 ms, finer pixels would add nothing."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import soundfile as sf

    x, sr = sf.read(wav, dtype="float32")
    dur = len(x) / sr
    width = max(200, int(np.ceil(dur * px_per_s)))
    n_fft, hop = 512, max(1, sr // 250)                       # 4 ms hop
    # Centred frames: frame k is centred on k*hop, so the picture lines up with the time
    # axis the boundaries are drawn on. Uncentred frames would shift it by up to 32 ms --
    # more than the 20 ms the check is about.
    xp = np.pad(x, (n_fft // 2, n_fft // 2))
    frames = np.lib.stride_tricks.sliding_window_view(xp, n_fft)[::hop] * np.hanning(n_fft)
    mag = np.abs(np.fft.rfft(frames, axis=1)).T
    keep = int(round(8000 / (sr / 2) * (mag.shape[0] - 1))) + 1
    db = 20 * np.log10(mag[:keep] + 1e-6)
    db = np.clip(db, db.max() - 70, None)
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    step = hop / sr
    ax.imshow(db, origin="lower", aspect="auto", cmap="gray_r", interpolation="bilinear",
              extent=(-step / 2, (db.shape[1] - 0.5) * step, 0, 8000))
    ax.set_xlim(0, dur)
    ax.axis("off")
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ------------------------------------------------------------------------------ loading
def _load_sessions(cfg: Mapping, cache_dir: Path, log) -> list[_Session]:
    from ..align import aligner_from_config
    from ..cli.main import _backends
    from ..extract import _measured_turn_db, child_alignment_inputs
    from ..normalize import normalize_corpus
    from ..realdata import load_cohort

    backends = _backends(cfg)
    aligner = aligner_from_config(cfg, cache_dir)
    rd = cfg["realdata"]
    sessions: list[_Session] = []
    for group in GROUPS:
        gcfg = {**cfg, "realdata": {**rd, "cohort": group}}
        raw, _ = load_cohort(rd["root"], gcfg, cache_dir=cache_dir)
        corpus, _ = normalize_corpus(raw, gcfg)
        log(f"[review] {group}: {len(corpus.subjects)} sessions")
        for s in corpus.subjects:
            turns = list(s.session.turns)
            kids = [t for t in turns if t.speaker is Speaker.CHILD and t.text]
            words = dict(zip([t.turn_id for t in kids], backends.segment([t.text for t in kids])))
            alignable, align_words = child_alignment_inputs(turns, words)
            spans = aligner.align(s.audio, s.sample_rate, alignable, align_words) if alignable else {}
            sessions.append(_Session(group, s.subject_id, s.audio, s.sample_rate, turns,
                                     dict(spans), _measured_turn_db(s, s.sample_rate)))
    return sessions


# ------------------------------------------------------------------------------ tasks
def _mfa(sessions, rng):
    out = []
    for g in GROUPS:
        by_case = {}
        for s in (x for x in sessions if x.group == g):
            by_case[s.case] = [(s, t) for t in s.turns
                               if len(s.spans.get(t.turn_id, ())) >= 3 and 0.6 <= t.duration <= 8.0]
        out += spread_pick(by_case, SIZES["mfa_per_group"], rng)
    return out


def _speaker(sessions, rng):
    out = []
    for g in GROUPS:
        for who in (Speaker.CHILD, Speaker.EXAMINER):
            by_case = {s.case: [(s, t) for t in s.turns
                                if t.speaker is who and 0.5 <= t.duration <= 8.0]
                       for s in sessions if s.group == g}
            out += spread_pick(by_case, SIZES["speaker_per_group"] // 2, rng)
    return out


def _echo(sessions, rng, cfg):
    from ..features.e_text import _overlap_ratio, pair_turns

    cut = float(cfg["features"]["e_stream"].get("exclude_verbatim_echo_overlap", 0.8))
    cands = {g: {} for g in GROUPS}
    for s in sessions:
        found = []
        for ex, ch in pair_turns(s.turns):
            if len(ch.text) < 2 or ch.duration < 0.3:
                continue
            ov = _overlap_ratio(ch.text, ex.text)
            if np.isfinite(ov) and ov >= cut:
                found.append((s, ch, ex, ov))
        cands[s.group][s.case] = found
    half = SIZES["echo_total"] // 2
    picked = {g: spread_pick(cands[g], half, rng) for g in GROUPS}
    short = SIZES["echo_total"] - sum(len(v) for v in picked.values())
    for g in GROUPS:                       # a group with too few echoes is topped up by the other
        if short <= 0:
            break
        rest = {c: [x for x in v if x not in picked[g]] for c, v in cands[g].items()}
        extra = spread_pick(rest, short, rng)
        picked[g] += extra
        short -= len(extra)
    return picked["TD"] + picked["ASD"]


def _a12(sessions, rng):
    out = []
    for g in GROUPS:
        group_sessions = [s for s in sessions if s.group == g]
        rng.shuffle(group_sessions)
        for s in group_sessions:
            kids = [t for t in s.turns if t.speaker is Speaker.CHILD and t.duration >= 0.8
                    and np.isfinite(s.turn_db.get(t.turn_id, np.nan))]
            if len(kids) < 10:
                continue
            db = np.array([s.turn_db[t.turn_id] for t in kids])
            med = float(np.median(db))
            target = kids[int(np.argmax(np.abs(db - med)))]
            ref = kids[int(np.argmin(np.abs(db - med)))]
            out.append((s, target, ref, med))
            if sum(1 for x in out if x[0].group == g) >= SIZES["a12_per_group"]:
                break
    return out


def _transcribe(sessions, rng):
    if SIZES["transcribe_per_group"] <= 0:
        return []
    picked = []
    for g in GROUPS:
        by_case = {s.case: [(s, t) for t in s.turns if t.duration >= 0.3]
                   for s in sessions if s.group == g}
        picked += spread_pick(by_case, SIZES["transcribe_per_group"], rng)
    return picked


# ------------------------------------------------------------------------------ driver
def prepare_package(cfg: Mapping, out_dir: Path, cache_dir: Path, seed: int = 20261001,
                    force: bool = False, log=print) -> Path:
    out_dir = Path(out_dir)
    if (out_dir / "items.json").exists():
        if not force:
            raise FileExistsError(f"{out_dir / 'items.json'} exists; answers refer to its codes. "
                                  "Pass force=True to set it aside and build a new one.")
        backup = out_dir.with_name(f"{out_dir.name}_{datetime.now():%Y%m%d_%H%M%S}")
        shutil.move(str(out_dir), str(backup))
        log(f"[review] previous package moved to {backup}")
    clips = out_dir / "clips"
    clips.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    sessions = _load_sessions(cfg, cache_dir, log)
    taken: set[str] = set()
    items: dict[str, list[dict]] = {t: [] for t in PUBLIC_FIELDS}
    key: dict[str, dict[str, dict]] = {t: {} for t in PUBLIC_FIELDS}

    def clip(code, s, t0, t1, suffix=""):
        t0 = max(0.0, t0)
        name = f"{code}{suffix}.wav"
        dur = _write_clip(clips / name, s.audio, s.sr, t0, t1)
        return name, round(dur, 3), t0

    # ---- mfa ----
    picks = _mfa(sessions, rng)
    for code, (s, t) in zip(_codes(rng, len(picks), taken), picks):
        name, dur, t0 = clip(code, s, t.t_start - _PAD_S, t.t_end + _PAD_S)
        bounds, syls = merge_boundaries(s.spans[t.turn_id], t0)
        for syl, lab in zip(syls, syllable_labels(t.text, s.spans[t.turn_id])):
            syl["text"] = lab
        _spectrogram_png(clips / f"{code}.png", clips / name)
        items["mfa"].append({"id": code, "clip": name, "png": f"{code}.png", "dur": dur,
                             "boundaries": bounds, "syllables": syls})
        key["mfa"][code] = {"group": s.group, "case": s.case, "turn_id": t.turn_id,
                            "clip_t0": t0, "mfa_boundaries": bounds}

    # ---- speaker ----
    picks = _speaker(sessions, rng)
    for code, (s, t) in zip(_codes(rng, len(picks), taken), picks):
        name, dur, t0 = clip(code, s, t.t_start - 0.1, t.t_end + 0.1)
        items["speaker"].append({"id": code, "clip": name, "dur": dur})
        key["speaker"][code] = {"group": s.group, "case": s.case, "turn_id": t.turn_id,
                                "label": t.speaker.name.lower()}

    # ---- echo ----
    picks = _echo(sessions, rng, cfg)
    for code, (s, ch, ex, ov) in zip(_codes(rng, len(picks), taken), picks):
        name, dur, _ = clip(code, s, ch.t_start - 0.1, ch.t_end + 0.1)
        items["echo"].append({"id": code, "clip": name, "dur": dur})
        key["echo"][code] = {"group": s.group, "case": s.case, "turn_id": ch.turn_id,
                             "child_text": ch.text, "examiner_text": ex.text,
                             "overlap": round(float(ov), 3)}

    # ---- a12 ----
    picks = _a12(sessions, rng)
    for code, (s, tgt, ref, med) in zip(_codes(rng, len(picks), taken), picks):
        name, dur, _ = clip(code, s, tgt.t_start - 0.1, tgt.t_end + 0.1)
        rname, rdur, _ = clip(code, s, ref.t_start - 0.1, ref.t_end + 0.1, suffix="_ref")
        items["a12"].append({"id": code, "clip": name, "dur": dur, "ref": rname, "ref_dur": rdur})
        key["a12"][code] = {"group": s.group, "case": s.case, "turn_id": tgt.turn_id,
                            "ref_turn_id": ref.turn_id,
                            "target_db": round(float(s.turn_db[tgt.turn_id]), 2),
                            "ref_db": round(float(s.turn_db[ref.turn_id]), 2),
                            "median_db": round(med, 2)}

    # ---- transcribe: sentences from every session, with what came just before ----
    picks = _transcribe(sessions, rng)
    for code, (s, t) in zip(_codes(rng, len(picks), taken), picks):
        name, dur, _ = clip(code, s, t.t_start - 0.1, t.t_end + 0.1)
        ctx, _, _ = clip(code, s, t.t_start - _CTX_S, t.t_start, suffix="_ctx")
        items["transcribe"].append({"id": code, "clip": name, "dur": dur, "ctx": ctx})
        key["transcribe"][code] = {"group": s.group, "case": s.case, "turn_id": t.turn_id,
                                   "label": t.speaker.name.lower(), "text": t.text}

    for task in PUBLIC_FIELDS:                           # groups interleaved
        rng.shuffle(items[task])

    meta = {"created": f"{datetime.now():%Y-%m-%d %H:%M}", "seed": seed,
            "counts": {t: len(v) for t, v in items.items()}}
    public = {"meta": meta, "tasks": {t: [public_view(t, x) for x in v] for t, v in items.items()}}
    (out_dir / "items.json").write_text(json.dumps(public, ensure_ascii=False), encoding="utf-8")
    (out_dir / "key.json").write_text(json.dumps({"meta": meta, "key": key}, ensure_ascii=False),
                                      encoding="utf-8")
    log(f"[review] items: {meta['counts']}")
    return out_dir


def refresh_mfa_display(cfg: Mapping, review_dir: Path, cache_dir: Path, log=print) -> int:
    """Re-render the boundary task's spectrograms and relabel its syllables in place.

    Item codes, clips and boundaries stay exactly as they were, so answers already given
    still refer to the same thing. Only what is drawn changes.
    """
    from ..normalize import normalize_corpus
    from ..realdata import load_cohort

    review_dir = Path(review_dir)
    items_path = review_dir / "items.json"
    items = json.loads(items_path.read_text(encoding="utf-8"))
    key = json.loads((review_dir / "key.json").read_text(encoding="utf-8"))["key"]["mfa"]
    rd = cfg["realdata"]
    text_of: dict[tuple[str, int], str] = {}
    for group in GROUPS:
        gcfg = {**cfg, "realdata": {**rd, "cohort": group}}
        raw, _ = load_cohort(rd["root"], gcfg, cache_dir=cache_dir)
        corpus, _ = normalize_corpus(raw, gcfg)
        for subj in corpus.subjects:
            for t in subj.session.turns:
                text_of[(subj.subject_id, t.turn_id)] = t.text
    relabelled = 0
    for it in items["tasks"]["mfa"]:
        k = key[it["id"]]
        _spectrogram_png(review_dir / "clips" / it["png"], review_dir / "clips" / it["clip"])
        chars = re.findall("[\u4e00-\u9fff]", text_of.get((k["case"], k["turn_id"]), ""))
        if len(chars) == len(it["syllables"]):
            for syl, ch in zip(it["syllables"], chars):
                syl["text"] = ch
            relabelled += 1
    items_path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    log(f"[review] spectrograms redrawn; {relabelled}/{len(items['tasks']['mfa'])} items "
        "labelled with characters")
    return relabelled
