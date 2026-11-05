"""E stream -- transcript semantics and reciprocity (plan doc 5.5b). The main stream.

Organised around **reciprocity**, following the plan doc: G1 and F1 are the two ends of one
axis (too dissimilar = off topic, too similar = echolalia), and F7/F8 are the multi-turn
and bidirectional versions of the same idea. All four are non-parametric -- zero trainable
parameters -- which is why they are usable at n~20.

Also computed here (DEVIATIONS D2): response latency and volume following (A12). The task
spec groups them with the E-stream interaction features; the plan doc lists A12 under the B
stream. The assignment is one line in `configs/pipeline.yaml`.

Turn-pairing rules are frozen in code, never adjusted case by case (audit P0-2), and the
same rules are reused by F7 and F8.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence

import numpy as np

from ..contracts import Speaker, Turn
from .common import child_turns, min_values, ols_slope, pool, sliding_windows, stat

__all__ = ["EStreamFeatures", "FUNCTION_WORDS", "pair_turns", "extract_e",
           "examiner_probe_features"]

# Repeated function-word strings ("然後然後", "就是就是") are developmentally normal at
# 4-6 years, so an n-gram made only of these does not count as stereotyped (audit P0-F3).
FUNCTION_WORDS: frozenset[str] = frozenset({
    "的", "了", "嗎", "呢", "吧", "子", "們", "著", "麼", "個", "嘛", "啦", "喔", "耶",
    "我", "你", "他", "她", "它", "是", "有", "在", "和", "跟", "就", "都", "也", "很",
    "不", "沒", "要", "會", "可以", "那", "這", "那個", "這個", "然後", "就是", "一個",
})

_MAX_PAIR_GAP_S = 10.0


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _char_ngrams(text: str, n: int) -> set[str]:
    return {text[i : i + n] for i in range(len(text) - n + 1)} if len(text) >= n else set()


def _overlap_ratio(child: str, examiner: str, n: int = 2) -> float:
    """Literal repetition: proportion of the child's character bigrams seen in the prompt."""
    c = _char_ngrams(child, n)
    if not c:
        return float("nan")
    return len(c & _char_ngrams(examiner, n)) / len(c)


@dataclass
class EStreamFeatures:
    g1_similarity: np.ndarray = field(default_factory=lambda: np.zeros(0))
    f1_embed_similarity: np.ndarray = field(default_factory=lambda: np.zeros(0))
    f1_overlap: np.ndarray = field(default_factory=lambda: np.zeros(0))
    response_latency: np.ndarray = field(default_factory=lambda: np.zeros(0))
    scalars: dict[str, float] = field(default_factory=dict)
    quality: dict[str, float] = field(default_factory=dict)
    min_n: int = 1                  # every pooled series here is pair-based

    def pooled(self, functionals: Sequence[str]) -> dict[str, float]:
        out = dict(self.scalars)
        g1 = pool(self.g1_similarity, functionals, "e_g1_sim", self.min_n)
        g1.pop("e_g1_sim_median", None)     # P20: the same number as e_g1_similarity_median
        out.update(g1)
        out.update(pool(self.f1_embed_similarity, functionals, "e_f1_embed", self.min_n))
        out.update(pool(self.response_latency, functionals, "e_latency", self.min_n))
        return out


def pair_turns(turns: Sequence[Turn]) -> list[tuple[Turn, Turn]]:
    """(examiner turn, child response) pairs. One rule, written once, reused by G1, F1, F8,
    A12 and response latency.

    A response is the child's **first** turn after an examiner turn. Until 2026-10-01 every
    later child turn was paired with the same examiner turn too, so a child carrying on
    with their own next sentence counted as answering the examiner: 24.6% of TD pairs were
    such continuations (health check P10), and their "latency" included the child's own
    previous sentence.
    """
    pairs: list[tuple[Turn, Turn]] = []
    last_examiner: Turn | None = None
    for t in turns:
        if t.speaker is Speaker.EXAMINER:
            last_examiner = t
        elif t.speaker is Speaker.CHILD:
            if (last_examiner is not None
                    and 0.0 <= t.t_start - last_examiner.t_end <= _MAX_PAIR_GAP_S):
                pairs.append((last_examiner, t))
            last_examiner = None
    return pairs


def speech_blocks(turns: Sequence[Turn], embeddings: Mapping[int, np.ndarray]
                  ) -> tuple[list[Turn], dict[int, np.ndarray]]:
    """Merge each maximal run of consecutive same-speaker turns into one block.

    Why F8 needs this. The two cohorts were segmented under different conventions: the ASD
    annotators were asked to split at pauses, the TD annotators were not, and the result is
    2.47 consecutive same-speaker turns per run against 1.80. F8 pairs a turn with the reply
    that follows it, and ``_reverse_pairs`` keeps only the *last* turn of a run -- so with
    fine segmentation it compares the reply against the final fragment of an utterance, and
    with coarse segmentation against the whole thing. The measured asymmetry then partly
    reflects who transcribed the session.

    Merging removes the difference by construction: after this, every block is followed by a
    different speaker in both cohorts. It is applied to **both** arms rather than only to the
    finer one -- correcting a single cohort would leave runs of 1.0 against 1.80 and simply
    reverse the asymmetry.

    The block embedding is the renormalised mean of its members' embeddings, not a re-encode
    of the concatenated text: the sentence encoder runs upstream and re-running it here would
    make this function depend on a GPU model. For sentences from one speaker in one breath
    group the two are close, but it is an approximation and is recorded as one.
    """
    if not turns:
        return [], {}
    ordered = sorted(turns, key=lambda t: t.t_start)
    blocks: list[Turn] = []
    members: list[list[Turn]] = []
    for t in ordered:
        if blocks and blocks[-1].speaker is t.speaker:
            members[-1].append(t)
            blocks[-1] = replace(
                blocks[-1], t_end=max(blocks[-1].t_end, t.t_end),
                text=f"{blocks[-1].text}{t.text}",
                is_question=blocks[-1].is_question or t.is_question,
                is_yes_no=blocks[-1].is_yes_no or t.is_yes_no,
            )
        else:
            blocks.append(t)
            members.append([t])

    emb: dict[int, np.ndarray] = {}
    for blk, mem in zip(blocks, members):
        vecs = [embeddings[m.turn_id] for m in mem if m.turn_id in embeddings]
        if not vecs:
            continue
        v = np.mean(np.stack(vecs), axis=0)
        n = float(np.linalg.norm(v))
        emb[blk.turn_id] = (v / n) if n > 1e-9 else v
    return blocks, emb


def _reverse_pairs(turns: Sequence[Turn]) -> list[tuple[Turn, Turn]]:
    """(child turn, the examiner turn that follows it) -- F8's mirror image.

    Deliberately the same adjacency rule as ``pair_turns`` with the roles swapped, so the
    difference measures behaviour and not algorithm (audit P0-F8).
    """
    pairs: list[tuple[Turn, Turn]] = []
    last_child: Turn | None = None
    for t in turns:
        if t.speaker is Speaker.CHILD:
            last_child = t
        elif t.speaker is Speaker.EXAMINER:
            if (last_child is not None
                    and 0.0 <= t.t_start - last_child.t_end <= _MAX_PAIR_GAP_S):
                pairs.append((last_child, t))
            last_child = None
    return pairs


def _repeated_ngrams(word_lists: Sequence[Sequence[str]], n_range: tuple[int, int],
                     min_repeats: int) -> tuple[int, int, int]:
    """(repeated types, total types, repeated tokens) for F3."""
    counts: dict[tuple[str, ...], int] = {}
    for words in word_lists:
        for n in range(n_range[0], n_range[1] + 1):
            for i in range(len(words) - n + 1):
                g = tuple(words[i : i + n])
                if all(w in FUNCTION_WORDS for w in g):
                    continue
                counts[g] = counts.get(g, 0) + 1
    if not counts:
        return 0, 0, 0
    repeated = {g: c for g, c in counts.items() if c >= min_repeats}
    return len(repeated), len(counts), sum(repeated.values())


def extract_e(
    turns: Sequence[Turn],
    embeddings: Mapping[int, np.ndarray],
    words: Mapping[int, Sequence[str]],
    turn_db: Mapping[int, float],
    cfg: Mapping,
) -> EStreamFeatures:
    e_cfg = cfg["features"]["e_stream"]
    min_chars = int(e_cfg["min_answer_chars"])
    n_range = tuple(e_cfg["ngram_range"])
    min_repeats = int(e_cfg["ngram_min_repeats"])
    windows = list(e_cfg["f7_windows"])
    f1_hi = float(e_cfg["f1_high_similarity_threshold"])
    g1_lo = float(e_cfg["g1_low_similarity_threshold"])
    drop_yes_no = bool(e_cfg["exclude_yes_no_questions"])
    echo_cut = float(e_cfg.get("exclude_verbatim_echo_overlap", 1.01))
    min_pairs = int(e_cfg["min_pairs_per_subject"])
    a12_min = int(e_cfg.get("a12_min_pairs", min_pairs))
    mv = min_values(cfg)
    nan = float("nan")

    pairs = pair_turns(turns)
    g1_sim, f1_embed, f1_overlap, latency = [], [], [], []
    n_excluded_yes_no = n_excluded_short = n_excluded_echo = n_excluded_not_question = 0

    for ex, ch in pairs:
        e_emb, c_emb = embeddings.get(ex.turn_id), embeddings.get(ch.turn_id)
        latency.append(ch.t_start - ex.t_end)
        if e_emb is not None and c_emb is not None:
            f1_embed.append(_cos(c_emb, e_emb))
        f1_overlap.append(_overlap_ratio(ch.text, ex.text))
        # G1 relevance: answers to the examiner's open-ended questions only (plan doc
        # 4b.1, audit P0-F1 "僅計開放式問答對"), long enough to embed. Until 2026-10-01 any
        # examiner turn qualified, and only 22% of TD G1 pairs followed a question
        # (health check P10).
        if not ex.is_question:
            n_excluded_not_question += 1
            continue
        if drop_yes_no and ex.is_yes_no:
            n_excluded_yes_no += 1
            continue
        if len(ch.text) < min_chars:
            n_excluded_short += 1
            continue
        # A verbatim repetition of the prompt says nothing about topical appropriateness:
        # it is the F1 construct, and leaving it in puts a large mass at similarity ~1.0
        # that swamps G1 (measured: G1 median correlated rho = +0.57 with the injected
        # echolalia rate but only +0.17 with topic relevance). Excluded here, counted by F1.
        ov = _overlap_ratio(ch.text, ex.text)
        if np.isfinite(ov) and ov >= echo_cut:
            n_excluded_echo += 1
            continue
        if e_emb is not None and c_emb is not None:
            g1_sim.append(_cos(c_emb, e_emb))

    g1 = np.asarray([v for v in g1_sim if np.isfinite(v)], dtype=float)
    f1e = np.asarray([v for v in f1_embed if np.isfinite(v)], dtype=float)
    f1o = np.asarray([v for v in f1_overlap if np.isfinite(v)], dtype=float)
    lat = np.asarray([v for v in latency if np.isfinite(v)], dtype=float)

    # ---- F7: multi-turn semantic adaptation, all three windows (audit P0-F7) ----
    # Computed and reported, but **not a feature** since 2026-09-12. The plan anticipated
    # this: section 4b.1 says "4-6 歲兒童本就不擅長維持長篇互惠對話。先檢視對照組分布
    # ... 此特徵在本年齡層可能不適用。這是要如實報告的發現。" Measured on the 19 TD
    # subjects: slope median about -0.0002, positive for only 4 of 19 (sign test p = 0.019),
    # and the three windows correlate 0.993/0.994 across subjects, so they are one test
    # rather than three. Kept in `quality` because the audit explicitly wants it reported.
    f7: dict[str, float] = {}
    for w in windows:
        wins = sliding_windows(g1, w)
        if len(wins) >= 3:
            means = np.array([float(np.mean(x)) for x in wins])
            f7[f"e_f7_slope_w{w}"] = ols_slope(np.arange(len(means)), means)
            f7[f"e_f7_var_shrink_w{w}"] = float(np.std(wins[-1]) - np.std(wins[0]))
        else:
            f7[f"e_f7_slope_w{w}"] = 0.0
            f7[f"e_f7_var_shrink_w{w}"] = 0.0
    f7 = {k: (0.0 if not np.isfinite(v) else float(v)) for k, v in f7.items()}
    scalars: dict[str, float] = {}

    # ---- F8: bidirectional asymmetry, identical algorithm in both directions ----
    # Paired on speech blocks rather than raw turns so the measure cannot inherit the two
    # cohorts' different segmentation conventions; see `speech_blocks`.
    blocks, bemb = speech_blocks(turns, embeddings)
    child_follow = [
        _cos(bemb[ch.turn_id], bemb[ex.turn_id])
        for ex, ch in pair_turns(blocks)
        if ch.turn_id in bemb and ex.turn_id in bemb
    ]
    examiner_follow = [
        _cos(bemb[ex.turn_id], bemb[ch.turn_id])
        for ch, ex in _reverse_pairs(blocks)
        if ch.turn_id in bemb and ex.turn_id in bemb
    ]
    cf = stat(child_follow, np.mean, min_pairs)
    ef = stat(examiner_follow, np.mean, min_pairs)
    scalars["e_f8_child_follows"] = cf
    scalars["e_f8_examiner_follows"] = ef
    scalars["e_f8_asymmetry"] = ef - cf          # NaN if either direction is unmeasurable

    # ---- F3 stereotyped / fixed phrases ----
    kids = child_turns(turns)
    kid_words = [list(words.get(t.turn_id, ())) for t in kids]
    rep_types, all_types, rep_tokens = _repeated_ngrams(kid_words, n_range, min_repeats)
    flat = [w for ws in kid_words for w in ws]
    enough_words = len(flat) >= mv
    scalars["e_f3_repeated_type_ratio"] = (float(rep_types / all_types)
                                           if enough_words and all_types else nan)
    scalars["e_f3_repeated_token_ratio"] = float(rep_tokens / len(flat)) if enough_words else nan
    scalars["e_f3_repeated_types"] = float(rep_types)

    # ---- lexical ----
    lens = [len(ws) for ws in kid_words if ws]
    scalars["e_mlu_words"] = stat(lens, np.mean, mv)
    scalars["e_mlu_sd"] = stat(lens, np.std, mv)
    scalars["e_ttr"] = float(len(set(flat)) / len(flat)) if enough_words else nan
    scalars["e_child_total_words"] = float(len(flat))

    # ---- G1 / F1 headline indicators (the frozen primary indicators, decision_rules.yaml) ----
    # Fewer than min_pairs_per_subject usable pairs: G1/F1 cannot be measured (P12).
    g1_ok = g1.size >= min_pairs
    scalars["e_g1_similarity_median"] = float(np.median(g1)) if g1_ok else nan
    # 🔴 Both ratios use an ABSOLUTE threshold, frozen in config.
    #
    # They used to compare each subject against a quantile of that same subject's own
    # distribution -- "the fraction of my pairs below my own 25th percentile" -- which is
    # 0.25 for everyone by definition. On real data all 19 subjects came out at
    # 0.250 +- 0.001, i.e. the features carried no information at all. They looked like they
    # worked on synthetic data only because echolalia produces many pairs tied at 1.0, and
    # ties push the quantile-based fraction away from 0.25.
    #
    # `e_f1_embed_high_ratio` is F1's pre-registered primary indicator in
    # decision_rules.yaml, so this was not a cosmetic bug.
    scalars["e_g1_low_ratio"] = float(np.mean(g1 < g1_lo)) if g1_ok else nan
    scalars["e_f1_embed_high_ratio"] = (float(np.mean(f1e >= f1_hi))
                                        if f1e.size >= min_pairs else nan)
    scalars["e_f1_overlap_mean"] = stat(f1o, np.mean, min_pairs)
    scalars["e_f1_overlap_p90"] = stat(f1o, lambda v: np.percentile(v, 90), min_pairs)

    # ---- interaction (DEVIATIONS D2) ----
    db_pairs = [(turn_db.get(ex.turn_id, np.nan), turn_db.get(ch.turn_id, np.nan))
                for ex, ch in pairs]
    db_pairs = [(a, b) for a, b in db_pairs if np.isfinite(a) and np.isfinite(b)]
    scalars["e_volume_following_slope"] = (
        float(ols_slope([a for a, _ in db_pairs], [b for _, b in db_pairs]))
        if len(db_pairs) >= a12_min else nan)
    scalars["e_latency_median"] = stat(lat, np.median, min_pairs)

    quality = {
        "e_n_pairs": float(len(pairs)),
        "e_n_g1_pairs": float(g1.size),
        "e_pairs_ok": float(len(pairs) >= min_pairs),
        "e_g1_pairs_ok": float(g1.size >= min_pairs),
        "e_excluded_yes_no": float(n_excluded_yes_no),
        "e_excluded_short": float(n_excluded_short),
        "e_excluded_verbatim_echo": float(n_excluded_echo),
        "e_excluded_not_question": float(n_excluded_not_question),
        "e_f7_windows_ok": float(all(len(sliding_windows(g1, w)) >= 3 for w in windows)),
        **f7,
    }
    return EStreamFeatures(
        g1_similarity=g1, f1_embed_similarity=f1e, f1_overlap=f1o,
        response_latency=lat, scalars=scalars, quality=quality, min_n=min_pairs,
    )


def examiner_probe_features(
    turns: Sequence[Turn],
    embeddings: Mapping[int, np.ndarray],
    words: Mapping[int, Sequence[str]],
    turn_db: Mapping[int, float],
) -> dict[str, float]:
    """Examiner-side-only descriptors for the examiner-leakage probe (audit P0-0C).

    Nothing here may touch a child turn -- that is the whole point of the probe.
    """
    ex = [t for t in turns if t.speaker is Speaker.EXAMINER]
    if not ex:
        return {k: 0.0 for k in (
            "p_n_turns", "p_mean_chars", "p_sd_chars", "p_question_ratio", "p_yes_no_ratio",
            "p_mean_dur", "p_speaking_rate", "p_mean_db", "p_sd_db", "p_mean_words",
            "p_self_similarity", "p_wait_median")}
    chars = np.array([len(t.text) for t in ex], dtype=float)
    durs = np.array([t.duration for t in ex], dtype=float)
    dbs = np.array([turn_db.get(t.turn_id, np.nan) for t in ex], dtype=float)
    dbs = dbs[np.isfinite(dbs)]
    nwords = np.array([len(words.get(t.turn_id, ())) for t in ex], dtype=float)
    sims = [
        _cos(embeddings[a.turn_id], embeddings[b.turn_id])
        for a, b in zip(ex[:-1], ex[1:])
        if a.turn_id in embeddings and b.turn_id in embeddings
    ]
    sims = [s for s in sims if np.isfinite(s)]
    waits = []
    for prev, nxt in zip(turns[:-1], turns[1:]):
        if prev.speaker is Speaker.CHILD and nxt.speaker is Speaker.EXAMINER:
            waits.append(nxt.t_start - prev.t_end)
    return {
        "p_n_turns": float(len(ex)),
        "p_mean_chars": float(np.mean(chars)),
        "p_sd_chars": float(np.std(chars)),
        "p_question_ratio": float(np.mean([t.is_question for t in ex])),
        "p_yes_no_ratio": float(np.mean([t.is_yes_no for t in ex])),
        "p_mean_dur": float(np.mean(durs)),
        "p_speaking_rate": float(np.mean(chars / np.maximum(durs, 1e-6))),
        "p_mean_db": float(np.mean(dbs)) if dbs.size else 0.0,
        "p_sd_db": float(np.std(dbs)) if dbs.size else 0.0,
        "p_mean_words": float(np.mean(nwords)),
        "p_self_similarity": float(np.mean(sims)) if sims else 0.0,
        "p_wait_median": float(np.median(waits)) if waits else 0.0,
    }
