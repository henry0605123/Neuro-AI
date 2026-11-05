"""Entry point behind `make smoke` / `make validate` / `make checklist`."""

from __future__ import annotations

import json
import pickle
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Mapping

import numpy as np

from ..assemble import build_table
from ..config import CACHE_DIR, OUTPUT_DIR, REPO_ROOT, load_decision_rules, load_pipeline_config
from ..eval import analysis as an
from ..eval import metrics as mt
from ..eval.checklist import evaluate_items, write_checklist
from ..eval.intake import write_intake_report
from ..eval.report import write_report
from ..extract import extract_corpus
from ..features.backends import Backends
from ..run import MODELS, run_cv
from ..synth.generator import SynthConfig, generate_corpus

STATE = OUTPUT_DIR / "state.pkl"


def _backends(cfg: Mapping) -> Backends:
    b = Backends.from_config(cfg)
    b.cache_dir = str(CACHE_DIR)
    return b


def _build(cfg: Mapping, backends: Backends, synth_over: Mapping | None = None):
    sc = SynthConfig.from_dict({**cfg["synth"], **(synth_over or {})})
    corpus = generate_corpus(sc)
    rows = extract_corpus(corpus, cfg, backends)
    return corpus, build_table(rows)


def _run_phase0_analyses(table, cfg: Mapping, rules: Mapping) -> tuple[dict, dict]:
    """Audit P0-4: criterion validity FIRST, group differences SECOND. Order is not optional."""
    n_boot = int(cfg["analysis"]["spearman_bootstrap"])
    criterion = an.criterion_validity(table, rules, n_boot=n_boot, seed=0)
    group = an.group_differences(table, n_boot=int(cfg["analysis"]["n_bootstrap"]) // 5, seed=0)
    return criterion, group


def cmd_smoke(cfg: Mapping, rules: Mapping) -> int:
    t0 = time.time()
    backends = _backends(cfg)
    backends.warm_up()
    t_warm = time.time() - t0

    t = time.time()
    corpus, table = _build(cfg, backends)
    t_extract = time.time() - t

    t = time.time()
    cv = run_cv(table, cfg, rules, with_ablation=True, run_probe=True)
    t_cv = time.time() - t

    t = time.time()
    criterion, group = _run_phase0_analyses(table, cfg, rules)
    t_analysis = time.time() - t

    versions = backends.versions()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with STATE.open("wb") as fh:
        pickle.dump({"table": table, "cv": cv, "criterion": criterion, "group": group,
                     "versions": versions, "cfg": cfg, "rules": rules}, fh)

    validate = _load_validate()
    write_report(OUTPUT_DIR / "report.md", table, cv, criterion, group, cfg, rules, versions,
                 timings={"warm_up": t_warm, "extract": t_extract, "cv": t_cv,
                          "analysis": t_analysis}, validate=validate)
    write_checklist(OUTPUT_DIR / "checklist_report.md",
                    evaluate_items(table, cv, criterion, cfg, versions))
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps({"across_seeds": cv["across_seeds"],
                    "delivered_model": cv["delivered_model"],
                    "examiner_probe": cv.get("examiner_probe", {}),
                    "criterion_validity": criterion,
                    "group_differences": group}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    print(f"[smoke] {len(table)} subjects, {cv['n_folds']} folds x {len(cv['seeds'])} seeds")
    for m in MODELS:
        v = cv["across_seeds"][m]
        print(f"[smoke]   {m:<3s} AUC {v['auc_mean']:.3f} +- {v['auc_sd']:.3f}")
    print(f"[smoke] delivered model: {cv['delivered_model']['model']}")
    print(f"[smoke] leakage guard recorded {cv['leakage_events']} fit events")
    print(f"[smoke] wrote {OUTPUT_DIR / 'report.md'}")
    print(f"[smoke] total {time.time() - t0:.1f}s")
    return 0


def null_auc_sd(n_pos: int, n_neg: int) -> float:
    """Sampling SD of the AUC under the null: sqrt((n1+n2+1) / (12*n1*n2)).

    At 6 vs 6 this is 0.1735, so a single dataset scatters over roughly [0.16, 0.84]
    with no leakage whatsoever. Every threshold below is derived from this number
    rather than picked by hand.
    """
    return float(np.sqrt((n_pos + n_neg + 1) / (12.0 * n_pos * n_neg)))


def summarise_null(per_model: Mapping[str, list], lo: float, hi: float, n_seeds: int,
                   n_subjects: int, alpha: float = 0.05) -> dict:
    """S2 verdict. Three checks, all with a stated error rate (see DEVIATIONS D1).

    1. the mean out-of-fold AUC sits inside the band -- this is the actual leakage test;
    2. the bootstrap CI of that mean does not sit entirely above 0.7;
    3. the count of single datasets above 0.7 is not significantly more than the null
       expects. The expected tail rate follows from `null_auc_sd`, so it is a one-sided
       binomial test rather than a hand-picked fraction. The lower tail is reported
       alongside it: noise scatters symmetrically, leakage does not.
    """
    from scipy.stats import binom, norm

    sd = null_auc_sd(n_subjects // 2, n_subjects - n_subjects // 2)
    p_tail = float(1.0 - norm.cdf((hi - 0.5) / sd))
    crit = int(binom.ppf(1.0 - alpha, n_seeds, p_tail))
    models = {}
    for m, vals in per_model.items():
        v = [x for x in vals if np.isfinite(x)]
        mean, ci_lo, ci_hi = mt.bootstrap_ci(v, n_boot=10000, seed=0)
        n_up = int(sum(x > hi for x in v))
        n_dn = int(sum(x < lo for x in v))
        p_binom = float(binom.sf(n_up - 1, len(v), p_tail))
        models[m] = {
            "mean_auc": mean, "ci_low": ci_lo, "ci_high": ci_hi,
            "n_above": n_up, "n_below": n_dn, "n_seeds": len(v),
            "frac_above_0_7": float(n_up / max(len(v), 1)),
            "binomial_p": p_binom, "per_seed": v,
            "pass": bool(lo <= mean <= hi and not (ci_lo > hi) and p_binom > alpha),
        }
    return {
        "n_seeds": n_seeds, "band": [lo, hi], "null_auc_sd": sd,
        "expected_tail_rate": p_tail, "critical_count": crit,
        "models": models, "pass": all(v["pass"] for v in models.values()),
    }


def summarise_sweep(by_model: Mapping[str, list], strengths: list, n_seeds: int,
                    n_subjects: int, final_min: float, min_trend: float) -> dict:
    """S3 verdict: monotone *trend*, not adjacent-pair monotonicity.

    With `n_seeds` datasets per point the SE of each point is null_auc_sd / sqrt(n_seeds)
    -- 0.100 at three seeds -- so demanding that adjacent points never dip by more than
    0.02 tests the noise, not the pipeline. Spearman rho between the injected strength and
    the AUC is the direct operationalisation of "AUC rises with s".
    """
    from scipy.stats import spearmanr

    se = null_auc_sd(n_subjects // 2, n_subjects - n_subjects // 2) / np.sqrt(max(n_seeds, 1))
    models = {}
    for m, vals in by_model.items():
        rho = float(spearmanr(strengths, vals).statistic)
        models[m] = {
            "auc_by_strength": [float(x) for x in vals],
            "trend_rho": rho,
            "monotonic": bool(rho >= min_trend),
            "final_ok": bool(vals[-1] > final_min),
            "max_adjacent_dip": float(max(
                [vals[i] - vals[i + 1] for i in range(len(vals) - 1)] or [0.0])),
        }
    return {"signal_strengths": strengths, "n_seeds": n_seeds, "per_point_se": float(se),
            "final_auc_min": final_min, "min_trend_rho": min_trend, "models": models}


def _quick_cv(table, cfg: Mapping, rules: Mapping, n_boot: int = 1000) -> dict:
    fast_rules = dict(rules)
    fast_rules["comparison"] = {**rules["comparison"], "n_boot": n_boot}
    return run_cv(table, cfg, fast_rules, with_ablation=False, run_probe=False)


def cmd_validate(cfg: Mapping, rules: Mapping) -> int:
    t0 = time.time()
    backends = _backends(cfg)
    backends.warm_up()
    v_cfg = cfg["validate"]
    out: dict = {}

    # ------------------------------------------------------------------ S2 null check
    nc = v_cfg["null_check"]
    lo, hi = nc["mean_auc_band"]
    per_model: dict[str, list[float]] = {m: [] for m in MODELS}
    for i in range(int(nc["n_seeds"])):
        _, table = _build(cfg, backends, {
            "seed": int(cfg["synth"]["seed"]) + 10_000 + i,
            "signal_strength": float(nc["signal_strength"]),
            "n_subjects": int(nc["n_subjects"]),
            "turns_per_subject": int(nc["turns_per_subject"]),
        })
        cv = _quick_cv(table, cfg, rules)
        for m in MODELS:
            per_model[m].append(cv["across_seeds"][m]["auc_mean"])
        print(f"[validate] null seed {i + 1}/{nc['n_seeds']}: "
              + " ".join(f"{m}={per_model[m][-1]:.2f}" for m in MODELS), flush=True)

    out["null_check"] = summarise_null(per_model, lo, hi, int(nc["n_seeds"]),
                                       int(nc["n_subjects"]), float(nc["alpha"]))
    (OUTPUT_DIR / "validate_null.json").parent.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "validate_null.json").write_text(
        json.dumps(out["null_check"], indent=2, default=str), encoding="utf-8")

    # ------------------------------------------------------------------ S3 sweep
    sw = v_cfg["sweep"]
    strengths = list(sw["signal_strengths"])
    by_model: dict[str, list[float]] = {m: [] for m in MODELS}
    for s in strengths:
        seed_vals: dict[str, list[float]] = {m: [] for m in MODELS}
        for j in range(int(sw["n_seeds"])):
            _, table = _build(cfg, backends, {
                "seed": int(cfg["synth"]["seed"]) + 20_000 + j,
                "signal_strength": float(s),
                "n_subjects": int(sw["n_subjects"]),
                "turns_per_subject": int(sw["turns_per_subject"]),
            })
            cv = _quick_cv(table, cfg, rules)
            for m in MODELS:
                seed_vals[m].append(cv["across_seeds"][m]["auc_mean"])
        for m in MODELS:
            by_model[m].append(float(np.nanmean(seed_vals[m])))
        print(f"[validate] sweep s={s}: "
              + " ".join(f"{m}={by_model[m][-1]:.2f}" for m in MODELS), flush=True)

    out["sweep"] = summarise_sweep(by_model, strengths, int(sw["n_seeds"]),
                                   int(sw["n_subjects"]), float(sw["final_auc_min"]),
                                   float(sw["min_trend_rho"]))
    sweep_models = out["sweep"]["models"]
    (OUTPUT_DIR / "validate_sweep.json").write_text(
        json.dumps(out["sweep"], indent=2, default=str), encoding="utf-8")

    # ------------------------------------------------------------------ examiner-leak control
    pc = v_cfg.get("examiner_leak_positive_control", {})
    if pc.get("enabled", False):
        base_acc, base_p, leak_acc, leak_p = [], [], [], []
        for j in range(int(pc["n_seeds"])):
            common = {"seed": int(cfg["synth"]["seed"]) + 30_000 + j,
                      "signal_strength": 1.0,
                      "n_subjects": int(sw["n_subjects"]),
                      "turns_per_subject": int(sw["turns_per_subject"])}
            for leak, accs, ps in ((0.0, base_acc, base_p),
                                   (float(pc["examiner_leak_strength"]), leak_acc, leak_p)):
                _, table = _build(cfg, backends, {**common, "examiner_leak_strength": leak})
                r = an.examiner_probe(table, n_permutations=2000, seed=0,
                                      n_folds=int(cfg["cv"]["outer"]["n_folds"]))
                accs.append(r["accuracy"])
                ps.append(r["permutation_p"])
        out["examiner_leak_positive_control"] = {
            "examiner_leak_strength": float(pc["examiner_leak_strength"]),
            "baseline_accuracy": float(np.mean(base_acc)),
            "baseline_p": float(np.mean(base_p)),
            "accuracy": float(np.mean(leak_acc)),
            "permutation_p": float(np.mean(leak_p)),
            "detected": bool(np.mean(leak_acc) > np.mean(base_acc) + 0.15),
        }
        print(f"[validate] examiner-leak control: baseline acc "
              f"{np.mean(base_acc):.2f} -> leaked {np.mean(leak_acc):.2f}", flush=True)

    (OUTPUT_DIR / "validate.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    _regenerate_report(out)

    ok = out["null_check"]["pass"]
    print(f"[validate] S2 null check: {'PASS' if ok else 'FAIL'}")
    for m, v in sweep_models.items():
        print(f"[validate] S3 {m}: trend_rho={v['trend_rho']:.2f} "
              f"final>{sw['final_auc_min']}={v['final_ok']} "
              f"curve={[round(x, 3) for x in v['auc_by_strength']]}")
    print(f"[validate] total {time.time() - t0:.1f}s")
    return 0


def _load_validate() -> dict | None:
    p = OUTPUT_DIR / "validate.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def _regenerate_report(validate: Mapping | None = None) -> None:
    if not STATE.exists():
        return
    with STATE.open("rb") as fh:
        st = pickle.load(fh)
    write_report(OUTPUT_DIR / "report.md", st["table"], st["cv"], st["criterion"],
                 st["group"], st["cfg"], st["rules"], st["versions"],
                 validate=validate if validate is not None else _load_validate())


def cmd_checklist(cfg: Mapping, rules: Mapping) -> int:
    if not STATE.exists():
        print("[checklist] no outputs/state.pkl -- run `make smoke` first", file=sys.stderr)
        return 1
    with STATE.open("rb") as fh:
        st = pickle.load(fh)
    path = write_checklist(OUTPUT_DIR / "checklist_report.md",
                           evaluate_items(st["table"], st["cv"], st["criterion"],
                                          st["cfg"], st["versions"]))
    print(f"[checklist] wrote {path}")
    return 0


def cmd_intake(cfg: Mapping, rules: Mapping) -> int:
    """Load a real cohort through the adapter and write the P0-0A/P0-1/P0-2 audit report.

    Transcripts only -- no audio is decoded, so this is fast and can be run before any
    decision about feature extraction is made.
    """
    from ..realdata import discover_cases, load_transcript

    rd = cfg["realdata"]
    root = Path(rd["root"])
    if not root.exists():
        print(f"[intake] {root} not found -- set realdata.root in configs/pipeline.yaml",
              file=sys.stderr)
        return 1
    cases = discover_cases(root, rd.get("exclude_cases", []), cohort=rd.get("cohort"))
    if not cases:
        print(f"[intake] no usable case folders under {root}", file=sys.stderr)
        return 1
    reports = []
    for c in cases:
        _, rep = load_transcript(c.transcript, cfg)
        reports.append(rep)
        print(f"[intake] {rep.case_id}: {rep.n_raw} -> {rep.n_kept} segments "
              f"({100 * rep.kept_ratio:.0f}%), {rep.n_qa_pairs} QA pairs "
              f"({rep.n_open_qa_pairs} open)", flush=True)
    path = write_intake_report(OUTPUT_DIR / "intake_report.md", reports, cfg,
                               excluded=list(rd.get("exclude_cases", [])))
    print(f"[intake] {len(reports)} cases, cohort '{rd['cohort']}'")
    print(f"[intake] wrote {path}")
    return 0


def cmd_align_check(cfg: Mapping, rules: Mapping) -> int:
    """Run MFA on the real cohort and measure what can be measured (audit P1-1)."""
    from ..align.evaluate import cross_alignment_agreement, export_spotcheck, read_dictionary_words
    from ..align.mfa import MFAAligner
    from ..contracts import Speaker
    from ..eval.alignment import align_case, segment_for_alignment, write_alignment_report
    from ..realdata import discover_cases, load_audio, load_transcript

    rd, mfa = cfg["realdata"], cfg["models"]["mfa"]
    limit = int(os.environ.get("ALIGN_CASES", "0")) or None
    cases = discover_cases(Path(rd["root"]), rd.get("exclude_cases", []),
                           cohort=rd.get("cohort"))[:limit]
    if not cases:
        print("[align] no cases found", file=sys.stderr)
        return 1

    backends = _backends(cfg)
    sr = int(cfg["synth"]["sample_rate"])
    dict_dir = Path(mfa["dictionary_dir"])
    lexicon = None
    lex_path = dict_dir / f"{mfa['dictionary']}.dict"
    if lex_path.exists():
        lexicon = read_dictionary_words(lex_path)

    def make(dictionary: str) -> MFAAligner:
        return MFAAligner(acoustic_model=mfa["acoustic_model"], dictionary=dictionary,
                          num_jobs=int(mfa["num_jobs"]), binary=mfa.get("binary") or None,
                          g2p_model=mfa.get("g2p_model") or None, timeout_s=5400,
                          beam=int(mfa.get("beam") or 0) or None,
                          fine_tune=bool(mfa.get("fine_tune", False)))

    # Cached, so re-running the report after `extract-real` costs almost nothing.
    from ..align import CachedAligner

    from ..align.mfa import SYLLABIFY_VERSION

    identity = {k: mfa.get(k) for k in
                ("acoustic_model", "dictionary", "g2p_model", "beam", "fine_tune")}
    identity["syllabify_version"] = SYLLABIFY_VERSION
    primary = CachedAligner(make(mfa["dictionary"]), CACHE_DIR, identity)
    if not primary.inner.available():
        print("[align] the mfa executable was not found; set models.mfa.binary or MFA_BIN",
              file=sys.stderr)
        return 1
    secondary = make(mfa["cross_check_dictionary"]) if mfa.get("cross_check_dictionary") else None

    results, cross_all = [], {"n": 0, "n_turns": 0, "median": [], "p90": [], "within": []}
    spot_dir = OUTPUT_DIR / "spotcheck"
    for c in cases:
        turns, _ = load_transcript(c.transcript, cfg, case_id=c.case_id)
        kids = [t for t in turns if t.speaker is Speaker.CHILD]
        words = segment_for_alignment(kids, backends)
        audio = load_audio(c.audio, sr, CACHE_DIR)
        res = align_case(c.case_id, audio, sr, kids, words, primary, lexicon)
        results.append(res)
        print(f"[align] {c.case_id}: {res.quality.n_turns_aligned}/"
              f"{res.quality.n_turns_attempted} turns, "
              f"{res.quality.n_syllables}/{res.quality.n_expected_syllables} syllables, "
              f"OOV {100 * res.quality.oov_token_rate:.1f}%, {res.seconds:.0f}s", flush=True)
        if secondary is not None:
            other = secondary.align(audio, sr, [t for t in kids if words.get(t.turn_id)], words)
            agree = cross_alignment_agreement(res.spans, other)
            if agree["n"]:
                cross_all["n"] += agree["n"]
                cross_all["n_turns"] += agree["n_turns"]
                cross_all["median"].append(agree["median_ms"])
                cross_all["p90"].append(agree["p90_ms"])
                cross_all["within"].append(agree["within_20ms"])
                print(f"[align]   cross-dictionary: median {agree['median_ms']:.1f} ms, "
                      f"{100 * agree['within_20ms']:.0f}% within 20 ms", flush=True)
        export_spotcheck(spot_dir, c.case_id, audio, sr, kids, res.spans,
                         n=int(mfa["spotcheck_per_case"]), seed=0)

    cross = None
    if cross_all["median"]:
        cross = {"n": cross_all["n"], "n_turns": cross_all["n_turns"],
                 "median_ms": float(np.median(cross_all["median"])),
                 "p90_ms": float(np.median(cross_all["p90"])),
                 "within_20ms": float(np.mean(cross_all["within"]))}
    path = write_alignment_report(OUTPUT_DIR / "alignment_report.md", results, cfg,
                                  cross=cross, spotcheck_dir=spot_dir)
    print(f"[align] wrote {path}")
    return 0


def cmd_extract_real(cfg: Mapping, rules: Mapping) -> int:
    """Run the full B/C/E extraction on the real cohort and report the distributions.

    No model is trained here. The cohort has a single group, so classification, group
    comparison and criterion validity are all impossible; what this answers is whether the
    extractors survive real recordings and whether their output is physically plausible.
    """
    from ..align import aligner_from_config
    from ..eval.reference import write_reference_report
    from ..realdata import load_cohort

    # Real recordings get the `real` profile (CREPE full + the mpnet sentence encoder),
    # not the deliberately-lightweight `smoke` one. DEVIATIONS D4 says so; the wiring did
    # not, and the first real run silently used the smoke models.
    cfg = {**cfg, "models": {**cfg["models"],
                             "profile": os.environ.get("ASDPIPE_PROFILE", "real")}}
    # ASDPIPE_COHORT selects which arm to extract. The report filename follows it so a run
    # on one cohort cannot overwrite another's -- with two arms on disk that is now a real
    # way to lose a 28-minute run's output.
    cohort = os.environ.get("ASDPIPE_COHORT") or cfg["realdata"].get("cohort", "TD")
    cfg = {**cfg, "realdata": {**cfg["realdata"], "cohort": cohort}}
    rd = cfg["realdata"]
    root = Path(rd["root"])
    if not root.exists():
        print(f"[real] {root} not found", file=sys.stderr)
        return 1

    backends = _backends(cfg)
    print(f"[real] profile={cfg['models']['profile']} "
          f"sbert={backends.sbert_id} crepe={backends.crepe_model}", flush=True)
    backends.warm_up()
    aligner = aligner_from_config(cfg, CACHE_DIR)
    if not aligner.available():
        print("[real] no aligner available; check models.mfa.binary", file=sys.stderr)
        return 1

    t = time.time()
    corpus, reports = load_cohort(root, cfg, cache_dir=CACHE_DIR)
    t_load = time.time() - t
    print(f"[real] loaded {len(corpus)} subjects in {t_load:.1f}s "
          f"({sum(r.session_s for r in reports) / 60:.0f} min of audio)", flush=True)

    # Source normalization, identical for every cohort (health check 0930 P1-P4). Applied
    # here rather than inside extract_corpus so the synthetic path, which has one convention
    # and ground-truth boundaries, is untouched.
    if cfg["realdata"].get("normalize", {}).get("enabled", False):
        from ..normalize import normalize_corpus

        corpus, norm = normalize_corpus(corpus, cfg)
        print(f"[real] normalized: {sum(r.n_in for r in norm)} -> {sum(r.n_out for r in norm)} "
              f"turns, {sum(r.n_trimmed for r in norm)} trimmed "
              f"({sum(r.seconds_trimmed for r in norm) / 60:.1f} min of silence removed), "
              f"{sum(r.n_no_voice_found for r in norm)} kept untrimmed", flush=True)

    t = time.time()
    rows = extract_corpus(corpus, cfg, backends, aligner=aligner)
    t_extract = time.time() - t
    table = build_table(rows)
    print(f"[real] extracted {len(table)} subjects in {t_extract:.1f}s", flush=True)

    notes = []
    if any(r.label_is_placeholder for r in reports):
        notes.append("**標籤是 placeholder**：`load_cohort` 沒有拿到標籤對照表，"
                     "每位受試者的 `label` 只是佔位值，任何用到它的下游分析都無效。")
    path = write_reference_report(
        OUTPUT_DIR / f"{cohort.lower()}_reference.md", table, cfg,
        timings={"load": t_load, "extract": t_extract}, extra_notes=notes)
    with (OUTPUT_DIR / "real_state.pkl").open("wb") as fh:
        pickle.dump({"table": table, "cfg": cfg, "reports": reports}, fh)
    print(f"[real] wrote {path}")
    return 0


def cmd_clean(cfg: Mapping, rules: Mapping) -> int:
    for p in (CACHE_DIR, OUTPUT_DIR):
        if p.exists():
            shutil.rmtree(p)
            print(f"[clean] removed {p}")
    return 0


def cmd_audio_audit(cfg: Mapping, rules: Mapping) -> int:
    """Recording-condition inventory over every cohort, transcript or not (audit P0-0A).

    Exists because the ASD arm arrived as raw video with no transcripts. Most of the
    pipeline cannot run on it, but the part that decides whether the acoustic features will
    ever be comparable across groups can -- and that part is worth checking before anyone
    spends weeks transcribing.
    """
    from ..eval.audio_audit import audit_cohort, write_audio_audit_report

    cfg = {**cfg, "models": {**cfg["models"],
                             "profile": os.environ.get("ASDPIPE_PROFILE", "real")}}
    rd = cfg["realdata"]
    root = Path(rd["root"])
    if not root.exists():
        print(f"[audit] {root} not found", file=sys.stderr)
        return 1
    cohorts = [d.name for d in sorted(root.iterdir()) if d.is_dir()] or [rd.get("cohort", "TD")]
    limit = int(os.environ.get("ASDPIPE_AUDIT_LIMIT", "0")) or None

    backends = _backends(cfg)
    print(f"[audit] cohorts={cohorts} profile={cfg['models']['profile']}", flush=True)
    audits = []
    for c in cohorts:
        t0 = time.time()
        got = audit_cohort(root, c, cfg, backends, CACHE_DIR,
                           exclude=rd.get("exclude_cases", []), limit=limit)
        audits.extend(got)
        print(f"[audit] {c}: {sum(1 for a in got if a.ok)}/{len(got)} usable "
              f"in {time.time() - t0:.0f}s", flush=True)

    out = write_audio_audit_report(OUTPUT_DIR / "audio_condition_audit.md", audits, cfg)
    print(f"[audit] wrote {out}")
    return 0


def cmd_convention_gate(cfg: Mapping, rules: Mapping) -> int:
    """decision_rules.yaml convention_invariance_gate, on TD only. No label is read."""
    from ..align import aligner_from_config
    from ..eval.convention_gate import run_gate, write_gate_report

    cfg = {**cfg, "models": {**cfg["models"],
                             "profile": os.environ.get("ASDPIPE_PROFILE", "real")}}
    backends = _backends(cfg)
    aligner = aligner_from_config(cfg, CACHE_DIR)
    g = rules["convention_invariance_gate"]
    t0 = time.time()
    res, fmt, meta = run_gate(cfg, rules, backends, aligner, CACHE_DIR,
                              log=lambda m: print(m, flush=True))
    out = write_gate_report(OUTPUT_DIR / "convention_gate.md", res, fmt, meta,
                            float(g["threshold_sd"]), float(g["format_check"]["threshold_sd"]))
    print(f"[gate] done in {time.time() - t0:.0f}s -> {out}")
    return 0


def _review_cfg(cfg: Mapping) -> Mapping:
    return {**cfg, "models": {**cfg["models"],
                              "profile": os.environ.get("ASDPIPE_PROFILE", "real")}}


def _mil_cfg(cfg: Mapping) -> Mapping:
    """The config of ``mil-checks`` and ``mil-run``: models.profile is "real" (CREPE "full"),
    set here and not read from the environment.

    The committed profile is smoke, whose CREPE is "tiny", and ``_review_cfg`` took the profile
    from ASDPIPE_PROFILE: the environment could swap the pitch tracker of either command outside
    the freeze, and nothing recorded it. The variable may still be set, but only to "real";
    anything else is refused (``RunRefused``).
    """
    from ..mil.run import RunRefused

    asked = os.environ.get("ASDPIPE_PROFILE")
    if asked is not None and asked != "real":
        raise RunRefused(f"ASDPIPE_PROFILE is '{asked}', but the mil commands run with the real "
                         "profile only (CREPE 'full'); unset it or set it to 'real'")
    return {**cfg, "models": {**cfg["models"], "profile": "real"}}


def cmd_review_prepare(cfg: Mapping, rules: Mapping) -> int:
    """Build outputs/review/ for the health check's step 5. ASDPIPE_REVIEW_FORCE=1 sets an
    existing package aside (moved, not deleted) and builds a new one."""
    from ..review.prepare import prepare_package

    force = os.environ.get("ASDPIPE_REVIEW_FORCE") == "1"
    prepare_package(_review_cfg(cfg), OUTPUT_DIR / "review", CACHE_DIR, force=force,
                    log=lambda m: print(m, flush=True))
    return 0


def cmd_review(cfg: Mapping, rules: Mapping) -> int:
    """Serve the review package on 127.0.0.1, building it first if it does not exist."""
    from ..review.server import serve

    if not (OUTPUT_DIR / "review" / "items.json").exists():
        cmd_review_prepare(cfg, rules)
    serve(OUTPUT_DIR / "review", port=int(os.environ.get("ASDPIPE_REVIEW_PORT", "8765")),
          open_browser=os.environ.get("ASDPIPE_REVIEW_NO_BROWSER") != "1")
    return 0


def cmd_review_score(cfg: Mapping, rules: Mapping) -> int:
    from ..review.score import score_package

    out, _ = score_package(OUTPUT_DIR / "review")
    print(f"[review] wrote {out}")
    return 0


def cmd_vocal_calibrate(cfg: Mapping, rules: Mapping) -> int:
    """Fix the untranscribed-vocalisation rule on TD alone (features/vocal.py).

    Reads no label and no ASD session. Writes outputs/vocal_calibration.md with the
    margin, floor and response window to copy into pipeline.yaml features.vocal.
    """
    from ..contracts import Speaker
    from ..features import b_voice
    from ..features.e_text import pair_turns
    from ..features.vocal import (attribute, calibrate, centroid, untranscribed_segments,
                                  vocal_features)
    from ..normalize import normalize_corpus
    from ..realdata import load_cohort

    cfg = _review_cfg(cfg)
    rd = cfg["realdata"]
    cfg = {**cfg, "realdata": {**rd, "cohort": "TD"}}
    v_cfg = cfg["features"]["vocal"]
    min_w = int(v_cfg["min_enroll_windows"])
    backends = _backends(cfg)
    raw, _ = load_cohort(rd["root"], cfg, cache_dir=CACHE_DIR)
    corpus, _ = normalize_corpus(raw, cfg)
    sessions = [(s.audio, s.sample_rate, list(s.session.turns)) for s in corpus.subjects]
    windows_of = lambda a, sr, turns, sp: b_voice.ecapa_windows(a, sr, turns, cfg, speaker=sp)
    t0 = time.time()
    res = calibrate(sessions, windows_of, backends.embed_windows, backends.embed_segments,
                    v_cfg, min_w)

    # response window: how long TD children take to start answering a question
    lat = np.asarray([ch.t_start - ex.t_end for _, _, turns in sessions
                      for ex, ch in pair_turns(sorted(turns, key=lambda x: x.t_start))
                      if ex.is_question], dtype=float)
    lat = lat[np.isfinite(lat) & (lat >= 0)]
    window = float(np.percentile(lat, 95))

    # the calibrated rule applied to whole TD sessions, enrolled on all transcribed speech
    rule = {**v_cfg, "margin": res["margin"], "floor": res["floor"], "response_window_s": window}
    per = []
    for audio, sr, turns in sessions:
        segs = untranscribed_segments(audio, sr, turns, rule)
        cc = centroid(backends.embed_windows(windows_of(audio, sr, turns, Speaker.CHILD)), min_w)
        ec = centroid(backends.embed_windows(windows_of(audio, sr, turns, Speaker.EXAMINER)), min_w)
        is_child = None
        if cc is not None and ec is not None:
            clips = [audio[int(a * sr):int(b * sr)] for a, b in segs]
            emb = backends.embed_segments(clips)
            is_child = list(attribute(emb, cc, ec, res["margin"], res["floor"])[0]) if len(segs) else []
        out = vocal_features(turns, segs, is_child, rule,
                             int(cfg["features"]["e_stream"]["min_pairs_per_subject"]))
        per.append({**out.scalars, **out.quality,
                    "untranscribed_total_s": float(sum(b - a for a, b in segs))})

    L = ["# 未轉錄發聲偵測：門檻校準（只用一般發展組）", "",
         f"產生時間：{time.strftime('%Y-%m-%d %H:%M')}｜TD {len(sessions)} 場，不讀任何標籤或自閉症組資料", "",
         "## 建議寫入 pipeline.yaml features.vocal 的值", "",
         "| 參數 | 值 | 規則 |", "|---|---|---|",
         f"| margin | {res['margin']:.4f} | 施測者片段被當成小孩的比例 ≤ 5% |",
         f"| floor | {res['floor']:.4f} | 小孩片段與自己聲紋相似度的第 5 百分位 |",
         f"| response_window_s | {window:.2f} | TD 小孩回答問題的延遲第 95 百分位（{len(lat)} 組問答） |", "",
         "## 判斷準不準：已知是誰說的片段（每場一半句子建聲紋、另一半切成 0.3–2 秒的片段來測）", "",
         "| 小孩聲紋用多少秒建立 | 小孩片段被正確認出 | 施測者片段被誤認成小孩 | 片段數（小孩／施測者） |",
         "|---|---|---|---|"]
    for budget, r in res["table"].items():
        name = "全部（一半句子）" if budget is None else f"只用 {budget:.0f} 秒"
        L.append(f"| {name} | {100 * r['child_recall']:.1f}% | {100 * r['exam_false_accept']:.1f}% | "
                 f"{r['n_child']}／{r['n_exam']} |")
    L += ["", "依片段長度（聲紋用全部建立）：", "", "| 片段長度 | 小孩片段被正確認出 | 片段數 |", "|---|---|---|"]
    for k, (rate, n) in res["table"][None]["recall_by_duration"].items():
        L.append(f"| {k} | {100 * rate:.1f}% | {n} |")
    ok = [r for r in per if r["vocal_attribution_available"]]
    q = lambda key: np.asarray([r.get(key, np.nan) for r in ok], dtype=float)
    L += ["", "## 套到整場（一般發展組）", "",
          "| 指標 | 中位數 | 範圍 |", "|---|---|---|"]
    for key, label in (("untranscribed_total_s", "沒轉成文字的有聲片段（秒）"),
                       ("vocal_untranscribed_child_s", "其中判斷為小孩的（秒）"),
                       ("e_vocal_all_seconds_per_min", "總發聲量（秒／分鐘）"),
                       ("e_vocal_untranscribed_share", "聽不清楚發聲的比例"),
                       ("e_vocal_response_rate", "問完有回應的比例")):
        v = q(key)
        v = v[np.isfinite(v)]
        L.append(f"| {label} | {np.median(v):.3f} | {v.min():.3f}–{v.max():.3f} |" if v.size
                 else f"| {label} | — | — |")
    L += ["", f"聲紋建立成功：{len(ok)}／{len(per)} 場。"]
    out_path = OUTPUT_DIR / "vocal_calibration.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[vocal] margin={res['margin']:.4f} floor={res['floor']:.4f} window={window:.2f}s "
          f"in {time.time() - t0:.0f}s -> {out_path}")
    return 0


def cmd_auto_checks(cfg: Mapping, rules: Mapping) -> int:
    """Health-check step 5, the part that needs no listener (eval/auto_checks.py)."""
    from ..eval.auto_checks import run_auto_checks

    run_auto_checks(_review_cfg(cfg), CACHE_DIR, OUTPUT_DIR / "auto_checks.md",
                    log=lambda m: print(m, flush=True))
    return 0


def cmd_mil_checks(cfg: Mapping, rules: Mapping) -> int:
    """Research plan 3.4 before the freeze, on TD only: no label, no ASD session.

    Writes outputs/mil_checks.md and .json (asdpipe.mil.run.mil_checks). The JSON's
    proposed_outcome is what goes into decision_rules.yaml mil_analysis.outcome, to be
    committed with the freeze before `mil-run` can start. The private ID maps are read from the
    git root, the project directory's parent.
    """
    from ..mil import run as mil

    try:
        cfg = _mil_cfg(cfg)                    # the real model profile, whatever the environment
    except mil.RunRefused as e:
        print(f"[mil-checks] refused: {e}", file=sys.stderr)
        return 1
    backends = _backends(cfg)
    t0 = time.time()
    try:
        out = mil.mil_checks(cfg, rules, project_dir=REPO_ROOT, id_root=REPO_ROOT.parent,
                             out_dir=OUTPUT_DIR, tools=mil.real_tools(cfg, backends),
                             cache_dir=CACHE_DIR, log=lambda m: print(m, flush=True))
    except mil.RunRefused as e:
        print(f"[mil-checks] refused: {e}", file=sys.stderr)
        return 1
    print(f"[mil-checks] proposed outcome.excluded: {out['proposed_outcome']['excluded']}")
    print(f"[mil-checks] done in {time.time() - t0:.0f}s -> {OUTPUT_DIR / 'mil_checks.md'}")
    return 0


def cmd_mil_run(cfg: Mapping, rules: Mapping) -> int:
    """The one labelled run (research plan 3.5-3.8), from the freeze commit.

    Refuses, with exit code 1 and nothing computed, when ASDPIPE_PROFILE names another profile
    than "real", when the research plan, configs/ or src/ differ from HEAD, while
    decision_rules.yaml mil_analysis.outcome has no 'excluded' list or records checks run on
    other code (code_hash), and unless CREPE runs on CUDA. Every F0 track and feature of both
    groups is computed in this process (ruling R33): the F0 cache is a new directory under
    cache/mil_run and the features pickle is neither read nor written.
    Writes outputs/mil_results_<label>.json, mil_report_<label>.md and the attention excerpts
    (asdpipe.mil.run.mil_run).
    """
    from ..align import aligner_from_config
    from ..mil import run as mil

    try:                       # checked again inside mil_run; here before any model is loaded
        cfg = _mil_cfg(cfg)
        mil.freeze_guard(REPO_ROOT)
        mil.outcome_exclusions(rules)
        mil.check_code_hash(REPO_ROOT, rules)
    except mil.RunRefused as e:
        print(f"[mil-run] refused: {e}", file=sys.stderr)
        return 1
    backends = _backends(cfg)
    aligner = aligner_from_config(cfg, CACHE_DIR)
    if not aligner.available():
        print("[mil-run] no aligner available; check models.mfa.binary", file=sys.stderr)
        return 1
    t0 = time.time()
    try:
        tools = mil.real_tools(cfg, backends, aligner, fresh_cache=CACHE_DIR / "mil_run")
        res = mil.mil_run(cfg, rules, project_dir=REPO_ROOT, id_root=REPO_ROOT.parent,
                          out_dir=OUTPUT_DIR, tools=tools, cache_dir=CACHE_DIR,
                          log=lambda m: print(m, flush=True))
    except mil.RunRefused as e:
        print(f"[mil-run] refused: {e}", file=sys.stderr)
        return 1
    print(f"[mil-run] {res['run_label']} at commit {res['head'][:12]}: done in "
          f"{(time.time() - t0) / 60:.0f} min -> {OUTPUT_DIR / res['files']['report']}")
    return 0


def _cap_gpu_memory() -> None:
    """ASDPIPE_GPU_MEM_FRACTION caps this process's share of GPU memory, so a run shares
    the card with someone else's training: going over raises here, not over there."""
    frac = os.environ.get("ASDPIPE_GPU_MEM_FRACTION")
    if not frac:
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(frac), 0)
        print(f"[gpu] this process limited to {float(frac):.0%} of GPU memory", flush=True)


def main(argv: list[str] | None = None) -> int:
    _cap_gpu_memory()
    argv = list(sys.argv[1:] if argv is None else argv)
    target = argv[0] if argv else "smoke"
    cfg = load_pipeline_config()
    rules = load_decision_rules()
    handlers = {"smoke": cmd_smoke, "validate": cmd_validate,
                "checklist": cmd_checklist, "intake": cmd_intake,
                "align-check": cmd_align_check, "extract-real": cmd_extract_real,
                "audio-audit": cmd_audio_audit, "convention-gate": cmd_convention_gate,
                "review-prepare": cmd_review_prepare, "review": cmd_review,
                "review-score": cmd_review_score, "vocal-calibrate": cmd_vocal_calibrate,
                "auto-checks": cmd_auto_checks,
                "mil-checks": cmd_mil_checks, "mil-run": cmd_mil_run,
                "clean": cmd_clean}
    if target not in handlers:
        print(f"unknown target {target!r}; expected one of {sorted(handlers)}", file=sys.stderr)
        return 2
    return handlers[target](cfg, rules)


if __name__ == "__main__":
    raise SystemExit(main())
