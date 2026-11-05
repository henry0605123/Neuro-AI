"""The two commands that take the segment-attention model to the real cohorts (research plan
3.4-3.8): ``mil-checks`` on TD before the freeze, and ``mil-run`` once after it.

**Two commands either side of the freeze.** ``mil_checks`` reads TD only and no label (plan
section 5, rule 1: nothing about the ASD children is computed before the freeze). It measures
what plan 3.4 asks for and proposes, by the frozen rules alone, which measures leave the model;
that proposal is copied into ``decision_rules.yaml`` (``mil_analysis.outcome``) and committed
with the plan and the code. ``mil_run`` starts only from that commit: ``freeze_guard`` refuses a
working tree whose plan, configs or source differ from HEAD, and ``outcome_exclusions`` refuses
rules without the recorded outcome. What decides the labelled run is therefore in git, so the
measure-level exclusions are read from the committed rules and never from ``outputs/`` (ruling
R2). The outcome carries the SHA-256 of the source and settings the checks ran on
(``code_hash``, ruling R32), which ``check_code_hash`` recomputes: checks decided on other code
than the run's own are refused. What is measured per session, the F0 check, is computed again at
run time for every session, TD and ASD alike: an ASD session has no check until then.

**One process, one device, for both groups** (ruling R33). The labelled run computes every F0
track and feature of TD and ASD afresh in its own process, on CUDA, and records the CREPE
settings and the environment with its results (``real_tools(fresh_cache=...)``,
``environment_record``). It writes its results as each variant finishes (``analyse``'s
``on_progress``) and to files named by the run's label, which it will not overwrite
(``output_files``); it runs only on the ASD samples the plan pre-registers
(``check_asd_sample``).

**Folder names never travel.** A case folder is named after the child. ``load_arm`` is the only
code that sees one: it looks the folder up in the private ID maps at the git root, renames the
record to its pseudonym before anything else is done with it, and keeps the folder, and anything
read from the transcript, out of any error it raises: such an error gives the case's cohort,
position and pseudonym, the failure's type and counts, never a message or a speaker label. From
there on every id, log line, error message and output holds pseudonyms only, and the children
are put in pseudonym order, so a folder name decides nothing, not even
which fold a child lands in. As a last line of defence the writers refuse an id that does not
look like a pseudonym (printable ASCII; the names are Chinese) instead of writing it. The one
output that holds transcript text, the attention excerpts, stays in ``outputs/`` and is never
printed.

**The variants** (plan 3.5-3.8) are all cut from the bags the run built, after the checks took
their measures out (ruling R12): a variant whose point is gone ("no_E5" once E5 was excluded) is
reported as not run, with the reason, never as a crash or as a copy of another variant. Only
``primary`` and ``transcript_only`` carry the permutation test. The second becomes the main
result if the clinician reports different recording set-ups for the two years
(``mil_analysis.recording_condition``), so the final model, its gates and its attention are
reported for both, and that answer needs no second run.

**The instruments are arguments.** ``Tools`` holds the word segmenter, the split-half and F0
measurements and the feature extraction as plain functions. ``real_tools`` builds them from
``Backends`` and the forced aligner for the CLI; a test passes stand-ins with known answers, so
both commands run end to end on synthetic cohorts without a model.
"""

from __future__ import annotations

import csv
import dataclasses
import gc
import hashlib
import json
import math
import operator
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import yaml

from ..contracts import Corpus, Label, Speaker, SubjectRecord, SubjectRow, Turn
from ..eval.auto_checks import SEEN_BEFORE
from ..eval.convention_gate import TRANSFORMS, GateResult
from ..extract import extract_corpus
from ..features.common import child_turns
from ..normalize import normalize_corpus
from ..realdata import EXCLUDED_ID_MAP, discover_cases, load_subject
from ..run import confirmatory_exclusions
from .bags import Bags, bags_from_arrays
from .checks import (SPLIT_HALF_MEASURES, b7_a11_halves, f0_group_rule, f0_session_pass,
                     f0_three_way, segment_gate, split_half_r)
from .evaluate import CVSettings, evaluate
from .model import MILConfig, fit_predict
from .segments import (CHILD_MEASURES, PITCH_MEASURES, SEGMENT_MEASURES, STREAMS,
                       segment_measures, segment_spans, turns_in_span)

__all__ = ["RunRefused", "Tools", "WordMemo", "Arm", "Child", "Prepared", "ID_MAPS",
           "ALL_MEASURES", "ASD_SAMPLES", "EXPOSED_ASD", "FULL_ASD", "FROZEN_PATHS", "pseudonyms",
           "lookup", "freeze_guard", "outcome_exclusions", "code_hash", "check_code_hash",
           "real_tools", "child_values", "load_arm", "checks_verdict", "mil_checks",
           "write_checks_report", "prepare_children", "build_bags", "analyse", "run_label",
           "check_asd_sample", "output_files", "id_map_hashes", "environment_record", "mil_run",
           "write_mil_report", "write_attention_excerpts"]

# The private ID maps at the git root: per cohort, the file and the column holding the id.
ID_MAPS: Mapping[str, tuple[str, str]] = {"TD": ("_private_id_map.csv", "pid"),
                                          "ASD": ("_private_id_map_asd.csv", "uid")}
ALL_MEASURES = SEGMENT_MEASURES + CHILD_MEASURES
E5 = "e_e5_talk_share"
FROZEN_PATHS = ("research_plan_1030.md", "configs", "src")     # ruling R3
CHECKS_SOURCE = "outputs/mil_checks.md"                        # ruling R27: outcome.source
FULL_ASD = 20       # research plan 二: twenty ASD children; the result is final with all of them
EXPOSED_ASD = 6     # DEVIATIONS D10: the six ASD sessions whose data were seen on 2026-09-12
# The two ASD samples plan 3.8 pre-registers: ASD sessions -> how many of them are unexposed. The
# preliminary run has the ten transcribed so far (6 exposed, 4 unexposed), the formal run all
# twenty (14 unexposed).
ASD_SAMPLES = {10: 4, FULL_ASD: 14}
GATE_OPEN = 0.5     # a gate above this counts as open in the stability share (plan 3.7)
FINAL_VARIANTS = ("primary", "transcript_only")

# What a pseudonym may look like: printable ASCII without spaces (P01, A01, syn-asd-000). The
# children's names are Chinese, so a name that slipped into an id cannot pass.
_PSEUDONYM = re.compile(r"[!-~]{1,40}")
_FLAT = 1e-12       # attention weights this close together weigh a child's segments alike

_VARIANT_TEXT = {"primary": "通過檢查的全部量測", "transcript_only": "只用逐字稿（不用孩子層級量測）",
                 "no_E5": "拿掉說話量 E5", "E5_only": "只用說話量 E5", "drop_E": "拿掉 E 流",
                 "drop_C": "拿掉 C 流", "drop_B": "拿掉 B 流",
                 "unexposed_only": "只用 TD 和未曝光的 ASD"}


class RunRefused(RuntimeError):
    """The labelled run cannot start: the freeze commit or the recorded checks are missing.
    The message says what to do, and nothing has been computed."""


# ------------------------------------------------------------------------------ pseudonyms
def _read_id_map(repo_root: Path, cohort: str) -> dict[str, str]:
    """One private map, session -> id; empty if the file is not there. An error names the file,
    a column or a line, never a value: in a malformed row any cell may hold a name."""
    filename, column = ID_MAPS[cohort]
    path = Path(repo_root) / filename
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as fh:       # a BOM is not data
        reader = csv.DictReader(fh)
        if not {column, "session"} <= set(reader.fieldnames or ()):
            raise ValueError(f"{filename} needs the columns '{column}' and 'session' on its "
                             "first line")
        for line, row in enumerate(reader, start=2):
            session = (row.get("session") or "").strip()
            pid = (row.get(column) or "").strip()
            if not session:
                raise ValueError(f"{filename} line {line}: no session")
            if not _PSEUDONYM.fullmatch(pid):
                raise ValueError(f"{filename} line {line}: the '{column}' column does not hold "
                                 "a pseudonym")
            if out.setdefault(session, pid) != pid:
                raise ValueError(f"{filename} line {line}: a session listed earlier with another "
                                 f"{column}")
    return out


def pseudonyms(repo_root: Path) -> dict[str, str]:
    """Session (case folder name) -> pseudonymous id, from ``_private_id_map.csv`` (TD, column
    ``pid``) and ``_private_id_map_asd.csv`` (ASD, column ``uid``) in ``repo_root``.

    A map that is not there adds nothing, so a session it would have named fails in ``lookup``.
    A session listed in both maps with two ids, or one id given to two sessions, is refused:
    one child is one unit (R2). Nothing read here is ever printed.
    """
    merged: dict[str, str] = {}
    holder: dict[str, str] = {}                   # id -> the session that has it
    for cohort in ID_MAPS:
        for session, pid in _read_id_map(repo_root, cohort).items():
            if merged.setdefault(session, pid) != pid:
                raise ValueError(f"a session is listed in both ID maps, as {merged[session]} "
                                 f"and as {pid}")
            if holder.setdefault(pid, session) != session:
                raise ValueError(f"the id {pid} is given to two sessions")
    return merged


def lookup(mapping: Mapping[str, str], session: str, cohort: str, index: int) -> str:
    """The pseudonym of ``session``, the ``index``-th case (counting from 1) of ``cohort``.

    A session the maps do not list raises ``KeyError`` naming the cohort and the position only,
    since the session is a child's name. No exception is chained to it, so no traceback shows it.
    """
    pid = mapping.get(session)
    if pid is None:
        raise KeyError(f"{cohort} case #{index} has no pseudonym")
    return pid


# ------------------------------------------------------------------------------ the freeze
def _git(project_dir: Path, *args: str) -> str:
    """A read-only git command in ``project_dir``; the run is refused if git cannot answer."""
    try:
        proc = subprocess.run(["git", "-C", str(project_dir), *args], capture_output=True,
                              text=True, encoding="utf-8", errors="replace")
    except OSError as e:
        raise RunRefused(f"git could not be run ({type(e).__name__}: {e}), so the freeze cannot "
                         "be checked") from None
    if proc.returncode != 0:
        raise RunRefused(f"`git {args[0]}` failed, so the freeze cannot be checked: "
                         f"{proc.stderr.strip()[-500:]}")
    return proc.stdout


def freeze_guard(project_dir: Path) -> str:
    """Refuse unless the research plan, ``configs/`` and all of ``src/`` are committed; return
    the HEAD commit the run will be reported against (ruling R3).

    Plan section 5, rule 1 in code: the labelled run happens once, from frozen code and frozen
    settings. Any difference from HEAD under those paths refuses it, untracked files included,
    since a new module the run imports changes it as much as an edit does. Only read-only git
    commands are run.
    """
    status = _git(project_dir, "status", "--porcelain", "--", *FROZEN_PATHS)
    changed = [line for line in status.splitlines() if line.strip()]
    if changed:
        shown = "; ".join(changed[:10]) + ("; ..." if len(changed) > 10 else "")
        raise RunRefused(f"{len(changed)} uncommitted change(s) under {', '.join(FROZEN_PATHS)} "
                         f"({shown}). Commit them, the freeze, before the labelled run.")
    return _git(project_dir, "rev-parse", "HEAD").strip()


_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def outcome_exclusions(rules: Mapping) -> list[str]:
    """The measures the frozen checks left out (``mil_analysis.outcome.excluded``), in the
    measures' own order (rulings R2 and R27).

    The run refuses to start while the outcome has no ``excluded`` list: the checks must have
    been run on TD (``mil-checks``) and their result committed with the freeze, so that the run
    is never the first to decide which measures it uses. ``[]`` records that nothing was
    excluded. A name that is not one of the ten measures is refused: misspelt, it would leave its
    measure in.

    The rest of the outcome's shape, ``{excluded, source, date, code_hash}`` (R27 and R32), is
    checked here too, and up front: ``source`` and ``date`` are strings and the date reads
    ``YYYY-MM-DD``, since PyYAML reads an unquoted ``date: 2026-11-02`` as a ``datetime.date``,
    which json cannot write (that used to surface after every AUC had been logged); and
    ``code_hash`` is the 64 hex characters ``mil-checks`` wrote (``check_code_hash`` compares it).
    """
    outcome = rules["mil_analysis"].get("outcome") or {}
    if not isinstance(outcome, Mapping) or "excluded" not in outcome:
        raise RunRefused(
            "decision_rules.yaml mil_analysis.outcome has no 'excluded' list. Run `mil-checks` on "
            "TD, copy proposed_outcome from outputs/mil_checks.json into it ({excluded: [...], "
            "source: ..., date: \"YYYY-MM-DD\", code_hash: ...}) and commit it with the freeze.")
    excluded = outcome["excluded"]
    if not isinstance(excluded, (list, tuple)):
        raise RunRefused("decision_rules.yaml mil_analysis.outcome.excluded must be a list of "
                         "measure names ([] when the checks excluded nothing)")
    unknown = sorted(set(map(str, excluded)) - set(ALL_MEASURES))
    if unknown:
        raise RunRefused(f"decision_rules.yaml mil_analysis.outcome.excluded names no such "
                         f"measure: {unknown}; the ten are {list(ALL_MEASURES)}")
    for key in ("source", "date", "code_hash"):
        if not isinstance(outcome.get(key), str) or not outcome[key]:
            raise RunRefused(f"decision_rules.yaml mil_analysis.outcome.{key} must be a quoted "
                             f"string (YAML reads an unquoted date or number as something else), "
                             f"as in proposed_outcome of outputs/mil_checks.json; it is "
                             f"{type(outcome.get(key)).__name__}")
    if not _valid_date(outcome["date"]):
        raise RunRefused("decision_rules.yaml mil_analysis.outcome.date must read YYYY-MM-DD")
    if not _SHA256.fullmatch(outcome["code_hash"]):
        raise RunRefused("decision_rules.yaml mil_analysis.outcome.code_hash must be the 64 "
                         "lower-case hex characters `mil-checks` wrote")
    return [m for m in ALL_MEASURES if m in set(excluded)]


def _valid_date(text: str) -> bool:
    if not _DATE.fullmatch(text):
        return False
    try:
        datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        return False
    return True


def _lf(data: bytes) -> bytes:
    """Line endings as git stores them: a checkout may have turned them into CRLF."""
    return data.replace(b"\r\n", b"\n")


def code_hash(project_dir: Path) -> str:
    """SHA-256 of what decides the labelled run, besides the data (ruling R32).

    Over the contents of every ``src/asdpipe/**/*.py`` (by sorted path, the path counting as well,
    so a renamed module changes it), of ``configs/pipeline.yaml``, and of
    ``configs/decision_rules.yaml`` parsed with its ``mil_analysis.outcome`` entry removed
    (a canonical JSON dump of the parsed YAML: comments and key order do not count). The outcome
    is left out because it holds the hash itself, and because Task 8 fills it in after
    ``mil-checks``; everything else in the two configs is a setting the run reads.

    ``mil-checks`` records the hash, the freeze commit carries it as ``outcome.code_hash``, and
    ``mil-run`` recomputes it (``check_code_hash``): an outcome decided on other code or other
    settings than the run's own is refused, which ``freeze_guard`` (tree equals HEAD) cannot see.
    A part that is missing is an error, not an empty part: a wrong directory would otherwise
    hash to the same nothing in both commands. Line endings are normalised, so a checkout with
    CRLF does not change it.
    """
    root = Path(project_dir)
    src = root / "src" / "asdpipe"
    files = sorted(src.rglob("*.py"), key=lambda p: p.relative_to(root).as_posix())
    if not files:
        raise RunRefused("no source file under src/asdpipe in the project directory, so the code "
                         "cannot be hashed")
    parts = {p.relative_to(root).as_posix(): hashlib.sha256(_lf(p.read_bytes())).hexdigest()
             for p in files}
    pipeline = root / "configs" / "pipeline.yaml"
    rules_path = root / "configs" / "decision_rules.yaml"
    for path in (pipeline, rules_path):
        if not path.is_file():
            raise RunRefused(f"configs/{path.name} is missing in the project directory, so the "
                             "code and settings cannot be hashed")
    parts["configs/pipeline.yaml"] = hashlib.sha256(_lf(pipeline.read_bytes())).hexdigest()
    rules = yaml.safe_load(rules_path.read_text(encoding="utf-8")) or {}
    if isinstance(rules.get("mil_analysis"), dict):
        rules["mil_analysis"].pop("outcome", None)
    canonical = json.dumps(rules, sort_keys=True, ensure_ascii=False, default=str)
    parts["configs/decision_rules.yaml (without mil_analysis.outcome)"] = hashlib.sha256(
        canonical.encode("utf-8")).hexdigest()
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest()


def check_code_hash(project_dir: Path, rules: Mapping) -> str:
    """Refuse unless the recorded ``outcome.code_hash`` is the hash of the code and settings as
    they are now (ruling R32); return that hash. The message gives the first twelve characters
    of each, nothing else."""
    recorded = (rules["mil_analysis"].get("outcome") or {}).get("code_hash")
    recorded = recorded if isinstance(recorded, str) else ""
    now = code_hash(project_dir)
    if now != recorded:
        raise RunRefused(
            f"decision_rules.yaml mil_analysis.outcome.code_hash ({recorded[:12] or 'missing'}) is "
            f"not the hash of the code and settings as they are now ({now[:12]}): the recorded "
            "checks were run on other code. Run `mil-checks` again, record its proposed_outcome "
            "and commit it with the freeze.")
    return now


# ------------------------------------------------------------------------------ instruments
class WordMemo:
    """A word segmenter that segments each distinct text once (CKIP is the slow part).

    ``segment_gate`` asks for the words of the TD corpus four times, once per convention
    transform, and most texts are the same in every pass.
    """

    def __init__(self, segment: Callable[[list[str]], list[list[str]]]):
        self._segment = segment
        self._known: dict[str, list[str]] = {}

    def __call__(self, texts: Sequence[str]) -> list[list[str]]:
        texts = list(texts)
        new = sorted({t for t in texts if t not in self._known})
        if new:
            got = self._segment(new)
            if len(got) != len(new):
                raise ValueError(f"the segmenter returned {len(got)} word lists for {len(new)} "
                                 "texts")
            self._known.update(zip(new, (list(w) for w in got)))
        return [list(self._known[t]) for t in texts]


def _nothing() -> None:
    return None


def _no_environment() -> dict:
    return {}


@dataclass(frozen=True)
class Tools:
    """The instruments both commands measure with, as plain functions (module docstring).

    ``words_of``: texts -> one word list per text (CKIP). ``halves``: a normalised session ->
    ``b7_a11_halves``. ``f0``: a normalised session -> ``f0_three_way`` of its child turns.
    ``child_rows``: a normalised corpus -> one mapping per session, in order, holding at least
    the four ``CHILD_MEASURES`` and the tracking-failure ratio (see ``child_values``).
    ``release`` is called once nothing more will be measured, before the models are trained.
    ``environment``: what the instruments ran on (``packages``, ``crepe``, ``cuda_device``), for
    the record both commands write (``environment_record``). It is asked for when something is
    written, never before, because the package versions are those of the models loaded by then.
    """

    words_of: Callable[[list[str]], list[list[str]]]
    halves: Callable[[SubjectRecord], Mapping[str, tuple[float, float]]]
    f0: Callable[[SubjectRecord], Mapping[str, float]]
    child_rows: Callable[[Corpus], Sequence[Mapping[str, float]]]
    release: Callable[[], None] = _nothing
    environment: Callable[[], Mapping] = _no_environment


def child_values(rows: Sequence[SubjectRow], ids: Sequence[str]) -> list[dict[str, float]]:
    """Every value the extraction gave each session, in ``ids`` order, as one mapping: its pooled
    features and its quality figures. The four child measures are pooled features (B7 and A11 in
    the B stream, A2 and B2 in the C stream); the tracking-failure ratio the F0 group rule reads
    is a C-stream quality figure, not a feature. A pooled value wins over a quality figure of the
    same name."""
    by_id = {r.subject_id: r for r in rows}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise RuntimeError(f"the extraction returned no row for {missing}")
    return [{**by_id[i].quality, **by_id[i].e_pooled, **by_id[i].c_pooled, **by_id[i].b_pooled}
            for i in ids]


def real_tools(cfg: Mapping, backends, aligner=None, *,
               fresh_cache: Path | None = None) -> Tools:
    """The production instruments: CKIP (memoised), ECAPA for the B7 halves, CREPE with pYIN and
    Praat for the F0 check, and ``extract_corpus`` with the forced aligner for the child-level
    measures (only ``mil-run`` extracts, so only it needs the aligner).

    **``fresh_cache``: the labelled run (ruling R33).** Given a directory, every F0 track and
    every feature the run uses is computed in this process, for both groups: the F0 cache is a
    new, empty directory under ``fresh_cache`` (``Backends.use_fresh_f0_cache``) and
    ``extract_corpus`` neither reads nor writes its features pickle. The caches are keyed on the
    audio and the settings but not on the device, the batch size or the torchcrepe version, so
    tracks left by an earlier run (TD's, from ``extract-real``) could be served beside ASD tracks
    made now under other settings, the one difference between groups no shuffle of labels can
    show. The decoded-audio and alignment caches stay: neither depends on the GPU, and their keys
    hold everything their content depends on (the file's identity and the sample rate; the audio
    samples, the turn boundaries, the words and the aligner's settings). CREPE must run on CUDA,
    the frozen device; any other device is refused, before anything is created.
    """
    if fresh_cache is not None:
        if not str(backends.device).startswith("cuda"):
            raise RunRefused(f"CREPE would run on '{backends.device}', not on CUDA. The labelled "
                             "run tracks pitch on the frozen device, for both groups, in this "
                             "process; nothing was computed.")
        backends.use_fresh_f0_cache(fresh_cache)

    def child_rows(corpus: Corpus) -> list[dict[str, float]]:
        if aligner is None:
            raise RuntimeError("the child-level measures need the forced aligner (models.mfa)")
        kw = {"use_cache": False} if fresh_cache is not None else {}
        rows = extract_corpus(corpus, cfg, backends, aligner=aligner, **kw)
        return child_values(rows, [s.subject_id for s in corpus.subjects])

    def environment() -> dict:
        import torch

        return {"packages": backends.versions(), "crepe": backends.crepe_settings(),
                "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None}

    def release() -> None:
        # The speech models leave the GPU, so the share this process may use
        # (ASDPIPE_GPU_MEM_FRACTION) goes to the models being trained.
        for name in ("ecapa", "sbert", "ckip_ws"):
            backends.__dict__.pop(name, None)          # functools.cached_property values
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return Tools(
        words_of=WordMemo(backends.segment),
        halves=lambda s: b7_a11_halves(s, cfg, backends),
        f0=lambda s: f0_three_way(s.audio, s.sample_rate, child_turns(s.session.turns), cfg,
                                  backends),
        child_rows=child_rows, release=release, environment=environment)


# ------------------------------------------------------------------------------ loading
@dataclass(frozen=True, eq=False)
class Arm:
    """One cohort as the commands use it: as transcribed (not yet normalised), every record named
    by its pseudonym, in pseudonym order. ``exposed`` marks the ASD sessions seen before the
    freeze (DEVIATIONS D10) and is False for every TD session."""

    cohort: str
    corpus: Corpus
    exposed: tuple[bool, ...]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(s.subject_id for s in self.corpus.subjects)


def _natural(text: str) -> list:
    """Sort key putting P2 before P10."""
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", text)]


def _renamed(record: SubjectRecord, pid: str) -> SubjectRecord:
    return replace(record, subject_id=pid,
                   session=replace(record.session, session_id=pid, subject_id=pid))


def _failure(e: BaseException) -> str:
    """What may be said about an error raised while a case was loaded: its type, and for a file
    error the operating system's own text, for a JSON syntax error its line and column. Never the
    message itself, which can hold a path through the child's folder or a value read from the
    transcript (``float()`` quotes the text it could not read)."""
    if isinstance(e, json.JSONDecodeError):
        return f"{type(e).__name__} at line {e.lineno}, column {e.colno}"
    if isinstance(e, OSError) and e.strerror:
        return f"{type(e).__name__} ({e.strerror})"
    return type(e).__name__


def load_arm(cfg: Mapping, cohort: str, id_root: Path, cache_dir: Path | None = None,
             log: Callable[[str], None] = print) -> Arm:
    """Load one cohort (``"TD"`` or ``"ASD"``) from ``realdata.root`` and pseudonymise it at once.

    The cases are ``discover_cases``': folders holding an audio file and a transcript, so an ASD
    session not yet transcribed is not a case. ``realdata.exclude_cases`` holds ids, which
    ``resolve_excluded`` turns into the folders to skip through the excluded-children table under
    ``id_root``. Every case is looked up in the ID maps under
    ``id_root`` before any is loaded (a case the maps do not list stops here, named by cohort and
    position), then loaded by ``load_subject`` under the frozen cleaning rules and renamed to its
    pseudonym. A case that cannot be loaded, or has no child turn, stops the run with its
    pseudonym and a reason that holds no path and nothing read from the transcript (the error's
    type, counts, the system's own text for a file error; see ``_failure``). An ASD session is
    exposed when its transcript predates
    ``SEEN_BEFORE`` (D10). The TD set is the one its map fixes (the study's 19 sessions), so a
    session the TD map lists but the folders lack stops the run too.
    """
    if cohort not in ID_MAPS:
        raise ValueError(f"cohort must be one of {list(ID_MAPS)}, got {cohort!r}")
    rd = cfg["realdata"]
    ccfg = {**cfg, "realdata": {**rd, "cohort": cohort}}
    label = Label.ASD if cohort == "ASD" else Label.CONTROL
    mapping = pseudonyms(id_root)
    try:
        cases = discover_cases(rd["root"], rd.get("exclude_cases", []), cohort=cohort,
                               id_root=id_root)
    except OSError as e:
        raise RuntimeError(f"{cohort}: the case folders could not be listed "
                           f"({type(e).__name__})") from None
    if not cases:
        raise RuntimeError(f"{cohort}: no case folder holds an audio file and a transcript")
    ids = [lookup(mapping, case.case_id, cohort, k) for k, case in enumerate(cases, 1)]

    records, exposed = [], []
    for k, (case, pid) in enumerate(zip(cases, ids), 1):
        try:
            record, report = load_subject(case, ccfg, label, cache_dir)
            dated = datetime.fromtimestamp(case.transcript.stat().st_mtime)
        except Exception as e:             # its message may hold a path or a transcript value
            raise RuntimeError(f"{cohort} case #{k} ({pid}) could not be loaded: "
                               f"{_failure(e)}") from None
        if not report.usable:
            # Counts only. The loading report's own text lists the speaker labels it could not
            # map, and a child labelled by name is exactly what leaves no child turn.
            raise RuntimeError(
                f"{cohort} case #{k} ({pid}) is unusable: {len(report.errors)} loading error(s); "
                f"{report.n_child} child turn(s) after role mapping, {report.n_kept} turn(s) kept, "
                f"{report.n_unmapped_labels} turn(s) with a speaker label not in the role table "
                "(labels are not shown: a label can be a name)")
        records.append(_renamed(record, pid))
        exposed.append(cohort == "ASD" and dated < SEEN_BEFORE)

    if cohort == "TD":
        missing = sorted(set(_read_id_map(Path(id_root), "TD").values()) - set(ids),
                         key=_natural)
        if missing:
            raise RuntimeError(f"TD: {len(missing)} session(s) of the TD ID map were not found "
                               f"({', '.join(missing)}); the TD set is the one that map fixes")
    order = sorted(range(len(ids)), key=lambda k: _natural(ids[k]))
    arm = Arm(cohort, Corpus(subjects=tuple(records[k] for k in order),
                             config_hash=f"real:{cohort}"),
              tuple(exposed[k] for k in order))
    seen = f", {sum(arm.exposed)} of them seen before the freeze (D10)" if cohort == "ASD" else ""
    log(f"[mil] {cohort}: {len(arm.ids)} sessions{seen}")
    return arm


# ------------------------------------------------------------------------------ mil-checks
def checks_verdict(split_r: Mapping[str, float], td_f0_pass: Sequence[bool],
                   gate: Sequence[GateResult], rules: Mapping) -> dict[str, list[str]]:
    """Measure -> why the frozen rules leave it out, for the measures they leave out, in the
    measures' order (ruling R27). Every threshold is ``decision_rules.yaml``'s:

    * B7 and A11 need a split-half r of at least ``split_half.threshold``; an r that could not be
      computed fails;
    * both pitch measures go when more than ``f0_session.drop_if_td_fail_share_above`` of the TD
      sessions fail the F0 check;
    * a segment measure goes when the convention gate moves it by ``threshold_sd`` or more under
      any transform, or could not judge it under one of them: the plan's criterion is a shift
      below the threshold under all three, which an unjudged transform has not shown.
    """
    ma = rules["mil_analysis"]
    why: dict[str, list[str]] = {}

    threshold = float(ma["split_half"]["threshold"])
    for m in SPLIT_HALF_MEASURES:
        r = float(split_r[m])
        if not r >= threshold:                         # NaN fails as well
            why.setdefault(m, []).append(f"單雙數句分半一致性 r = {r:.3f} < {threshold:.2f}"
                                         if math.isfinite(r) else "單雙數句分半一致性算不出來")

    passed = [bool(p) for p in td_f0_pass]
    if not passed:
        raise ValueError("no TD session to judge the F0 check by")
    n_fail = passed.count(False)
    limit = float(ma["f0_session"]["drop_if_td_fail_share_above"])
    if n_fail / len(passed) > limit:
        for m in PITCH_MEASURES:
            why.setdefault(m, []).append(f"TD {n_fail}／{len(passed)} 場沒通過音高三法比對"
                                         f"（超過 {limit:.0%}）")

    shift_limit = float(rules["convention_invariance_gate"]["threshold_sd"])
    judged = {r.feature: r for r in gate}
    for m in SEGMENT_MEASURES:
        shifts = dict(judged[m].shifts) if m in judged else {}
        unjudged = [t for t in TRANSFORMS if t not in shifts]
        if unjudged:
            why.setdefault(m, []).append(f"慣例關卡無法判定（{'、'.join(unjudged)}）")
        for t, v in shifts.items():
            if not v < shift_limit:
                why.setdefault(m, []).append(f"慣例關卡 {t} 移動 {v:.3f} SD（≥ {shift_limit:.2f}）")
    return {m: why[m] for m in ALL_MEASURES if m in why}


def mil_checks(cfg: Mapping, rules: Mapping, *, project_dir: Path, id_root: Path, out_dir: Path,
               tools: Tools, cache_dir: Path | None = None, log: Callable[[str], None] = print,
               today: str | None = None) -> dict:
    """Research plan 3.4 on TD, before the freeze: the split-half reliability of B7 and A11, the
    three-tracker F0 check of every session and the convention gate of the six segment measures.
    No label and no ASD session is read.

    Writes ``mil_checks.json`` and ``mil_checks.md`` to ``out_dir`` and returns the JSON's
    content, whose ``proposed_outcome`` (``excluded``, ``source``, ``date`` and ``code_hash``:
    rulings R27 and R32) is what goes into ``decision_rules.yaml mil_analysis.outcome``. The
    ``code_hash`` is that of ``project_dir``'s source and settings (``code_hash``), taken before
    the first measurement and again before the files are written: checks that ran while the code
    changed are tied to neither version, so they are refused. The split halves and the F0 check
    are taken on the normalised sessions, the ones the measures are computed from; the gate starts
    from the transcripts as they are, as ``segment_gate`` requires.
    """
    t0 = time.time()
    today = today or _today()
    ma = rules["mil_analysis"]
    chash = code_hash(project_dir)
    arm = load_arm(cfg, "TD", id_root, cache_dir, log)
    sessions, _ = normalize_corpus(arm.corpus, cfg)
    ids = list(arm.ids)
    n = len(ids)

    log(f"[mil-checks] split halves of B7 and A11 in {n} sessions")
    halves = [tools.halves(s) for s in sessions.subjects]
    split = {}
    for m in SPLIT_HALF_MEASURES:
        odd = np.array([h[m][0] for h in halves], dtype=float)
        even = np.array([h[m][1] for h in halves], dtype=float)
        r = split_half_r(odd, even)
        split[m] = {"r": r, "n_children": int((np.isfinite(odd) & np.isfinite(even)).sum()),
                    "halves": {i: [o, e] for i, o, e in zip(ids, odd, even)}}
        log(f"[mil-checks]   {m}: r = {r:.3f} over {split[m]['n_children']} children")

    fs = ma["f0_session"]
    f0 = {}
    for k, s in enumerate(sessions.subjects, 1):
        check = dict(tools.f0(s))
        check["pass"] = f0_session_pass(check, float(fs["min_agreement"]),
                                        float(fs["max_octave_error"]))
        f0[s.subject_id] = check
        log(f"[mil-checks] F0 check {k}/{n} {s.subject_id}: "
            f"{'pass' if check['pass'] else 'fail'}")
    n_fail = sum(not c["pass"] for c in f0.values())

    log("[mil-checks] convention gate of the six segment measures (four passes)")
    gate = segment_gate(arm.corpus, cfg, rules, tools.words_of)

    why = checks_verdict({m: v["r"] for m, v in split.items()},
                         [c["pass"] for c in f0.values()], gate, rules)
    if code_hash(project_dir) != chash:
        raise RunRefused("the source or the settings changed while the checks were running, so "
                         "no outcome can be tied to them. Run `mil-checks` again without editing.")
    out = {
        "kind": "mil_checks", "date": today, "cohort": "TD", "n_sessions": n, "ids": ids,
        "code_hash": chash, "profile": (cfg.get("models") or {}).get("profile"),
        "environment": environment_record(tools, id_root),
        "split_half": {"threshold": float(ma["split_half"]["threshold"]),
                       "statistic": ma["split_half"].get("statistic"), "measures": split},
        "f0_session": {"min_agreement": float(fs["min_agreement"]),
                       "max_octave_error": float(fs["max_octave_error"]),
                       "drop_if_td_fail_share_above": float(fs["drop_if_td_fail_share_above"]),
                       "n_pass": n - n_fail, "n_fail": n_fail, "fail_share": n_fail / n,
                       "sessions": f0},
        "convention_gate": {
            "threshold_sd": float(rules["convention_invariance_gate"]["threshold_sd"]),
            "measures": {r.feature: {"sd": r.sd, "shifts": dict(r.shifts),
                                     "spread": dict(r.spread)} for r in gate}},
        "excluded_because": why,
        "proposed_outcome": {"excluded": list(why), "source": CHECKS_SOURCE, "date": today,
                             "code_hash": chash},
    }
    out_dir = Path(out_dir)
    _check_ids([*ids, *f0])
    _write_json(out_dir / "mil_checks.json", out)
    write_checks_report(out_dir / "mil_checks.md", out)
    log(f"[mil-checks] proposed exclusions: {list(why) or 'none'} ({_elapsed(t0)})")
    return out


def write_checks_report(path: Path, checks: Mapping) -> Path:
    """``mil_checks.md``: the three checks of plan 3.4 and the outcome they propose, as tables
    of pseudonyms and numbers."""
    sh, fs, cg = checks["split_half"], checks["f0_session"], checks["convention_gate"]
    why = checks["excluded_because"]
    _check_ids([*checks.get("ids", []), *fs["sessions"]])
    L = ["# 建模前的自動檢查（只用 TD）", "",
         f"{checks['date']}｜TD {checks['n_sessions']} 場｜不讀任何標籤，不讀 ASD｜研究計畫 3.4", ""]

    L += ["## 1. 單雙數句交錯分半一致性", "",
          f"門檻 r ≥ {sh['threshold']:.2f}（跨孩子的 Pearson r，不做 Spearman-Brown 校正）。", "",
          "| 量測 | r | 兩半都有值的孩子 | 結果 |", "|---|---|---|---|"]
    for m, v in sh["measures"].items():
        ok = v["r"] is not None and v["r"] >= sh["threshold"]
        L.append(f"| `{m}` | {_num(v['r'])} | {v['n_children']} | {'通過' if ok else '沒通過'} |")

    L += ["", "## 2. 音高三法交叉比對（CREPE 對 pYIN、CREPE 對 Praat）", "",
          f"每場一致率 > {fs['min_agreement']:.2f}、八度錯誤 < {fs['max_octave_error']:.2f} 才算通過。"
          f"通過 {fs['n_pass']}／{checks['n_sessions']} 場；沒通過 {fs['fail_share']:.0%}"
          f"（超過 {fs['drop_if_td_fail_share_above']:.0%} 才把 A2、B2 整欄留空）。", "",
          "| 編號 | 一致率 pYIN | 一致率 Praat | 八度錯誤 pYIN | 八度錯誤 Praat | 比對幀數 | 結果 |",
          "|---|---|---|---|---|---|---|"]
    for pid in sorted(fs["sessions"], key=_natural):
        c = fs["sessions"][pid]
        L.append(f"| {pid} | {_num(c.get('agree_pyin'))} | {_num(c.get('agree_praat'))} | "
                 f"{_num(c.get('octave_pyin'))} | {_num(c.get('octave_praat'))} | "
                 f"{c.get('n_frames', '—')} | {'通過' if c['pass'] else '沒通過'} |")

    names = list(TRANSFORMS)
    L += ["", "## 3. 慣例關卡（六個段落量測）", "",
          f"三種逐字稿寫法轉換下，移動量都要 < {cg['threshold_sd']:.2f} 個 TD 標準差。", "",
          "| 量測 | TD 標準差 | " + " | ".join(names) + " | 結果 |",
          "|---|---|" + "---|" * len(names) + "---|"]
    for m, g in cg["measures"].items():
        cells = [_num(g["shifts"].get(t)) for t in names]
        bad = m in why and any("慣例關卡" in w for w in why[m])
        L.append(f"| `{m}` | {_num(g['sd'])} | " + " | ".join(cells)
                 + f" | {'沒通過' if bad else '通過'} |")

    out = checks["proposed_outcome"]
    L += ["", "## 建議寫入 decision_rules.yaml 的 mil_analysis.outcome", "",
          "依凍結的規則判定，沒有調任何門檻。抄進去並和凍結 commit 一起提交，`mil-run` 才會開始。"
          "`code_hash` 是這次檢查時的程式和設定的雜湊值；之後程式或設定有任何改動，`mil-run` "
          "都會拒絕，要重跑 `mil-checks`。日期和雜湊值都要加引號。", "",
          "```yaml", "  outcome:",
          "    excluded: [" + ", ".join(out["excluded"]) + "]",
          f"    source: {out['source']}", f"    date: \"{out['date']}\"",
          f"    code_hash: \"{out['code_hash']}\"", "```", ""]
    if why:
        L += ["| 拿掉的量測 | 原因 |", "|---|---|"]
        L += [f"| `{m}` | {'；'.join(reasons)} |" for m, reasons in why.items()]
    else:
        L.append("十個量測都通過，沒有要拿掉的。")
    return _write_text(path, L)


# ------------------------------------------------------------------------------ mil-run inputs
@dataclass(frozen=True, eq=False)
class Child:
    """One child as the run sees it: pseudonym, label (1 ASD, 0 TD), exposure (D10), the
    normalised turns, every value the extraction gave the session, and its F0 check."""

    id: str
    y: int
    exposed: bool
    turns: tuple[Turn, ...]
    values: Mapping[str, float]
    f0: Mapping[str, float]


@dataclass(frozen=True, eq=False)
class Prepared:
    """What the run's rules make of the children before any bag is built.

    ``child`` ``(N, 4)``: the ``CHILD_MEASURES`` columns, the pitch ones NaN for a session that
    failed the F0 check. ``seg_names`` and ``child_names``: the measures left, in their own
    order. ``excluded``: measure -> every reason it is out. ``blanked``: the ids whose pitch
    values were blanked. ``group_rule``: the F0 group rule's AUC and verdict.
    """

    child: np.ndarray
    seg_names: tuple[str, ...]
    child_names: tuple[str, ...]
    excluded: dict[str, list[str]]
    f0_pass: tuple[bool, ...]
    blanked: tuple[str, ...]
    group_rule: dict


def _value(values: Mapping, name: str) -> float:
    v = values.get(name)
    return float("nan") if v is None else float(v)


def prepare_children(children: Sequence[Child], rules: Mapping,
                     excluded: Sequence[str]) -> Prepared:
    """The child-level array and the measures left, after every rule of the run, in the order
    the brief gives (steps 4-6):

    0. before anything else, a child measure that is not already out (``excluded``, the D11 list)
       and that not one child has a value for is refused (``RunRefused``): the extraction no
       longer gives it under that name, and it would go on as an empty column (F8);
    1. a session that fails the F0 check (``f0_session_pass`` with ``mil_analysis.f0_session``)
       has its two pitch values set to NaN, which the model imputes from the training folds;
    2. the measures of ``excluded`` (``outcome_exclusions``: the frozen checks) leave every
       variant, and so does any measure the D11 convention gate listed (ruling R28);
    3. the F0 group rule: if the tracking-failure ratio alone tells ASD from TD (its AUC outside
       ``f0_group_rule.auc_keep_range``, or no AUC at all), both pitch measures leave the model.
    """
    unknown = sorted(set(excluded) - set(ALL_MEASURES))
    if unknown:
        raise ValueError(f"no such measure: {unknown}")
    ma = rules["mil_analysis"]
    fs, gr = ma["f0_session"], ma["f0_group_rule"]
    y = np.array([c.y for c in children], dtype=np.int64)

    child = np.array([[_value(c.values, m) for m in CHILD_MEASURES] for c in children],
                     dtype=np.float64).reshape(len(children), len(CHILD_MEASURES))
    # A measure the extraction no longer gives under this name reads as NaN for everybody (_value),
    # and the model would carry a constant column without saying so. Looked at before the F0
    # check blanks any pitch value, and only for the measures not already out.
    out_already = set(excluded) | set(confirmatory_exclusions(rules))
    silent = [m for j, m in enumerate(CHILD_MEASURES)
              if m not in out_already and np.isnan(child[:, j]).all()]
    if silent:
        raise RunRefused(f"no child has a value for {silent}: the extraction gives no measure "
                         "under that name (renamed?), and the model would carry an empty column. "
                         "Nothing was written.")
    f0_pass = tuple(f0_session_pass(c.f0, float(fs["min_agreement"]),
                                    float(fs["max_octave_error"])) for c in children)
    failed = ~np.array(f0_pass, dtype=bool)
    child[np.ix_(failed, [CHILD_MEASURES.index(m) for m in PITCH_MEASURES])] = np.nan

    why: dict[str, list[str]] = {}
    outcome = ma.get("outcome") or {}
    where = f"（{outcome.get('source', 'mil_analysis.outcome')}，{outcome.get('date', '日期未記')}）"
    for m in excluded:
        why.setdefault(m, []).append(f"建模前的檢查沒通過{where}")
    for m in confirmatory_exclusions(rules):
        if m in ALL_MEASURES:
            why.setdefault(m, []).append("D11 慣例不變性關卡已排除")

    ratio = np.array([_value(c.values, gr["feature"]) for c in children])
    lo, hi = (float(v) for v in gr["auc_keep_range"])
    auc, keep = f0_group_rule(ratio, y, lo, hi)
    if not keep:
        for m in PITCH_MEASURES:
            why.setdefault(m, []).append(
                f"音高追蹤失敗率的組間 AUC {auc:.3f} 不在 {lo:.2f}–{hi:.2f}" if math.isfinite(auc)
                else "音高追蹤失敗率的組間 AUC 算不出來")
    why = {m: why[m] for m in ALL_MEASURES if m in why}
    return Prepared(
        child=child, seg_names=tuple(m for m in SEGMENT_MEASURES if m not in why),
        child_names=tuple(m for m in CHILD_MEASURES if m not in why), excluded=why,
        f0_pass=f0_pass, blanked=tuple(c.id for c, ok in zip(children, f0_pass) if not ok),
        group_rule={"feature": gr["feature"], "auc": auc, "keep_range": [lo, hi], "keep": keep,
                    "n_children": int(np.isfinite(ratio).sum())})


def build_bags(children: Sequence[Child], prepared: Prepared,
               words_of: Callable[[list[str]], list[list[str]]], cfg: Mapping) -> dict[int, Bags]:
    """The children's bags at every segment length: ``mil.segment_length`` first, then each
    ``mil.sensitivity_lengths``, all with the measures ``prepared`` left.

    The words F3 counts are CKIP's segmentation of the child's normalised turns, asked once for
    every distinct text. A child with no segment at all stops here (``bags_from_arrays``).
    """
    mil = cfg["mil"]
    lengths = [operator.index(mil["segment_length"]),
               *(operator.index(n) for n in mil["sensitivity_lengths"])]
    kid_turns = [[t for t in c.turns if t.speaker is Speaker.CHILD and t.text] for c in children]
    texts = sorted({t.text for turns in kid_turns for t in turns})
    segmented = words_of(texts) if texts else []
    if len(segmented) != len(texts):
        raise ValueError(f"words_of returned {len(segmented)} word lists for {len(texts)} texts")
    by_text = dict(zip(texts, segmented))
    words = [{t.turn_id: list(by_text[t.text]) for t in turns} for turns in kid_turns]

    ids, y = [c.id for c in children], [c.y for c in children]
    exposed = [c.exposed for c in children]
    out = {}
    for length in lengths:
        seg = [segment_measures(c.turns, w, cfg, length) for c, w in zip(children, words)]
        bags = bags_from_arrays(ids, y, seg, prepared.child, SEGMENT_MEASURES, CHILD_MEASURES,
                                exposed)
        out[length] = bags.select(seg_names=prepared.seg_names,
                                  child_names=prepared.child_names)
    return out


# ------------------------------------------------------------------------------ analyse
def _not_run(reason: str, why: str) -> dict:
    return {"status": f"not run ({reason})", "why": why}


def _alpha(rules: Mapping) -> float:
    rule = str(rules["mil_analysis"]["separable_if"])
    m = re.fullmatch(r"\s*permutation p\s*<=\s*([0-9.]+)\s*", rule)
    if m is None:
        raise ValueError(f"mil_analysis.separable_if must read 'permutation p <= x', got {rule!r}")
    return float(m.group(1))


def _same_children(bags_by_len: Mapping[int, Bags], lengths: Sequence[int]) -> Bags:
    """The bags of the main length, once every length's bags are known to hold the same
    children (ids, labels, exposure) and the same measures and child values."""
    missing = [n for n in lengths if n not in bags_by_len]
    if missing:
        raise ValueError(f"bags_by_len has no bags for segment length(s) {missing}")
    base = bags_by_len[lengths[0]]
    for length in lengths[1:]:
        b = bags_by_len[length]
        if not (b.ids == base.ids and np.array_equal(b.y, base.y)
                and np.array_equal(b.exposed, base.exposed) and b.seg_names == base.seg_names
                and b.child_names == base.child_names
                and np.array_equal(b.child, base.child, equal_nan=True)):
            raise ValueError(f"the bags of length {length} do not hold the same children and "
                             f"measures as those of length {lengths[0]}")
    return base


def _plans(base: Bags, bags_by_len: Mapping[int, Bags], main: int, extra: Sequence[int],
           n_perm: int, min_unexposed: int) -> dict:
    """Variant -> (bags, segment length, permutations), or a not-run record (ruling R12).

    Every name list comes from the bags' own names, which the checks have already thinned.
    """
    seg, child = list(base.seg_names), list(base.child_names)
    plans: dict = {"primary": (base, main, n_perm)}
    plans["transcript_only"] = (
        (base.select(child_names=()), main, n_perm) if child else
        _not_run("the primary variant has no child measure: it is transcript-only already",
                 "primary 已經沒有孩子層級量測，本身就是只用逐字稿"))
    for length in extra:
        plans[f"seg_len_{length}"] = (bags_by_len[length], length, 0)
    if E5 in seg:
        plans["no_E5"] = (base.select(seg_names=[m for m in seg if m != E5]), main, 0)
        plans["E5_only"] = (base.select(seg_names=[E5], child_names=()), main, 0)
    else:
        for name in ("no_E5", "E5_only"):
            plans[name] = _not_run(f"{E5} was left out by the checks", "說話量 E5 已被檢查拿掉")
    for stream, names in STREAMS.items():
        gone = set(names)
        if gone & set(seg + child):
            plans[f"drop_{stream}"] = (base.select(seg_names=[m for m in seg if m not in gone],
                                                   child_names=[m for m in child if m not in gone]),
                                       main, 0)
        else:
            plans[f"drop_{stream}"] = _not_run(f"no {stream}-stream measure is left after the "
                                               "checks", f"{stream} 流的量測都已被檢查拿掉")
    unexposed = int(((base.y == 1) & ~base.exposed).sum())
    plans["unexposed_only"] = (
        (base.select(rows=(base.y == 0) | ~base.exposed), main, 0) if unexposed >= min_unexposed
        else _not_run(f"{unexposed} unexposed ASD < {min_unexposed}",
                      f"未曝光的 ASD 只有 {unexposed} 位，少於 {min_unexposed} 位"))
    for name, plan in list(plans.items()):
        if isinstance(plan, tuple) and not (plan[0].seg_names or plan[0].child_names):
            plans[name] = _not_run("no measure is left", "沒有量測可用")
    return plans


def _variant_result(bags: Bags, length: int, n_perm: int, result: Mapping) -> dict:
    out = {"status": "ok", "segment_length": length, "seg_names": list(bags.seg_names),
           "child_names": list(bags.child_names), "n_children": len(bags.ids),
           "n_asd": int((bags.y == 1).sum()), "n_td": int((bags.y == 0).sum()),
           "n_asd_exposed": int((bags.exposed & (bags.y == 1)).sum()), "n_permutations": n_perm}
    for model in ("mil", "logreg"):
        r = result[model]
        out[model] = {"auc": float(r["auc"]), "ci": [float(r["ci"][0]), float(r["ci"][1])],
                      "p": float(r["p"]) if n_perm else None,
                      "null": [float(v) for v in r["null"]],
                      "oof": {i: float(p) for i, p in zip(bags.ids, r["oof_mean"])}}
    out["gates_by_fold"] = np.asarray(result["mil"]["gates_by_fold"], dtype=float).tolist()
    return out


def _final_model(bags: Bags, gates_by_fold, mcfg: MILConfig, cv: CVSettings, seed: int) -> dict:
    """Plan 3.7: the model trained on every child with ``cv.seeds`` seeds (``seed`` + 0, 1, ...).

    Its gates are their mean over the seeds. A gate's stability is the share of the variant's
    ``repeats x n_folds`` cross-validation fits (each already the mean of its seeds) in which it
    is above ``GATE_OPEN``. A child's top segment is the real segment with the most attention,
    averaged over the seeds; ``None`` when the model weighed every segment alike, which is what
    it does with every segment gate shut.
    """
    n = len(bags.ids)
    seeds = [seed + s for s in range(cv.seeds)]
    fit = fit_predict(bags, np.ones((cv.seeds, n), dtype=bool), np.tile(bags.y, (cv.seeds, 1)),
                      seeds, mcfg, device=cv.device)
    names = [*bags.seg_names, *bags.child_names]
    folds = np.asarray(gates_by_fold, dtype=float)
    attention = np.asarray(fit.attention, dtype=float).mean(axis=0)
    top = {}
    for i, cid in enumerate(bags.ids):
        real = np.flatnonzero(bags.mask[i])
        w = attention[i, real]
        if real.size > 1 and np.ptp(w) <= _FLAT:
            top[cid] = {"segment": None, "weight": float(w.max()), "n_segments": int(real.size)}
        else:
            k = int(real[np.argmax(w)])
            top[cid] = {"segment": k, "weight": float(attention[i, k]),
                        "n_segments": int(real.size)}
    return {"seeds": seeds,
            "gates": dict(zip(names, map(float, np.asarray(fit.gates).mean(axis=0)))),
            "stability": dict(zip(names, map(float, (folds > GATE_OPEN).mean(axis=(0, 1))))),
            "n_fold_fits": int(folds.shape[0] * folds.shape[1]), "top_segment": top}


def analyse(bags_by_len: Mapping[int, Bags], cfg: Mapping, rules: Mapping,
            log: Callable[[str], None] = print,
            on_progress: Callable[[dict], None] | None = None) -> dict:
    """Every variant of plan 3.5-3.8, by ``evaluate``, and the final models of plan 3.7.

    ``on_progress`` (F6) is called with the analysis as it stands, ``settings``, ``children``
    and the ``variants`` and ``final`` models finished so far, each time one is finished (before
    its log line), so that a caller can write it down: a crash in the last variant then does not
    lose what the log has already shown.

    ``bags_by_len`` maps ``mil.segment_length`` and each of ``mil.sensitivity_lengths`` to the
    bags of the same children at that length, holding the measures the checks left. The
    variants: ``primary`` (every measure left) and ``transcript_only`` (no child measure), both
    with ``mil.n_permutations`` shuffles for both models; then, without shuffles, ``seg_len_*``
    (another segment length), ``no_E5`` and ``E5_only``, ``drop_E``/``drop_C``/``drop_B`` (one
    stream of ``STREAMS`` out), and ``unexposed_only`` (TD and the unexposed ASD, run only with at
    least ``mil_analysis.unexposed_only_min_asd`` of them).

    Returns ``settings``, ``children`` (ids, labels, exposure), ``variants`` (name -> result, or
    ``{"status": "not run (...)", "why": ...}``) and ``final`` (for each of ``primary`` and
    ``transcript_only`` that ran: gates, stability and every child's top segment), as plain
    numbers, strings and lists.
    """
    mil = cfg["mil"]
    mcfg = MILConfig.from_cfg(cfg)
    cv = CVSettings.from_cfg(cfg)
    n_perm = operator.index(mil["n_permutations"])
    seed = operator.index(mil["permutation_seed"])
    C = float(mil["logreg_C"])
    main = operator.index(mil["segment_length"])
    extra = [operator.index(n) for n in mil["sensitivity_lengths"]]
    min_unexposed = operator.index(rules["mil_analysis"]["unexposed_only_min_asd"])
    alpha = _alpha(rules)
    base = _same_children(bags_by_len, [main, *extra])
    if not (base.seg_names or base.child_names):
        raise ValueError("the checks left no measure at all: there is nothing to model")
    plans = _plans(base, bags_by_len, main, extra, n_perm, min_unexposed)

    variants: dict = {}
    final: dict = {}
    out = {
        "settings": {"segment_length": main, "sensitivity_lengths": extra,
                     "n_permutations": n_perm, "permutation_seed": seed, "logreg_C": C,
                     "separable_if_p_at_most": alpha, "unexposed_only_min_asd": min_unexposed,
                     "cv": {"n_folds": cv.n_folds, "repeats": cv.repeats, "seeds": cv.seeds,
                            "n_boot": cv.n_boot},
                     "model": dataclasses.asdict(mcfg), "device": cv.device},
        "children": {"ids": list(base.ids), "y": base.y.tolist(),
                     "exposed": base.exposed.tolist()},
        "variants": variants,
        "final": final,
    }

    def progress() -> None:
        if on_progress is not None:
            on_progress(out)

    for name, plan in plans.items():
        if isinstance(plan, dict):
            variants[name] = plan
            progress()
            log(f"[mil] {name}: {plan['status']}")
            continue
        bags, length, perm = plan
        log(f"[mil] {name}: {len(bags.ids)} children ({int(bags.y.sum())} ASD), "
            f"{len(bags.seg_names)} segment and {len(bags.child_names)} child measures, "
            f"segments of {length} exchanges, {perm} permutations")
        t = time.time()
        variants[name] = _variant_result(bags, length, perm,
                                         evaluate(bags, mcfg, cv, perm, seed, C))
        progress()
        m, lr = variants[name]["mil"], variants[name]["logreg"]
        log(f"[mil] {name}: model AUC {_auc_ci(m)}{_p_log(m['p'])}; "
            f"logistic AUC {_auc_ci(lr)}{_p_log(lr['p'])} ({_elapsed(t)})")

    for name in FINAL_VARIANTS:
        if variants[name]["status"] == "ok":
            log(f"[mil] final model ({name}): every child, {cv.seeds} seeds")
            final[name] = _final_model(plans[name][0], variants[name]["gates_by_fold"], mcfg, cv,
                                       seed)
            progress()
    return out


# ------------------------------------------------------------------------------ mil-run
def run_label(n_asd: int) -> str:
    """"正式" with all twenty ASD children (plan 3.8), otherwise "初步（ASD n=…）"."""
    return "正式" if n_asd == FULL_ASD else f"初步（ASD n={n_asd}）"


def check_asd_sample(n_asd: int, n_unexposed: int) -> None:
    """Refuse unless the ASD sessions are one of the pre-registered samples (``ASD_SAMPLES``).

    ``discover_cases`` takes any ``*.json`` of a case folder for its transcript and any speaker
    label that maps to a role for a speaker, so a stray file (a diarisation export, say) in a
    folder not yet transcribed would load as one more unexposed child, and the exposure check
    alone, which counts the six exposed sessions, would not notice. The message holds counts only.
    """
    if ASD_SAMPLES.get(n_asd) != n_unexposed:
        allowed = " or ".join(f"{n} ({u} unexposed)" for n, u in ASD_SAMPLES.items())
        raise RunRefused(f"{n_asd} ASD sessions, {n_unexposed} of them unexposed. Plan 3.8 "
                         f"pre-registers {allowed}: a session that is missing, or one that is "
                         "not a transcribed ASD session, changes the count. Refused before any "
                         "feature was computed.")


def output_files(n_asd: int) -> dict[str, str]:
    """The files a labelled run writes (F9), each carrying the run's label so that the formal
    run cannot replace the preliminary one's: ``results``, ``report`` and, keyed by variant, the
    attention excerpts of ``primary`` and ``transcript_only``."""
    slug = "final" if n_asd == FULL_ASD else f"prelim_asd{n_asd}"
    return {"results": f"mil_results_{slug}.json", "report": f"mil_report_{slug}.md",
            "primary": f"mil_attention_{slug}.md",
            "transcript_only": f"mil_attention_transcript_only_{slug}.md"}


def id_map_hashes(repo_root: Path) -> dict[str, str | None]:
    """File name -> SHA-256 of each private ID map, and of the table of excluded children
    (``None`` where the file is not there).

    The maps decide who is in the run and, with the children put in pseudonym order, which fold
    each lands in; the excluded table decides which two TD children ``exclude_cases`` leaves out.
    So a record of the run holds their hashes: a changed map is then visible. The hash is all
    that leaves this function; nothing read from a map is kept or shown.
    """
    out: dict[str, str | None] = {}
    for filename in (*(f for f, _ in ID_MAPS.values()), EXCLUDED_ID_MAP):
        path = Path(repo_root) / filename
        out[filename] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return out


def environment_record(tools: Tools, id_root: Path) -> dict:
    """What a command ran on, for the JSON it writes (F7, ruling R33): the Python executable and
    version, whatever the instruments report of themselves (``Tools.environment``: package
    versions, the CREPE model, batch size, device and torchcrepe version, the CUDA device's name)
    and the hashes of the private ID maps. Asked for when a file is written, once the models
    that count are loaded."""
    return {"python": sys.executable, "python_version": platform.python_version(),
            **tools.environment(), "id_map_sha256": id_map_hashes(id_root)}


def mil_run(cfg: Mapping, rules: Mapping, *, project_dir: Path, id_root: Path, out_dir: Path,
            tools: Tools, cache_dir: Path | None = None, log: Callable[[str], None] = print,
            today: str | None = None) -> dict:
    """The labelled run (plan 3.5-3.8), once, from the freeze commit.

    1. ``freeze_guard``, ``outcome_exclusions`` and ``check_code_hash`` first: a refusal reads no
       data.
    2. Both cohorts are loaded and pseudonymised (``load_arm``): TD with label 0, the transcribed
       ASD sessions with label 1. The ASD sessions dated before ``SEEN_BEFORE`` must be the
       six D10 lists, or exposure cannot be read from the dates and the run refuses; the ASD
       sample must be one of the two the plan pre-registers (``check_asd_sample``, F4); and no
       file this run would write may exist already (``output_files``, F9).
    3. Both go through the same normalisation, then each cohort through the extraction on its
       own (``tools.child_rows``), for the four child measures and the tracking-failure ratio,
       and every session through the F0 check (ruling R2). With the instruments of
       ``real_tools(fresh_cache=...)`` all of it is computed in this process, for both groups
       (ruling R33).
    4. ``prepare_children``: pitch values blanked per failed session, the recorded outcome and
       the D11 list applied, the F0 group rule decided. ``build_bags`` at lengths 5, 3 and 7.
    5. ``analyse``; the reports. The run is "正式" with twenty ASD children, otherwise
       "初步（ASD n=…）".

    Writes ``mil_results_<label>.json``, ``mil_report_<label>.md`` and the attention excerpts
    (``primary`` in ``mil_attention_<label>.md``, ``transcript_only`` beside it) to ``out_dir``
    (``<label>`` is ``final`` or ``prelim_asd<n>``); returns the results. The results file is
    written after every variant and final model as well, with ``"complete": false`` (F6), and
    replaced by the finished one: a crash late in the run leaves what the log had shown.
    """
    t0 = time.time()
    today = today or _today()
    head = freeze_guard(project_dir)
    excluded = outcome_exclusions(rules)
    check_code_hash(project_dir, rules)
    log(f"[mil] frozen at {head[:12]}; the recorded checks leave out {excluded or 'nothing'}")

    arms = [load_arm(cfg, cohort, id_root, cache_dir, log) for cohort in ("TD", "ASD")]
    n_exposed = sum(arms[1].exposed)
    if n_exposed != EXPOSED_ASD:
        raise RunRefused(f"{n_exposed} ASD transcripts predate {SEEN_BEFORE:%Y-%m-%d}, but "
                         f"DEVIATIONS D10 has {EXPOSED_ASD} sessions seen before the freeze: a "
                         "transcript's date has changed or a session is missing, so exposure "
                         "cannot be read from the dates")
    n_td, n_asd = len(arms[0].ids), len(arms[1].ids)
    check_asd_sample(n_asd, n_asd - n_exposed)
    files = output_files(n_asd)
    out_dir = Path(out_dir)
    taken = [name for name in files.values() if (out_dir / name).exists()]
    if taken:
        raise RunRefused(f"{len(taken)} output file(s) of this run exist already in "
                         f"{out_dir.name}/: {', '.join(taken)}. A labelled run does not overwrite "
                         "results; move them away first (a results file with \"complete\": false "
                         "is what an earlier run left when it stopped).")
    sessions = [normalize_corpus(arm.corpus, cfg)[0] for arm in arms]

    children = []
    for arm, corpus in zip(arms, sessions):
        log(f"[mil] {arm.cohort}: child-level measures (feature extraction)")
        values = list(tools.child_rows(corpus))
        if len(values) != len(corpus):
            raise RuntimeError(f"{arm.cohort}: the extraction gave {len(values)} rows for "
                               f"{len(corpus)} sessions")
        for k, (s, v, seen) in enumerate(zip(corpus.subjects, values, arm.exposed), 1):
            log(f"[mil] {arm.cohort}: F0 check {k}/{len(corpus)} {s.subject_id}")
            children.append(Child(id=s.subject_id, y=int(arm.cohort == "ASD"), exposed=bool(seen),
                                  turns=tuple(s.session.turns), values=dict(v),
                                  f0=dict(tools.f0(s))))

    prepared = prepare_children(children, rules, excluded)
    limit = float(rules["mil_analysis"]["f0_session"]["drop_if_td_fail_share_above"])
    f0_summary: dict = {}
    for arm in arms:
        ok = [p for c, p in zip(children, prepared.f0_pass) if c.y == int(arm.cohort == "ASD")]
        f0_summary[arm.cohort] = {"n": len(ok), "n_pass": sum(ok)}
    td_share = 1.0 - f0_summary["TD"]["n_pass"] / f0_summary["TD"]["n"]
    f0_summary.update(td_fail_share=td_share, td_share_limit=limit,
                      td_share_rule_now=td_share > limit,
                      outcome_drops_pitch=set(PITCH_MEASURES) <= set(excluded))
    log(f"[mil] F0 check: TD {f0_summary['TD']['n_pass']}/{f0_summary['TD']['n']} and ASD "
        f"{f0_summary['ASD']['n_pass']}/{f0_summary['ASD']['n']} sessions pass; pitch values "
        f"blanked for {list(prepared.blanked) or 'nobody'}")
    if f0_summary["td_share_rule_now"] != f0_summary["outcome_drops_pitch"]:
        log(f"[mil] note: {td_share:.0%} of the TD sessions fail the F0 check now, and the "
            "recorded checks decided otherwise about A2 and B2; the recorded outcome holds (R2)")
    g = prepared.group_rule
    log(f"[mil] F0 group rule: tracking-failure AUC {_num(g['auc'])} "
        f"({'keeps' if g['keep'] else 'drops'} A2 and B2); measures left out: "
        f"{list(prepared.excluded) or 'none'}")

    environment = environment_record(tools, id_root)   # the models that count are loaded by now
    results_head = {
        "kind": "mil_results", "run_label": run_label(n_asd), "date": today, "head": head,
        "outcome": dict(rules["mil_analysis"].get("outcome") or {}),
        "recording_condition": dict(rules["mil_analysis"].get("recording_condition") or {}),
        "profile": (cfg.get("models") or {}).get("profile"), "environment": environment,
        "counts": {"td": n_td, "asd": n_asd, "asd_exposed": n_exposed,
                   "asd_unexposed": n_asd - n_exposed},
        "excluded": prepared.excluded,
        "f0_sessions": {c.id: {**c.f0, "pass": ok} for c, ok in zip(children, prepared.f0_pass)},
        "f0_summary": f0_summary,
        "pitch_blanked": list(prepared.blanked),
        "f0_group_rule": prepared.group_rule,
        "files": files,
    }

    def checkpoint(part: Mapping) -> None:
        """F6: the results so far, in the file the finished results will replace."""
        partial = {**results_head, "complete": False, **part}
        _check_ids(_result_ids(partial))
        _write_json(out_dir / files["results"], partial)

    bags_by_len = build_bags(children, prepared, tools.words_of, cfg)
    tools.release()
    analysis = analyse(bags_by_len, cfg, rules, log, on_progress=checkpoint)

    # where in its session each child's top segment lies, for the report and the excerpts
    main = operator.index(cfg["mil"]["segment_length"])
    by_id = {c.id: c for c in children}
    spans_of: dict[str, list[tuple[float, float]]] = {}
    excerpts: dict[str, dict[str, tuple[float, float]]] = {}
    for name, fin in analysis["final"].items():
        excerpts[name] = {}
        for cid, top in fin["top_segment"].items():
            if top["segment"] is None:
                continue
            spans = spans_of.setdefault(cid, segment_spans(by_id[cid].turns, cfg, main))
            if len(spans) != top["n_segments"]:
                raise RuntimeError(f"{cid}: {len(spans)} segment spans for "
                                   f"{top['n_segments']} segments")
            lo, hi = spans[top["segment"]]
            top.update(start_s=float(lo), end_s=float(hi))
            excerpts[name][cid] = (lo, hi)

    results = {**results_head, "complete": True, **analysis}
    _check_ids(_result_ids(results))
    _write_json(out_dir / files["results"], results)
    write_mil_report(out_dir / files["report"], results)
    turns = {c.id: c.turns for c in children}
    for name, spans in excerpts.items():
        write_attention_excerpts(out_dir / files[name], spans, turns)
    log(f"[mil] {results['run_label']}: wrote {files['results']}, {files['report']} and the "
        f"attention excerpts to {out_dir.name}/ ({_elapsed(t0)})")
    return results


# ------------------------------------------------------------------------------ reports
def _result_ids(results: Mapping) -> list:
    """Every child id a results dict holds, wherever it holds one."""
    ids = list((results.get("children") or {}).get("ids", []))
    for v in (results.get("variants") or {}).values():
        for model in ("mil", "logreg"):
            ids += list(((v.get(model) or {}).get("oof")) or {})
    for fin in (results.get("final") or {}).values():
        ids += list(fin.get("top_segment") or {})
    return ids + list(results.get("pitch_blanked") or []) + list(results.get("f0_sessions") or {})


def _check_ids(ids) -> None:
    """Refuse to write anything if an id does not look like a pseudonym. The message names its
    position only: an id that fails may be a child's name."""
    for k, i in enumerate(ids, 1):
        if not (isinstance(i, str) and _PSEUDONYM.fullmatch(i)):
            raise ValueError(f"id #{k} in the results does not look like a pseudonym; "
                             "nothing was written")


def _num(x, digits: int = 3) -> str:
    if x is None or not math.isfinite(float(x)):
        return "—"
    return f"{float(x):.{digits}f}"


def _auc_ci(r: Mapping) -> str:
    return f"{_num(r['auc'])} [{_num(r['ci'][0])}, {_num(r['ci'][1])}]"


def _p_log(p) -> str:
    return "" if p is None else f" p {p:.3f}"


def _p_cell(p, alpha: float) -> str:
    if p is None:
        return "—"
    return f"{p:.3f}" + ("（分得開）" if p <= alpha else "")


def _delta(a, b) -> str:
    if a is None or b is None or not (math.isfinite(a) and math.isfinite(b)):
        return "—"
    return f"{a - b:+.3f}"


def _clock(seconds: float) -> str:
    tenths = int(round(float(seconds) * 10))
    return f"{tenths // 600:02d}:{tenths % 600 / 10:04.1f}"


def _elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{s:.0f} s" if s < 120 else f"{s / 60:.1f} min"


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _variant_text(name: str) -> str:
    if name.startswith("seg_len_"):
        return f"段落長度 {name[len('seg_len_'):]} 組對答"
    return _VARIANT_TEXT.get(name, "")


def _stream_of(measure: str) -> str:
    return next((s for s, names in STREAMS.items() if measure in names), "")


def write_mil_report(path: Path, results: Mapping) -> Path:
    """``mil_report_<label>.md``: the run's results as tables of pseudonyms and numbers.

    It reads named fields only (counts, settings, the variants' AUCs, the final models) and
    never a free-text field it was not written for; no transcript text is written (the excerpts
    have files of their own). An id that does not look like a pseudonym stops it before anything
    is written.
    """
    _check_ids(_result_ids(results))
    variants = results["variants"]
    s = results.get("settings") or {}
    alpha = float(s.get("separable_if_p_at_most", 0.05))
    label = results.get("run_label")
    L = [f"# 段落注意力模型：ASD 對 TD" + (f"（{label}）" if label else ""), ""]

    meta = []
    if results.get("date"):
        meta.append(str(results["date"]))
    if results.get("head"):
        meta.append(f"commit `{str(results['head'])[:12]}`")
    counts = results.get("counts")
    if counts:
        meta.append(f"TD {counts['td']} 位、ASD {counts['asd']} 位（曾曝光 {counts['asd_exposed']} 位）")
    cv = s.get("cv")
    if cv:
        meta.append(f"{cv['n_folds']} 份 × {cv['repeats']} 次重複 × {cv['seeds']} 個種子；"
                    f"打亂標籤 {s.get('n_permutations', 0):,} 次；bootstrap {cv['n_boot']:,} 次")
    L += ["｜".join(meta), "", "只放編號和數字；注意力最高段落的原文在 `outputs/mil_attention*.md`，不貼出。", ""]

    L += ["## 主要結果（模型和邏輯迴歸都做打亂標籤檢定）", "",
          "| 版本 | 量測（段落＋孩子） | 模型 AUC [95% 區間] | 打亂 p | LR AUC [95% 區間] | 打亂 p | 模型 − LR |",
          "|---|---|---|---|---|---|---|"]
    for name in FINAL_VARIANTS:
        v = variants.get(name)
        if v is None:
            continue
        if v.get("status") != "ok":
            L.append(f"| {name} | 未跑：{v.get('why', '')} | | | | | |")
            continue
        m, lr = v["mil"], v["logreg"]
        L.append(f"| {name} | {len(v['seg_names'])}＋{len(v['child_names'])} | {_auc_ci(m)} | "
                 f"{_p_cell(m['p'], alpha)} | {_auc_ci(lr)} | {_p_cell(lr['p'], alpha)} | "
                 f"{_delta(m['auc'], lr['auc'])} |")
    L += ["", f"打亂 p ≤ {alpha:g} 才寫「分得開」（真實 AUC 高過 95% 的打亂結果）。人數太少，兩個模型之間不做檢定。"]
    if (results.get("recording_condition") or {}).get("pending"):
        L.append("錄音條件待醫師確認：若 2025（TD）和 2026（ASD）年的條件不同，以 transcript_only 為主（計畫 3.8）。")
    # R30, said whatever the clinician answers: "transcript-only" is not wholly free of the audio
    L.append("transcript_only 不用孩子層級的聲學量測；但 E5（說話量）和 C1（語速）用到的每句起訖時間，"
             "在正規化時會依錄音的音量修剪，所以這兩個量測仍然會受錄音條件影響。")
    L.append("")

    L += ["## 附加分析（不做打亂檢定）", "",
          "| 版本 | 說明 | 孩子（ASD／TD） | 量測（段落＋孩子） | 模型 AUC [95% 區間] | LR AUC [95% 區間] | 模型 − LR |",
          "|---|---|---|---|---|---|---|"]
    for name, v in variants.items():
        if name in FINAL_VARIANTS:
            continue
        if v.get("status") != "ok":
            L.append(f"| {name} | {_variant_text(name)} | 未跑：{v.get('why', '')} | | | | |")
            continue
        L.append(f"| {name} | {_variant_text(name)} | {v['n_asd']}／{v['n_td']} | "
                 f"{len(v['seg_names'])}＋{len(v['child_names'])} | {_auc_ci(v['mil'])} | "
                 f"{_auc_ci(v['logreg'])} | {_delta(v['mil']['auc'], v['logreg']['auc'])} |")
    L.append("")

    primary = variants.get("primary") or {}
    used = set(primary.get("seg_names", [])) | set(primary.get("child_names", []))
    excluded = results.get("excluded") or {}
    L += ["## 量測的去留", "", "| 量測 | 流 | 層級 | 去留 | 原因 |", "|---|---|---|---|---|"]
    for m in ALL_MEASURES:
        level = "段落" if m in SEGMENT_MEASURES else "孩子"
        reason = "；".join(excluded.get(m, [])) or ("—" if m in used else "（未記錄）")
        L.append(f"| `{m}` | {_stream_of(m)} | {level} | {'納入' if m in used else '拿掉'} | {reason} |")
    # Spec 3.4 (research_plan_1030.md): said in every report, whatever else it holds
    L += ["", "音高沒有人工驗證：稽核清單 P1-3 原本還要求人工抽聽音高，這次依照「不需人工驗證」的原則"
          "沒有做；A2、B2 只靠自動的三法比對把關，沒有人聽過音高追蹤得準不準。"]
    f0 = results.get("f0_summary")
    if f0:
        blanked = results.get("pitch_blanked") or []
        L += ["", f"音高三法：TD {f0['TD']['n_pass']}／{f0['TD']['n']} 場、ASD {f0['ASD']['n_pass']}／"
              f"{f0['ASD']['n']} 場通過；沒通過的場次 A2、B2 當缺值（{'、'.join(blanked) or '無'}）。"
              f"TD 沒通過 {f0['td_fail_share']:.0%}（超過 {f0['td_share_limit']:.0%} 才整欄拿掉）。"]
    g = results.get("f0_group_rule")
    if g:
        lo, hi = g["keep_range"]
        L.append(f"音高追蹤失敗率的組間 AUC：{_num(g['auc'])}（保留範圍 {lo:.2f}–{hi:.2f}，"
                 f"{g['n_children']} 位孩子）→ {'保留' if g['keep'] else '拿掉'} A2、B2。")
    L.append("")

    children = results.get("children") or {}
    group = {i: ("ASD" + ("（曝光）" if e else "") if y else "TD")
             for i, y, e in zip(children.get("ids", []), children.get("y", []),
                                children.get("exposed", []))}
    for name, fin in (results.get("final") or {}).items():
        L += [f"## 最終模型：{name}（全部孩子、{len(fin['seeds'])} 個種子）", "",
              f"| 量測 | 閘門平均 | 穩定度（{fin['n_fold_fits']} 次折內擬合中閘門 > {GATE_OPEN} 的比例） |",
              "|---|---|---|"]
        L += [f"| `{m}` | {_num(gate)} | {_num(fin['stability'][m], 2)} |"
              for m, gate in fin["gates"].items()]
        L += ["", "每位孩子注意力最高的段落（段落從 1 數起）：", "",
              "| 編號 | 組別 | 段落 | 權重 | 起訖（秒） |", "|---|---|---|---|---|"]
        for cid in sorted(fin["top_segment"], key=_natural):
            top = fin["top_segment"][cid]
            if top["segment"] is None:
                where, span = f"各段權重相同（共 {top['n_segments']} 段）", "—"
            else:
                where = f"{top['segment'] + 1}／{top['n_segments']}"
                span = (f"{top['start_s']:.1f}–{top['end_s']:.1f}" if "start_s" in top else "—")
            L.append(f"| {cid} | {group.get(cid, '')} | {where} | {_num(top['weight'])} | {span} |")
        L.append("")
    return _write_text(path, L)


def write_attention_excerpts(path: Path, top_segments: Mapping[str, tuple[float, float]],
                             turns_by_id: Mapping[str, Sequence[Turn]]) -> Path:
    """The segment each child's final model weighed most, as transcript text (ruling R5).

    One block per pseudonym: the segment's start and end, then every turn it holds
    (``turns_in_span``) with its times and speaker, in the normalised text the measures read.
    It is the one output with transcript text, which can name people, so it is written to
    ``outputs/`` (git-ignored) and never printed or quoted in a report.
    """
    _check_ids(list(top_segments))
    L = ["# 注意力最高的段落（原文）", "",
         "這份檔案有逐字稿原文，只能留在 `outputs/`（已 gitignore），不貼進對話、報告或 commit。", "",
         "每位孩子一段：最終模型（全部孩子訓練，種子平均）注意力權重最高的段落，和段落裡的每一句"
         "（正規化後的文字；時間是 分:秒）。", ""]
    for cid in sorted(top_segments, key=_natural):
        lo, hi = top_segments[cid]
        L += [f"## {cid} · {_clock(lo)}–{_clock(hi)}", ""]
        for t in turns_in_span(turns_by_id[cid], lo, hi):
            who = "孩子" if t.speaker is Speaker.CHILD else "醫師"
            L.append(f"- `{_clock(t.t_start)}–{_clock(t.t_end)}` {who}：{t.text or '（沒有文字）'}")
        L.append("")
    return _write_text(path, L)


def _jsonable(x):
    """Plain JSON values: numpy numbers and arrays as Python ones, NaN and infinity as null, a
    date or a datetime as its ISO text (YAML turns an unquoted ``2026-11-02`` into one)."""
    if isinstance(x, date):                      # datetime is a date as well
        return x.isoformat()
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else None
    return x


def _write_json(path: Path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_text(path: Path, lines: Sequence[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
