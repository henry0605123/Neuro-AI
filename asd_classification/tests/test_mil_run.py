"""Wiring the segment-attention model to the real cohorts (``asdpipe.mil.run``).

The brief's four tests come first. The blocks after them pin the controller's rulings for this
task (task-7-context.md), so that a later edit cannot change them unnoticed:

* R27: ``mil-run`` refuses to start while ``decision_rules.yaml mil_analysis.outcome`` has no
  ``excluded`` list, and ``mil-checks`` proposes that list by the frozen rules alone;
* R12: every variant is built from the measures the bags still hold, so a measure the checks
  left out (E5, a whole stream, every child measure) makes a variant "not run", never a crash;
* a session that fails the F0 check loses its two pitch values, the F0 group rule takes the
  pitch measures out of every variant, and (R28) a child measure the D11 gate excluded stays out;
* R3: the freeze guard reads ``git status`` of the plan, configs/ and src/, untracked files
  included; R4: 12 ASD, 6 of them exposed, leaves too few unexposed children for
  ``unexposed_only``; R5: the attention excerpts.

The final review's fixes are pinned beside what they fix: F3 and R32 (the outcome's shape,
``{excluded, source, date, code_hash}``, as PyYAML loads it, and the hash that ties it to the
code), F2 (the profile of both commands), R33 (features computed fresh, on CUDA, with the CREPE
settings recorded), F4 (the two pre-registered ASD samples), F8 (a measure nobody has), F9 (output
names carry the run's label), F6 (results written as variants finish), F7 (the environment in
both JSON files), F11 (the report says the pitch was never listened to).

Synthetic data and hand-built sessions only. The folder names here are made up (假名甲 ...) and
stand for the children's names the real folders carry; the last tests save synthetic cohorts
under such names, run both commands on them, and check that no name reaches a log line, an error
or a written file.
"""

from __future__ import annotations

import copy
import csv
import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
import traceback
from datetime import date, datetime

import numpy as np
import pytest
import soundfile as sf
import yaml

from asdpipe.config import load_decision_rules
from asdpipe.contracts import Label, Speaker, SubjectRow
from asdpipe.eval.convention_gate import TRANSFORMS, GateResult
from asdpipe.mil import run as mil
from asdpipe.mil.bags import synthetic_bags
from asdpipe.mil.model import MILFit
from asdpipe.mil.run import (Child, RunRefused, Tools, WordMemo, analyse, build_bags,
                             checks_verdict, child_values, freeze_guard, lookup, mil_checks,
                             mil_run, outcome_exclusions, prepare_children, pseudonyms,
                             run_label, write_attention_excerpts, write_mil_report)
from asdpipe.mil.segments import (CHILD_MEASURES, PITCH_MEASURES, SEGMENT_MEASURES, STREAMS,
                                  segment_measures, segment_spans, turns_in_span)
from asdpipe.realdata import EXCLUDED_ID_MAP
from asdpipe.run import confirmatory_exclusions
from asdpipe.synth.generator import SynthConfig, generate_subject
from conftest import make_turn

CHILD, EXAMINER = Speaker.CHILD, Speaker.EXAMINER
E5 = "e_e5_talk_share"
VARIANTS = ["primary", "transcript_only", "seg_len_3", "seg_len_7", "no_E5", "E5_only",
            "drop_E", "drop_C", "drop_B", "unexposed_only"]


# ---------------------------------------------------------------- fixtures and helpers
@pytest.fixture(scope="module")
def rules():
    return load_decision_rules()


def _with_mil(cfg, *, repeats, seeds, n_permutations, epochs, n_boot=None):
    c = copy.deepcopy(cfg)
    c["mil"]["cv"].update(repeats=repeats, seeds=seeds)
    c["mil"]["n_permutations"] = n_permutations
    c["mil"]["model"]["epochs"] = epochs
    c["mil"]["device"] = "cpu"
    if n_boot is not None:
        c["mil"]["n_boot"] = n_boot
    return c


@pytest.fixture(scope="module")
def small_cfg(cfg):
    """The brief's reduced settings: 2 repeats, 2 seeds, 9 permutations, 100 epochs, on the CPU."""
    return _with_mil(cfg, repeats=2, seeds=2, n_permutations=9, epochs=100)


@pytest.fixture(scope="module")
def tiny_cfg(cfg):
    """For the tests that follow the wiring and read no AUC: 1 repeat, 1 seed, no shuffle."""
    return _with_mil(cfg, repeats=1, seeds=1, n_permutations=0, epochs=20, n_boot=100)


def _cohort_bags(n_asd=12, n_td=12, n_exposed=6, signal=1.5, seed=0):
    """Bags of segment lengths 3, 5 and 7 for one synthetic cohort: the same children, labels,
    exposure and child measures at every length, the longer segments fewer, as on real data. The
    first ``n_exposed`` ASD children are the exposed ones."""
    base = synthetic_bags(n_asd, n_td, signal=signal, seed=seed, n_seg=(10, 24))
    exposed = np.zeros(n_asd + n_td, dtype=bool)
    exposed[:n_exposed] = True
    out = {}
    for length, more in ((3, 2), (5, 0), (7, -2)):
        segs = synthetic_bags(n_asd, n_td, signal=signal, seed=seed + 100 + length,
                              n_seg=(10 + more, 24 + more))
        out[length] = dataclasses.replace(segs, child=base.child, exposed=exposed)
    return out


@pytest.fixture(scope="module")
def bags_by_len():
    return _cohort_bags()                       # R4: 12 ASD, 6 of them exposed


@pytest.fixture(scope="module")
def analysed(small_cfg, rules, bags_by_len):
    return analyse(bags_by_len, small_cfg, rules, log=lambda m: None)


def _outcome(rules, outcome):
    r = copy.deepcopy(rules)
    r["mil_analysis"]["outcome"] = outcome
    return r


OTHER_HASH = "a" * 64


def _full_outcome(excluded=(), **over):
    """An outcome in the shape rulings R27 and R32 fix: {excluded, source, date, code_hash}, as
    ``mil-checks`` proposes it and Task 8 copies it into decision_rules.yaml."""
    return {"excluded": list(excluded), "source": "outputs/mil_checks.md", "date": "2026-11-02",
            "code_hash": OTHER_HASH, **over}


def _recorded(rules, repo, excluded=()):
    """``rules`` carrying a full outcome whose code_hash is that of ``repo`` as it stands."""
    return _outcome(rules, _full_outcome(excluded, code_hash=mil.code_hash(repo)))


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True).stdout


@pytest.fixture
def tmp_git_repo(tmp_path, monkeypatch):
    """A committed project of its own: the research plan, configs/ and src/, and tests/ beside
    them. Git is kept from looking above tmp_path, so nothing here can reach another repo. It
    holds the three things the code hash reads (R32): src/asdpipe/*.py, pipeline.yaml and
    decision_rules.yaml."""
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    repo = tmp_path / "project"
    for rel, text in {"research_plan_1030.md": "plan\n", "configs/x.yaml": "a: 1\n",
                      "configs/pipeline.yaml": "mil: {segment_length: 5}\n",
                      "configs/decision_rules.yaml": "mil_analysis:\n  outcome: {}\n",
                      "src/m.py": "x = 1\n", "src/asdpipe/m.py": "x = 1\n",
                      "tests/t.py": "\n"}.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text, encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=test", "-c", "user.email=test@example.invalid",
         "commit", "-q", "-m", "freeze")
    return repo


def _write_maps(root, td=(), asd=(), bom=False, excluded=()):
    """The private ID maps in their real layout: (session, id) pairs per cohort, and, when
    ``excluded`` has (id, folder-name fragment) pairs, the table of excluded children."""
    encoding = "utf-8-sig" if bom else "utf-8"
    if excluded:
        with (root / EXCLUDED_ID_MAP).open("w", encoding=encoding, newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["xid", "session"])
            w.writerows(excluded)
    with (root / "_private_id_map.csv").open("w", encoding=encoding, newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pid", "fold", "session"])
        w.writerows((pid, k, session) for k, (session, pid) in enumerate(td, 1))
    with (root / "_private_id_map_asd.csv").open("w", encoding=encoding, newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["uid", "session"])
        w.writerows((uid, session) for session, uid in asd)


def _session(n_pairs, text="我喜歡吃蘋果"):
    """A hand-built session: ``n_pairs`` examiner prompts, each answered after half a second."""
    turns, t = [], 0.0
    for k in range(n_pairs):
        turns.append(make_turn(2 * k, EXAMINER, t, t + 1.0, "你喜歡什麼"))
        turns.append(make_turn(2 * k + 1, CHILD, t + 1.5, t + 2.5, text[: 2 + k % 5]))
        t += 4.0
    return tuple(turns)


def _kid(i, y, *, f0_ok=True, ratio=0.1, turns=()):
    """One child as the run sees it after extraction and the F0 check."""
    values = {"c_exaggerated_utterance_ratio": 0.10 + i, "b7_mean_pairwise_cosdist": 0.20 + i,
              "c_pitch_range_st_median": 3.0 + i, "b_level_rel_examiner_db": -5.0 + i,
              "c_f0_tracking_failure_ratio": ratio, "b_unrelated_feature": 99.0}
    check = {"agree_pyin": 0.97 if f0_ok else 0.40, "agree_praat": 0.96, "octave_pyin": 0.0,
             "octave_praat": 0.01, "n_frames": 500}
    return Child(id=f"{'A' if y else 'P'}{i:02d}", y=y, exposed=False, turns=tuple(turns),
                 values=values, f0=check)


def _chars(texts):
    return [list(t) for t in texts]


def _untouched_tools():
    """Instruments that fail the test if anything is measured."""
    def boom(*_):
        raise AssertionError("nothing may be measured here")
    return Tools(words_of=boom, halves=boom, f0=boom, child_rows=boom, release=boom)


# ---------------------------------------------------------------- the brief's four tests
def test_analyse_on_synthetic(small_cfg, rules, bags_by_len):   # repeats=2, seeds=2, n_permutations=9, epochs=100
    b3, b5, b7 = bags_by_len[3], bags_by_len[5], bags_by_len[7]
    res = analyse({3: b3, 5: b5, 7: b7}, small_cfg, rules, log=lambda m: None)
    assert {"primary", "transcript_only", "seg_len_3", "E5_only", "drop_B"} <= set(res["variants"])
    assert res["variants"]["unexposed_only"]["status"].startswith("not run")
    assert 0.0 <= res["variants"]["primary"]["mil"]["auc"] <= 1.0


def _messages(err):
    """What a traceback says about ``err`` and every exception chained to it, without the frames'
    source lines (in a test those are the test's own code, literals and all)."""
    out = []
    while err is not None:
        out.append("".join(traceback.format_exception_only(err)))
        err = err.__cause__ or (None if err.__suppress_context__ else err.__context__)
    return "".join(out)


def test_pseudonym_error_never_names_the_folder(tmp_path):
    with pytest.raises(KeyError) as e:
        lookup(pseudonyms(tmp_path), "某某某", cohort="TD", index=3)
    assert "某某某" not in str(e.value)
    # nor in the traceback: no exception that carries the name (a KeyError of the session) is
    # chained to it, which "During handling of the above exception" would print
    assert "某某某" not in _messages(e.value)
    assert e.value.args == ("TD case #3 has no pseudonym",)


@pytest.fixture
def res_with_ids(analysed):
    """A results dict the way mil_run writes it, for synthetic children, which also carries a
    folder name in two fields no report may read (a real results dict never holds one)."""
    res = copy.deepcopy(analysed)
    res.update(run_label=run_label(12), head="0123456789abcdef" * 2 + "01234567",
               date="2026-11-02", counts={"td": 12, "asd": 12, "asd_exposed": 6,
                                          "asd_unexposed": 6}, excluded={})
    res["sessions"] = {"syn-asd-000": "某某某"}
    res["variants"]["primary"]["note"] = "某某某"
    return res


def test_report_holds_ids_and_numbers_only(tmp_path, res_with_ids):
    p = write_mil_report(tmp_path / "r.md", res_with_ids)
    assert "某某某" not in p.read_text(encoding="utf-8")


def test_freeze_guard_refuses_dirty_tree(tmp_git_repo):
    (tmp_git_repo / "configs/x.yaml").write_text("a: 2"); pytest.raises(RuntimeError, freeze_guard, tmp_git_repo)


# ---------------------------------------------------------------- the private ID maps
def test_pseudonyms_come_from_both_private_maps(tmp_path):
    _write_maps(tmp_path, td=[("假名甲", "P01"), ("假名乙", "P02")], asd=[("假名丙", "A01")],
                bom=True)                                   # a byte-order mark is not data
    mapping = pseudonyms(tmp_path)
    assert mapping == {"假名甲": "P01", "假名乙": "P02", "假名丙": "A01"}
    assert lookup(mapping, "假名丙", cohort="ASD", index=1) == "A01"


@pytest.mark.parametrize("td, asd", [
    ([("假名甲", "P01"), ("假名甲", "P02")], []),          # one session, two ids
    ([("假名甲", "P01"), ("假名乙", "P01")], []),          # one id, two sessions
    ([("假名甲", "P01")], [("假名乙", "P01")]),            # ... across the two maps
    ([("假名甲", "P01")], [("假名甲", "A01")]),            # one session in both maps
    ([("假名甲", "假名甲")], []),                         # the id column holds a name
    ([("假名甲", "")], []),                              # no id
])
def test_a_faulty_map_is_refused_without_quoting_it(tmp_path, td, asd):
    _write_maps(tmp_path, td=td, asd=asd)
    with pytest.raises(ValueError) as e:
        pseudonyms(tmp_path)
    assert "假名" not in str(e.value)


def test_a_map_without_its_header_is_refused_without_quoting_it(tmp_path):
    (tmp_path / "_private_id_map.csv").write_text("P01,1,假名甲\n", encoding="utf-8")
    with pytest.raises(ValueError) as e:
        pseudonyms(tmp_path)
    assert "假名" not in str(e.value)


# ---------------------------------------------------------------- R3: the freeze guard
def test_freeze_guard_returns_head_and_counts_untracked_files(tmp_git_repo):
    head = _git(tmp_git_repo, "rev-parse", "HEAD").strip()
    assert freeze_guard(tmp_git_repo) == head
    # outside the plan, configs/ and src/, a change does not stop the run
    (tmp_git_repo / "tests/t.py").write_text("changed\n", encoding="utf-8")
    (tmp_git_repo / "outputs").mkdir()
    (tmp_git_repo / "outputs/report.md").write_text("x", encoding="utf-8")
    assert freeze_guard(tmp_git_repo) == head
    # a new module is as much a change as an edit: the run would import it
    (tmp_git_repo / "src/new.py").write_text("y = 2\n", encoding="utf-8")
    with pytest.raises(RunRefused, match="src/new.py"):
        freeze_guard(tmp_git_repo)
    (tmp_git_repo / "src/new.py").unlink()
    (tmp_git_repo / "research_plan_1030.md").write_text("changed plan\n", encoding="utf-8")
    with pytest.raises(RunRefused, match="research_plan_1030.md"):
        freeze_guard(tmp_git_repo)


def test_freeze_guard_refuses_outside_a_repository(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(RunRefused):
        freeze_guard(plain)


# ---------------------------------------------------------------- R27: the recorded outcome
def test_the_run_refuses_while_the_outcome_has_no_excluded_list(rules):
    for outcome in ({}, {"source": "outputs/mil_checks.md", "date": "2026-11-02"},
                    {"excluded": None}, {"excluded": E5}):
        with pytest.raises(RunRefused, match="excluded"):
            outcome_exclusions(_outcome(rules, outcome))
    with pytest.raises(RunRefused, match=r"\['e_e5_talk_shar'\]"):       # a typo is refused
        outcome_exclusions(_outcome(rules, _full_outcome(["e_e5_talk_shar"])))
    assert outcome_exclusions(_outcome(rules, _full_outcome())) == []
    # in the measures' own order, however the outcome lists them
    got = outcome_exclusions(_outcome(rules, _full_outcome(["b_level_rel_examiner_db",
                                                            "e_f3_fixed_phrase_cover"])))
    assert got == ["e_f3_fixed_phrase_cover", "b_level_rel_examiner_db"]


# ---------------------------------------------------------------- F3: the outcome as YAML gives it
def _loaded(rules, date_text, hash_text=f'"{OTHER_HASH}"'):
    """The outcome after ``yaml.safe_load`` of the text a person would write into
    decision_rules.yaml, with the date and the hash written as given."""
    text = ("mil_analysis:\n  outcome:\n    excluded: [e_f1_echo_bigram]\n"
            f"    source: outputs/mil_checks.md\n    date: {date_text}\n    code_hash: {hash_text}\n")
    return _outcome(rules, yaml.safe_load(text)["mil_analysis"]["outcome"])


def test_an_unquoted_date_in_the_outcome_is_refused_before_anything_runs(rules):
    # PyYAML reads `date: 2026-11-02` as a datetime.date, which json cannot write: the run used
    # to find that out at its very last step, after every AUC had been logged
    assert outcome_exclusions(_loaded(rules, '"2026-11-02"')) == ["e_f1_echo_bigram"]
    unquoted = _loaded(rules, "2026-11-02")
    assert isinstance(unquoted["mil_analysis"]["outcome"]["date"], date)      # the trap itself
    with pytest.raises(RunRefused, match="date"):
        outcome_exclusions(unquoted)
    for not_a_date in ('"2026-11-2"', '"02/11/2026"', '"2026-13-02"', "20261102", '""'):
        with pytest.raises(RunRefused, match="date"):
            outcome_exclusions(_loaded(rules, not_a_date))
    for key in ("source", "date", "code_hash"):                    # all four keys are needed
        outcome = _full_outcome()
        del outcome[key]
        with pytest.raises(RunRefused, match=key):
            outcome_exclusions(_outcome(rules, outcome))
    with pytest.raises(RunRefused, match="source"):
        outcome_exclusions(_outcome(rules, _full_outcome(source=5)))
    for not_a_hash in ('"abc"', f'"{"g" * 64}"', "12345"):          # 64 hex characters, quoted
        with pytest.raises(RunRefused, match="code_hash"):
            outcome_exclusions(_loaded(rules, '"2026-11-02"', not_a_hash))


def test_dates_are_written_to_json_as_text(tmp_path):
    path = mil._write_json(tmp_path / "x.json", {"d": date(2026, 11, 2),
                                                 "t": datetime(2026, 11, 2, 3, 4, 5)})
    assert json.loads(path.read_text(encoding="utf-8")) == {"d": "2026-11-02",
                                                           "t": "2026-11-02T03:04:05"}


# ---------------------------------------------------------------- R32: the outcome and the code
def test_the_code_hash_follows_the_code_and_settings_and_not_the_recorded_outcome(tmp_git_repo):
    (tmp_git_repo / "src/asdpipe/m.py").write_bytes(b"x = 1\n")      # LF, whatever the platform
    h = mil.code_hash(tmp_git_repo)
    assert re.fullmatch(r"[0-9a-f]{64}", h) and mil.code_hash(tmp_git_repo) == h
    # what the run executes, and the settings it reads, are all in it
    for rel in ("src/asdpipe/m.py", "configs/pipeline.yaml"):
        path = tmp_git_repo / rel
        old = path.read_text(encoding="utf-8")
        path.write_text(old + "# edit\n", encoding="utf-8")
        assert mil.code_hash(tmp_git_repo) != h, rel
        path.write_text(old, encoding="utf-8")
        assert mil.code_hash(tmp_git_repo) == h
    new = tmp_git_repo / "src/asdpipe/sub/n.py"                    # a new module, however deep
    new.parent.mkdir()
    new.write_text("y = 2\n", encoding="utf-8")
    assert mil.code_hash(tmp_git_repo) != h
    new.rename(new.with_name("o.py"))                              # and its name counts as well
    renamed = mil.code_hash(tmp_git_repo)
    assert renamed != h
    (tmp_git_repo / "src/asdpipe/sub/o.py").unlink()
    # outside src/asdpipe and the two configs nothing counts; nor do line endings (git may check
    # a file out with CRLF)
    (tmp_git_repo / "tests/t.py").write_text("changed\n", encoding="utf-8")
    (tmp_git_repo / "src/m.py").write_text("x = 5\n", encoding="utf-8")
    (tmp_git_repo / "src/asdpipe/m.py").write_bytes(b"x = 1\r\n")
    assert mil.code_hash(tmp_git_repo) == h


def test_the_code_hash_reads_the_rules_without_their_outcome(tmp_git_repo):
    path = tmp_git_repo / "configs/decision_rules.yaml"
    path.write_text("mil_analysis:\n  separable_if: 'p <= 0.05'\n  outcome: {}\n", encoding="utf-8")
    h = mil.code_hash(tmp_git_repo)
    # the outcome filled in (which is what Task 8 does), a comment, another key order: same hash
    path.write_text("# a comment\nmil_analysis:\n  outcome: {excluded: [], source: x, date: "
                    "'2026-11-02', code_hash: abc}\n  separable_if: 'p <= 0.05'\n", encoding="utf-8")
    assert mil.code_hash(tmp_git_repo) == h
    path.write_text("mil_analysis:\n  separable_if: 'p <= 0.01'\n  outcome: {}\n", encoding="utf-8")
    assert mil.code_hash(tmp_git_repo) != h                        # a rule is not the outcome


def test_the_code_hash_needs_all_three_parts(tmp_git_repo):
    for rel in ("configs/pipeline.yaml", "configs/decision_rules.yaml"):
        path = tmp_git_repo / rel
        text = path.read_text(encoding="utf-8")
        path.unlink()
        with pytest.raises(RunRefused, match=re.escape(rel)):
            mil.code_hash(tmp_git_repo)
        path.write_text(text, encoding="utf-8")
    (tmp_git_repo / "src/asdpipe/m.py").unlink()
    with pytest.raises(RunRefused, match="src/asdpipe"):
        mil.code_hash(tmp_git_repo)


def test_the_run_refuses_an_outcome_recorded_for_other_code(cfg, rules, tmp_git_repo, tmp_path):
    c = copy.deepcopy(cfg)
    # exclude_cases holds ids, which need the table under id_root; this test is about the hash
    c["realdata"].update(root=str(tmp_path / "no-data-here"), exclude_cases=[])
    kw = dict(project_dir=tmp_git_repo, id_root=tmp_path, out_dir=tmp_path / "out",
              tools=_untouched_tools(), log=lambda m: None)
    here = mil.code_hash(tmp_git_repo)
    with pytest.raises(RunRefused, match="code_hash") as e:
        mil_run(c, _outcome(rules, _full_outcome(code_hash=OTHER_HASH)), **kw)
    assert here[:12] in str(e.value) and OTHER_HASH[:12] in str(e.value)    # the two, shortened
    assert here not in str(e.value) and not (tmp_path / "out").exists()
    # with the matching hash the run gets past the check, to the data that is not there
    with pytest.raises(RuntimeError) as e:
        mil_run(c, _recorded(rules, tmp_git_repo), **kw)
    assert not isinstance(e.value, RunRefused)


def test_the_cli_checks_the_code_hash_before_it_loads_a_model(monkeypatch, cfg, rules, capsys):
    from asdpipe.cli import main as cli

    monkeypatch.delenv("ASDPIPE_PROFILE", raising=False)
    monkeypatch.setattr(mil, "freeze_guard", lambda project_dir: "0" * 40)
    monkeypatch.setattr(cli, "_backends", lambda cfg: pytest.fail("a model was loaded"))
    assert cli.cmd_mil_run(cfg, _outcome(rules, _full_outcome(code_hash=OTHER_HASH))) == 1
    assert "code_hash" in capsys.readouterr().err


def test_mil_run_refuses_before_it_reads_any_data(cfg, rules, tmp_git_repo, tmp_path):
    c = copy.deepcopy(cfg)
    c["realdata"]["root"] = str(tmp_path / "no-data-here")
    kw = dict(project_dir=tmp_git_repo, id_root=tmp_path, out_dir=tmp_path / "out",
              tools=_untouched_tools(), log=lambda m: None)
    with pytest.raises(RunRefused, match="excluded"):
        mil_run(c, _outcome(rules, {}), **kw)
    # and the freeze is checked first of all
    (tmp_git_repo / "configs/x.yaml").write_text("a: 3", encoding="utf-8")
    with pytest.raises(RunRefused, match="uncommitted"):
        mil_run(c, _outcome(rules, {"excluded": []}), **kw)
    assert not (tmp_path / "out").exists()


def test_the_cli_knows_both_commands(monkeypatch):
    from asdpipe.cli import main as cli

    monkeypatch.delenv("ASDPIPE_GPU_MEM_FRACTION", raising=False)
    seen = []
    monkeypatch.setattr(cli, "cmd_mil_checks", lambda cfg, rules: seen.append("checks") or 0)
    monkeypatch.setattr(cli, "cmd_mil_run", lambda cfg, rules: seen.append("run") or 0)
    assert cli.main(["mil-checks"]) == 0 and cli.main(["mil-run"]) == 0
    assert seen == ["checks", "run"]


def test_the_cli_refuses_the_run_before_it_loads_a_model(monkeypatch, cfg, rules, capsys):
    # Never call cmd_mil_run unpatched in a test: on the frozen tree it would start the real run.
    from asdpipe.cli import main as cli

    def refuse(project_dir):
        raise RunRefused("2 uncommitted change(s) under research_plan_1030.md, configs, src")

    monkeypatch.delenv("ASDPIPE_PROFILE", raising=False)
    monkeypatch.setattr(mil, "freeze_guard", refuse)
    monkeypatch.setattr(cli, "_backends", lambda cfg: pytest.fail("a model was loaded"))
    assert cli.cmd_mil_run(cfg, rules) == 1
    assert "[mil-run] refused: 2 uncommitted change(s)" in capsys.readouterr().err


def _gate(shifts_by_measure):
    """One GateResult per segment measure, every shift 0 unless given; None drops a transform."""
    out = []
    for m in SEGMENT_MEASURES:
        shifts = {t: 0.0 for t in TRANSFORMS}
        shifts.update(shifts_by_measure.get(m, {}))
        out.append(GateResult(feature=m, sd=1.0,
                              shifts={t: v for t, v in shifts.items() if v is not None}))
    return out


def test_the_checks_propose_exclusions_by_the_frozen_rules_alone(rules):
    f3, f1, e5 = SEGMENT_MEASURES[:3]
    gate = _gate({f3: {"merge_same_speaker": 0.20},        # at the threshold: out (< 0.20 passes)
                  f1: {"merge_same_speaker": 0.1999},       # just under it: in
                  e5: {"stitch_speaker_changes": None}})    # one transform not judged: out
    split = {"b7_mean_pairwise_cosdist": 0.70, "b_level_rel_examiner_db": 0.6999}   # >= 0.70 passes
    why = checks_verdict(split, [True] * 10 + [False] * 9, gate, rules)          # 9 of 19 fail
    assert list(why) == [f3, e5, "b_level_rel_examiner_db"]
    # 10 of 19 TD sessions failing is more than half: both pitch measures go; NaN r fails
    why = checks_verdict({**split, "b_level_rel_examiner_db": float("nan")},
                         [True] * 9 + [False] * 10, gate, rules)
    assert list(why) == [f3, e5, *PITCH_MEASURES, "b_level_rel_examiner_db"]
    assert all(why[m] for m in why)                                       # every one says why
    # exactly half failing is not more than half
    assert not set(PITCH_MEASURES) & set(checks_verdict(split, [True, False] * 10, gate, rules))


# ---------------------------------------------------------------- the child-level measures
def test_a_session_that_fails_the_f0_check_loses_its_two_pitch_values_and_nothing_else(rules):
    kids = [_kid(i, i % 2, f0_ok=(i != 3)) for i in range(10)]
    prep = prepare_children(kids, rules, excluded=[])
    col = {m: j for j, m in enumerate(CHILD_MEASURES)}
    for i, kid in enumerate(kids):
        for m in CHILD_MEASURES:
            blank = i == 3 and m in PITCH_MEASURES
            assert np.isnan(prep.child[i, col[m]]) == blank
            if not blank:
                assert prep.child[i, col[m]] == kid.values[m]
    assert prep.blanked == ("A03",)
    assert prep.f0_pass == tuple(i != 3 for i in range(10))
    # one failure ratio for every child cannot tell the groups apart: the measures stay in
    assert prep.group_rule["auc"] == 0.5 and prep.group_rule["keep"] is True
    assert prep.seg_names == SEGMENT_MEASURES and prep.child_names == CHILD_MEASURES
    assert prep.excluded == {}


def test_the_f0_group_rule_takes_the_pitch_measures_out_of_every_variant(tiny_cfg, rules):
    # a tracking-failure ratio that alone tells ASD from TD: AUC 1, outside 0.30-0.70
    kids = [_kid(i, int(i >= 5), ratio=0.5 if i >= 5 else 0.1, turns=_session(8 + i))
            for i in range(10)]
    prep = prepare_children(kids, rules, excluded=[])
    assert prep.group_rule["auc"] == 1.0 and prep.group_rule["keep"] is False
    assert list(prep.excluded) == list(PITCH_MEASURES)
    assert all("AUC" in why for m in PITCH_MEASURES for why in prep.excluded[m])
    res = analyse(build_bags(kids, prep, _chars, tiny_cfg), tiny_cfg, rules, log=lambda m: None)
    ran = [v for v in res["variants"].values() if v["status"] == "ok"]
    assert len(ran) == 9                    # every variant but unexposed_only (5 unexposed ASD)
    assert all(not set(PITCH_MEASURES) & set(v["child_names"]) for v in ran)


def test_a_child_measure_the_d11_gate_left_out_stays_out(rules):
    r = copy.deepcopy(rules)
    r["convention_invariance_gate"]["outcome"]["excluded_from_confirmatory"] = [
        *confirmatory_exclusions(rules), "b_level_rel_examiner_db"]
    prep = prepare_children([_kid(i, i % 2) for i in range(10)], r, excluded=[])
    assert "b_level_rel_examiner_db" not in prep.child_names
    assert any("D11" in why for why in prep.excluded["b_level_rel_examiner_db"])


def test_the_recorded_outcome_takes_its_measures_out_and_says_where_it_came_from(rules):
    r = _outcome(rules, _full_outcome(["e_f1_echo_bigram", "b7_mean_pairwise_cosdist"],
                                      date="2026-11-03"))
    kids = [_kid(i, i % 2) for i in range(10)]
    prep = prepare_children(kids, r, excluded=outcome_exclusions(r))
    assert "e_f1_echo_bigram" not in prep.seg_names
    assert "b7_mean_pairwise_cosdist" not in prep.child_names
    assert "2026-11-03" in prep.excluded["e_f1_echo_bigram"][0]
    with pytest.raises(ValueError):
        prepare_children(kids, r, excluded=["no_such_measure"])


def test_child_values_read_each_measure_where_the_extraction_puts_it():
    row = SubjectRow(subject_id="P01", label=Label.CONTROL,
                     b_pooled={"b7_mean_pairwise_cosdist": 0.3, "b_level_rel_examiner_db": -4.0},
                     c_pooled={"c_exaggerated_utterance_ratio": 0.1, "c_pitch_range_st_median": 5.0},
                     e_pooled={}, c_sequence=np.zeros((0, 8)), egemaps_functionals=np.zeros(88),
                     examiner_probe={}, ados={}, latents={},
                     quality={"c_f0_tracking_failure_ratio": 0.2})
    (got,) = child_values([row], ["P01"])
    assert {m: got[m] for m in CHILD_MEASURES} == {
        "c_exaggerated_utterance_ratio": 0.1, "b7_mean_pairwise_cosdist": 0.3,
        "c_pitch_range_st_median": 5.0, "b_level_rel_examiner_db": -4.0}
    assert got["c_f0_tracking_failure_ratio"] == 0.2              # a quality value, not a feature
    with pytest.raises(RuntimeError):
        child_values([row], ["P01", "P02"])


# ---------------------------------------------------------------- segments, words, bags
def test_segment_spans_and_turns_in_span_are_the_segments_the_measures_read(cfg):
    turns = (
        make_turn(0, EXAMINER, 0.0, 2.0, "你今天做什麼"),
        make_turn(1, CHILD, 2.5, 4.0, "我去公園"),
        make_turn(2, Speaker.OVERLAP, 4.1, 4.4, "嗯"),           # overlap turns take no part
        make_turn(3, EXAMINER, 5.0, 6.0, "跟誰去"),
        make_turn(4, CHILD, 6.0, 7.5, "跟媽媽"),
        make_turn(5, CHILD, 20.0, 21.0, "我喜歡溜滑梯"),         # child-led: 14 s of silence before
        make_turn(6, EXAMINER, 21.0, 22.0, "好棒"),              # starts where that one ends
        make_turn(7, CHILD, 22.5, 23.0, "對"),
    )
    words = {t.turn_id: list(t.text) for t in turns if t.speaker is CHILD}
    assert segment_spans(turns, cfg, 1) == [(0.0, 4.0), (5.0, 7.5), (20.0, 21.0), (21.0, 23.0)]
    assert segment_spans(turns, cfg, 2) == [(0.0, 7.5), (5.0, 21.0), (20.0, 23.0)]
    assert segment_spans(turns, cfg, 5) == [(0.0, 23.0)]
    assert [t.turn_id for t in turns_in_span(turns, 0.0, 4.5)] == [0, 1]
    assert [t.turn_id for t in turns_in_span(turns, 20.0, 21.0)] == [5]  # right edge open
    # A segment ends at the latest end among its exchanges, not at its last exchange's end: an
    # examiner turn that starts during a long reply opens an exchange that ends first.
    nested = (make_turn(0, EXAMINER, 0.0, 1.0, "你今天做什麼"),
              make_turn(1, CHILD, 1.5, 9.0, "我早上去公園然後去吃飯"),
              make_turn(2, EXAMINER, 3.0, 4.0, "嗯"))
    assert segment_spans(nested, cfg, 2) == [(0.0, 9.0)]
    for length in (1, 2, 3, 5):
        rows = segment_measures(turns, words, cfg, length)
        spans = segment_spans(turns, cfg, length)
        assert len(spans) == len(rows)
        for row, (lo, hi) in zip(rows, spans):
            inside = turns_in_span(turns, lo, hi)
            child_s = sum(t.duration for t in inside if t.speaker is CHILD)
            spoken_s = child_s + sum(t.duration for t in inside if t.speaker is EXAMINER)
            assert row[SEGMENT_MEASURES.index(E5)] == pytest.approx(child_s / spoken_s)


def test_words_are_segmented_once_per_distinct_text():
    calls = []

    def segment(texts):
        calls.append(list(texts))
        return [list(t) for t in texts]

    words = WordMemo(segment)
    assert words(["你好", "我", "你好"]) == [["你", "好"], ["我"], ["你", "好"]]
    assert words(["我", "他"]) == [["我"], ["他"]]
    assert words([]) == []
    assert calls == [["你好", "我"], ["他"]]
    with pytest.raises(ValueError):
        WordMemo(lambda texts: [])(["一句"])


def test_bags_hold_each_childs_segments_at_every_length(cfg, rules):
    kids = [_kid(i, i % 2, turns=_session(3 + i)) for i in range(6)]
    prep = prepare_children(kids, rules, excluded=[])
    bags = build_bags(kids, prep, _chars, cfg)
    assert list(bags) == [5, 3, 7]                       # segment_length, then sensitivity_lengths
    for length, b in bags.items():
        assert b.ids == tuple(k.id for k in kids)
        assert (b.seg_names, b.child_names) == (SEGMENT_MEASURES, CHILD_MEASURES)
        assert np.array_equal(b.child, prep.child.astype(np.float32), equal_nan=True)
        for i, kid in enumerate(kids):
            words = {t.turn_id: list(t.text) for t in kid.turns if t.speaker is CHILD}
            want = segment_measures(kid.turns, words, cfg, length)
            assert b.mask[i].sum() == len(want)
            assert np.allclose(b.seg[i, : len(want)], want, equal_nan=True)
    with pytest.raises(ValueError):
        build_bags(kids, prep, lambda texts: _chars(texts)[:-1], cfg)


# ---------------------------------------------------------------- analyse: the variants
def test_each_variant_is_the_one_the_plan_describes(analysed, bags_by_len):
    v = analysed["variants"]
    assert list(v) == VARIANTS
    seg, child = list(SEGMENT_MEASURES), list(CHILD_MEASURES)
    want = {
        "primary": (seg, child, 5),
        "transcript_only": (seg, [], 5),
        "seg_len_3": (seg, child, 3),
        "seg_len_7": (seg, child, 7),
        "no_E5": ([m for m in seg if m != E5], child, 5),
        "E5_only": ([E5], [], 5),
        "drop_E": ([m for m in seg if m not in STREAMS["E"]], child, 5),
        "drop_C": ([m for m in seg if m not in STREAMS["C"]],
                   [m for m in child if m not in STREAMS["C"]], 5),
        "drop_B": (seg, [m for m in child if m not in STREAMS["B"]], 5),
    }
    for name, (s, c, length) in want.items():
        got = v[name]
        assert (got["seg_names"], got["child_names"], got["segment_length"]) == (s, c, length)
        assert (got["n_children"], got["n_asd"], got["n_td"]) == (24, 12, 12)
        perm = 9 if name in ("primary", "transcript_only") else 0
        assert got["n_permutations"] == perm
        for model in ("mil", "logreg"):
            r = got[model]
            assert len(r["null"]) == perm and (r["p"] is None) == (perm == 0)
            assert 0.0 <= r["ci"][0] <= r["ci"][1] <= 1.0
            assert set(r["oof"]) == set(bags_by_len[5].ids)
    assert v["unexposed_only"]["status"] == "not run (6 unexposed ASD < 10)"


def test_the_final_model_reports_gates_stability_and_top_segments(analysed, bags_by_len):
    b = bags_by_len[5]
    assert set(analysed["final"]) == {"primary", "transcript_only"}
    for name, fin in analysed["final"].items():
        v = analysed["variants"][name]
        names = v["seg_names"] + v["child_names"]
        assert list(fin["gates"]) == names and all(0.0 <= g <= 1.0 for g in fin["gates"].values())
        folds = np.array(v["gates_by_fold"])                    # (2 repeats, 5 folds, measures)
        assert fin["n_fold_fits"] == 10
        assert list(fin["stability"]) == names
        assert list(fin["stability"].values()) == pytest.approx(
            (folds > 0.5).mean(axis=(0, 1)).tolist())
        assert set(fin["top_segment"]) == set(b.ids)
        for i, cid in enumerate(b.ids):
            top = fin["top_segment"][cid]
            assert top["n_segments"] == b.mask[i].sum()
            assert top["segment"] is None or 0 <= top["segment"] < top["n_segments"]


def test_the_final_model_trains_on_every_child_and_reads_each_childs_top_segment(
        monkeypatch, tiny_cfg, rules, bags_by_len):
    calls = []

    def record(bags, train_mask, labels, seeds, cfg, device="cpu"):
        calls.append((np.array(train_mask), np.array(labels), list(seeds), bags.child_names))
        n, t = bags.mask.shape
        att = np.zeros((len(seeds), n, t))
        for i in range(n):
            real = np.flatnonzero(bags.mask[i])
            att[:, i, real] = 1.0 / real.size
            if i:                                    # child 0 weighs its segments alike
                att[:, i, real] = 0.5 / real.size
                att[:, i, real[i % real.size]] += 0.5
        d = len(bags.seg_names) + len(bags.child_names)
        return MILFit(scores=np.zeros((len(seeds), n)),
                      gates=np.tile(np.linspace(0.0, 1.0, d), (len(seeds), 1)), attention=att)

    monkeypatch.setattr(mil, "fit_predict", record)       # the final model only; CV is untouched
    res = analyse(bags_by_len, tiny_cfg, rules, log=lambda m: None)
    b = bags_by_len[5]
    (mask, labels, seeds, child_names), (_, _, _, child_names_2) = calls
    assert mask.shape == (1, len(b.ids)) and mask.all()            # every child, one seed
    assert (labels == b.y).all() and seeds == [tiny_cfg["mil"]["permutation_seed"]]
    assert child_names == CHILD_MEASURES and child_names_2 == ()   # primary, transcript_only
    fin = res["final"]["primary"]
    names = list(SEGMENT_MEASURES + CHILD_MEASURES)
    assert fin["gates"] == pytest.approx(dict(zip(names, np.linspace(0.0, 1.0, len(names)))))
    assert fin["top_segment"][b.ids[0]]["segment"] is None
    for i, cid in enumerate(b.ids[1:], 1):
        n_real = int(b.mask[i].sum())
        top = fin["top_segment"][cid]
        assert top["segment"] == i % n_real and top["n_segments"] == n_real
        assert top["weight"] == pytest.approx(0.5 / n_real + 0.5)


def test_variants_are_built_from_the_measures_the_checks_left(tiny_cfg, rules, bags_by_len):
    seg = [m for m in SEGMENT_MEASURES if m != E5]
    child = ["c_exaggerated_utterance_ratio"]                 # every B-stream measure left out
    res = analyse({L: b.select(seg_names=seg, child_names=child) for L, b in bags_by_len.items()},
                  tiny_cfg, rules, log=lambda m: None)
    v = res["variants"]
    for name in ("no_E5", "E5_only", "drop_B", "unexposed_only"):
        assert v[name]["status"].startswith("not run"), name
    for name in ("primary", "transcript_only", "seg_len_3", "seg_len_7", "drop_E", "drop_C"):
        assert v[name]["status"] == "ok", name
    assert (v["primary"]["seg_names"], v["primary"]["child_names"]) == (seg, child)
    assert v["drop_C"]["seg_names"] == ["e_f3_fixed_phrase_cover", "e_f1_echo_bigram",
                                        "e_f4_you_per_100c"]
    assert v["drop_C"]["child_names"] == []


def test_with_no_child_measure_left_the_primary_is_already_transcript_only(tiny_cfg, rules,
                                                                          bags_by_len):
    res = analyse({L: b.select(child_names=()) for L, b in bags_by_len.items()}, tiny_cfg, rules,
                  log=lambda m: None)
    assert res["variants"]["transcript_only"]["status"].startswith("not run")
    assert set(res["final"]) == {"primary"}


def test_unexposed_only_runs_on_td_and_the_unexposed_asd_when_there_are_enough(tiny_cfg, rules):
    v = analyse(_cohort_bags(n_exposed=2), tiny_cfg, rules,              # 10 unexposed ASD
                log=lambda m: None)["variants"]["unexposed_only"]
    assert v["status"] == "ok" and (v["n_asd"], v["n_td"], v["n_children"]) == (10, 12, 22)
    assert not {"syn-asd-000", "syn-asd-001"} & set(v["mil"]["oof"])


def test_analyse_refuses_bags_that_hold_nothing_or_do_not_match(tiny_cfg, rules, bags_by_len):
    def boom(*_):
        raise AssertionError("nothing may be trained")

    with pytest.raises(ValueError):
        analyse({L: b.select(seg_names=(), child_names=()) for L, b in bags_by_len.items()},
                tiny_cfg, rules, log=boom)
    fewer = {**bags_by_len, 7: bags_by_len[7].select(rows=np.arange(1, 24))}
    with pytest.raises(ValueError):
        analyse(fewer, tiny_cfg, rules, log=boom)
    with pytest.raises(ValueError):
        analyse({5: bags_by_len[5], 3: bags_by_len[3]}, tiny_cfg, rules, log=boom)


def test_run_label():
    assert run_label(20) == "正式"
    assert run_label(10) == "初步（ASD n=10）"


# ---------------------------------------------------------------- reports
def test_the_report_shows_both_models_the_label_and_ids(tmp_path, res_with_ids):
    text = write_mil_report(tmp_path / "r.md", res_with_ids).read_text(encoding="utf-8")
    for name in ("primary", "transcript_only"):
        for model in ("mil", "logreg"):
            assert f"{res_with_ids['variants'][name][model]['auc']:.3f}" in text
    assert "初步（ASD n=12）" in text and "syn-td-011" in text and "unexposed_only" in text


def _transcript_only_note(text):
    return [line for line in text.splitlines()
            if "transcript_only" in line and "E5" in line and "C1" in line and "正規化" in line]


def test_the_report_says_transcript_only_still_hears_the_recording(tmp_path, res_with_ids):
    # R30: E5's and C1's turn edges pass through normalisation's energy trim. Said whatever the
    # clinician's answer about the recording set-ups, which this results dict does not even hold.
    for condition in (None, {"pending": True}, {"pending": False}):
        res = copy.deepcopy(res_with_ids)
        if condition is not None:
            res["recording_condition"] = condition
        note = _transcript_only_note(
            write_mil_report(tmp_path / "r.md", res).read_text(encoding="utf-8"))
        assert len(note) == 1 and "錄音" in note[0]


def test_the_report_says_the_pitch_was_never_listened_to(tmp_path, res_with_ids):
    # F11, spec 3.4: audit P1-3 asked for a manual listening to the pitch, which this run does not
    # do. Said whatever the results hold: with or without the F0 summary, any recording condition
    for drop in ((), ("f0_summary", "f0_group_rule"), ("f0_summary", "f0_group_rule", "excluded")):
        res = {k: v for k, v in copy.deepcopy(res_with_ids).items() if k not in drop}
        text = write_mil_report(tmp_path / "r.md", res).read_text(encoding="utf-8")
        said = [line for line in text.splitlines() if "P1-3" in line]
        assert len(said) == 1
        assert "人工抽聽音高" in said[0] and "沒有做" in said[0] and "A2" in said[0]


def test_a_report_never_writes_an_id_that_is_not_a_pseudonym(tmp_path, res_with_ids):
    res = copy.deepcopy(res_with_ids)
    res["children"]["ids"][0] = "某某某"
    with pytest.raises(ValueError) as e:
        write_mil_report(tmp_path / "r.md", res)
    assert "某某某" not in str(e.value) and not (tmp_path / "r.md").exists()


def test_attention_excerpts_hold_the_segments_turns_with_times_and_speakers(tmp_path):
    turns = (make_turn(0, EXAMINER, 0.0, 2.0, "你叫什麼名字"),
             make_turn(1, CHILD, 2.5, 3.5, "我不知道"),
             make_turn(2, Speaker.OVERLAP, 3.6, 3.9, "重疊的話"),
             make_turn(3, EXAMINER, 4.0, 5.0, "那你幾歲"),        # starts at the right edge
             make_turn(4, CHILD, 65.0, 66.5, "五歲"))
    p = write_attention_excerpts(tmp_path / "a.md", {"P10": (60.0, 70.0), "P02": (0.0, 4.0)},
                                 {"P02": turns, "P10": turns})
    blocks = p.read_text(encoding="utf-8").split("\n## ")[1:]
    assert [b.split()[0] for b in blocks] == ["P02", "P10"]
    first, second = blocks
    assert "你叫什麼名字" in first and "我不知道" in first
    assert "重疊的話" not in first and "那你幾歲" not in first
    assert "醫師" in first and "孩子" in first and "00:02.5" in first
    assert "五歲" in second and "01:05.0" in second
    with pytest.raises(ValueError):
        write_attention_excerpts(tmp_path / "b.md", {"某某某": (0.0, 1.0)}, {"某某某": turns})
    assert not (tmp_path / "b.md").exists()


# ---------------------------------------------------------------- both commands, saved cohorts
TD_NAMES = tuple(f"2025010{k}-假名{c}" for k, c in enumerate("甲乙丙丁戊己", 1))
ASD_NAMES = tuple(f"202602{k:02d}-假名{c}" for k, c in enumerate("庚辛壬癸子丑寅卯辰巳", 1))


PUNCTUATION = "。？"


def _save_case(folder, subject, child_label="兒童"):
    """One synthetic session saved the way the clinic exports it, audio and an _edited.json,
    with punctuation added the way a TD transcript has it (normalisation strips it again).
    ``child_label`` is the child's speaker label in the export (a real one can be a name)."""
    folder.mkdir(parents=True)
    sf.write(str(folder / f"{folder.name}.wav"), subject.audio, subject.sample_rate,
             subtype="FLOAT")
    segments = [{"speaker": child_label if t.speaker is CHILD else "測試者", "start": t.t_start,
                 "end": t.t_end, "text": t.text + ("。" if t.speaker is CHILD else "？")}
                for t in subject.session.turns]
    path = folder / f"{folder.name}_edited.json"
    path.write_text(json.dumps({"metadata": {}, "segments": segments}, ensure_ascii=False),
                    encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def saved_cohorts(tmp_path_factory):
    """Six TD and ten ASD synthetic sessions in folders named like the real ones: ten ASD is the
    preliminary sample (plan 3.8). The first six ASD transcripts are dated 2026-09-12, before
    SEEN_BEFORE (exposed, D10), the other four are not. The pseudonyms run against the folders'
    order, so ordering by folder and by pseudonym differ."""
    root = tmp_path_factory.mktemp("cohorts")
    data, ids = root / "data", root / "ids"
    ids.mkdir()
    sc = SynthConfig(turns_per_subject=30, seed=7)
    for k, name in enumerate(TD_NAMES):
        _save_case(data / "TD" / name, generate_subject(k, Label.CONTROL, sc))
    seen = datetime(2026, 9, 12).timestamp()
    for k, name in enumerate(ASD_NAMES):
        transcript = _save_case(data / "ASD" / name, generate_subject(100 + k, Label.ASD, sc))
        if k < 6:
            os.utime(transcript, (seen, seen))
    _write_maps(ids, td=[(n, f"P{len(TD_NAMES) - k:02d}") for k, n in enumerate(TD_NAMES)],
                asd=[(n, f"A{len(ASD_NAMES) - k:02d}") for k, n in enumerate(ASD_NAMES)],
                excluded=[("TDX01", "20250199-假名子")])     # matches no folder
    return data, ids


def _normalised(texts):
    """Every instrument measures normalised sessions: no punctuation is left in what they get."""
    assert not any(p in t for t in texts for p in PUNCTUATION), "an instrument got raw text"


FAKE_ENVIRONMENT = {"packages": {"torch": "9.9"}, "cuda_device": "Test GPU",
                    "crepe": {"model": "full", "batch_size": 1024, "device": "cuda",
                              "torchcrepe_version": "0.0.24"}}


def _fake_tools(f0_fail=("P02", "A03")):
    """Instruments with known answers, keyed by pseudonym: split halves that agree (r = 1), an
    F0 check that fails for ``f0_fail``, child measures that grow with the id's number, and a
    tracking-failure ratio the same for everyone. Each checks that it gets normalised text."""
    def number(sid):
        return int(sid[1:])

    def words_of(texts):
        _normalised(texts)
        return _chars(texts)

    def halves(s):
        _normalised([t.text for t in s.session.turns])
        v = number(s.subject_id) / 10.0
        return {"b7_mean_pairwise_cosdist": (v, v + 0.01), "b_level_rel_examiner_db": (-v, 0.02 - v)}

    def f0(s):
        _normalised([t.text for t in s.session.turns])
        bad = s.subject_id in f0_fail
        return {"agree_pyin": 0.40 if bad else 0.97, "agree_praat": 0.96, "octave_pyin": 0.01,
                "octave_praat": 0.0, "n_frames": 300}

    def child_rows(corpus):
        _normalised([t.text for s in corpus.subjects for t in s.session.turns])
        return [{"c_exaggerated_utterance_ratio": 0.01 * number(s.subject_id),
                 "b7_mean_pairwise_cosdist": 0.2 + 0.01 * number(s.subject_id),
                 "c_pitch_range_st_median": 4.0 + 0.1 * number(s.subject_id),
                 "b_level_rel_examiner_db": -3.0 + 0.2 * number(s.subject_id),
                 "c_f0_tracking_failure_ratio": 0.1} for s in corpus.subjects]

    return Tools(words_of=words_of, halves=halves, f0=f0, child_rows=child_rows,
                 environment=lambda: FAKE_ENVIRONMENT)


def test_no_log_line_error_or_report_names_a_session(cfg, rules, saved_cohorts, tmp_git_repo,
                                                    tmp_path, capsys):
    data, ids = saved_cohorts
    c = _with_mil(cfg, repeats=1, seeds=1, n_permutations=2, epochs=10, n_boot=50)
    c["realdata"].update(root=str(data), exclude_cases=[])
    out, lines = tmp_path / "outputs", []
    kw = dict(project_dir=tmp_git_repo, id_root=ids, out_dir=out, cache_dir=tmp_path / "cache",
              log=lines.append, today="2026-11-02")
    checks = mil_checks(c, rules, tools=_fake_tools(), **kw)
    assert checks["n_sessions"] == 6 and checks["cohort"] == "TD"
    # R32: the hash of the code and settings rides in the JSON and in the outcome to be recorded
    assert set(checks["proposed_outcome"]) == {"excluded", "source", "date", "code_hash"}
    assert checks["code_hash"] == checks["proposed_outcome"]["code_hash"] == mil.code_hash(
        tmp_git_repo)
    assert f'code_hash: "{checks["code_hash"]}"' in (out / "mil_checks.md").read_text(
        encoding="utf-8")
    res = mil_run(c, _outcome(rules, checks["proposed_outcome"]), tools=_fake_tools(), **kw)
    printed = capsys.readouterr()
    written = {p.name: p.read_text(encoding="utf-8") for p in out.iterdir()}
    # F9: the run's label is in the names of what it writes, so a later run cannot replace them
    names = mil.output_files(10)
    assert names == {"results": "mil_results_prelim_asd10.json",
                     "report": "mil_report_prelim_asd10.md",
                     "primary": "mil_attention_prelim_asd10.md",
                     "transcript_only": "mil_attention_transcript_only_prelim_asd10.md"}
    assert set(written) == {"mil_checks.md", "mil_checks.json", *names.values()}
    assert lines
    for text in [*lines, printed.out, printed.err, *written.values()]:
        assert "假名" not in text

    # children by pseudonym, TD first; exposure read from the transcripts' dates
    assert res["children"]["ids"] == [f"P{k:02d}" for k in range(1, 7)] + [
        f"A{k:02d}" for k in range(1, 11)]
    assert res["children"]["exposed"] == [False] * 10 + [True] * 6
    assert res["counts"] == {"td": 6, "asd": 10, "asd_exposed": 6, "asd_unexposed": 4}
    assert res["pitch_blanked"] == ["P02", "A03"]
    assert res["run_label"] == "初步（ASD n=10）"
    assert res["head"] == _git(tmp_git_repo, "rev-parse", "HEAD").strip()
    assert res["variants"]["unexposed_only"]["status"] == "not run (4 unexposed ASD < 10)"
    final = json.loads(written[names["results"]])
    assert final["run_label"] == res["run_label"] and final["complete"] is True
    assert final["files"] == names
    tops = {cid for cid, top in res["final"]["primary"]["top_segment"].items()
            if top["segment"] is not None}
    assert {b.split()[0] for b in written[names["primary"]].split("\n## ")[1:]} == tops
    said = [line for line in written[names["primary"]].splitlines() if line.startswith("- `")]
    assert said
    _normalised(said)                                   # the transcript lines are the text the measures read

    # F7: what the run ran on, in both JSON files: the profile, the Python executable, the
    # packages and the GPU as the instruments report them, and the hashes of the two private ID
    # maps and of the excluded-children table (no more than the hashes), which decide who is in
    # the run and the order of the folds
    maps = {f: hashlib.sha256((ids / f).read_bytes()).hexdigest()
            for f in ("_private_id_map.csv", "_private_id_map_asd.csv", EXCLUDED_ID_MAP)}
    for name in ("mil_checks.json", names["results"]):
        doc = json.loads(written[name])
        env = doc["environment"]
        assert doc["profile"] == c["models"]["profile"]
        assert env["python"] == sys.executable
        assert env["packages"] == FAKE_ENVIRONMENT["packages"]
        assert env["cuda_device"] == "Test GPU"
        assert env["id_map_sha256"] == maps
    # R33: the CREPE model, batch size, device and torchcrepe version of the labelled run
    assert final["environment"]["crepe"] == FAKE_ENVIRONMENT["crepe"]

    # F9: the same label again would overwrite them, which is refused, before any extraction
    # (the instruments here fail the test if they are asked to measure anything)
    with pytest.raises(RunRefused, match=re.escape(names["results"])) as refused:
        mil_run(c, _outcome(rules, checks["proposed_outcome"]), tools=_untouched_tools(), **kw)
    assert "假名" not in str(refused.value)
    assert {p.name: p.read_text(encoding="utf-8") for p in out.iterdir()} == written


def test_load_arm_resolves_exclude_cases_ids_through_the_table_under_id_root(
        cfg, saved_cohorts, tmp_path):
    """exclude_cases holds ids and the table under id_root turns them into folders. The sixth TD
    session is in no TD map, so it is loaded, and stops the run, only if the id fails to exclude
    it."""
    data, _ = saved_cohorts
    ids = tmp_path / "ids"
    ids.mkdir()
    _write_maps(ids, td=[(n, f"P{k:02d}") for k, n in enumerate(TD_NAMES[:5], 1)],
                excluded=[("TDX01", TD_NAMES[5]), ("TDX02", "20250199-假名子")])
    c = copy.deepcopy(cfg)
    c["realdata"].update(root=str(data), exclude_cases=["TDX01"])
    lines = []
    arm = mil.load_arm(c, "TD", ids, log=lines.append)
    assert arm.ids == tuple(f"P{k:02d}" for k in range(1, 6))
    assert c["realdata"]["exclude_cases"] == ["TDX01"]          # the config never holds a name
    assert lines and not any("假名" in x for x in lines)
    c["realdata"]["exclude_cases"] = []
    with pytest.raises(KeyError) as unmapped:                   # named by position only
        mil.load_arm(c, "TD", ids, log=lines.append)
    assert "假名" not in str(unmapped.value)
    c["realdata"]["exclude_cases"] = ["TDX09"]                  # an id the table lacks
    with pytest.raises(ValueError) as missing:
        mil.load_arm(c, "TD", ids, log=lines.append)
    assert "TDX09" in str(missing.value) and "假名" not in str(missing.value)


def test_the_checks_refuse_when_the_code_changes_while_they_run(cfg, rules, saved_cohorts,
                                                               tmp_git_repo, tmp_path):
    # R32: an outcome decided while the code was edited would be tied to neither version
    data, ids = saved_cohorts
    c = copy.deepcopy(cfg)
    c["realdata"].update(root=str(data), exclude_cases=[])
    tools = _fake_tools()

    def halves_and_edit(s):
        (tmp_git_repo / "src/asdpipe/m.py").write_text("x = 2\n", encoding="utf-8")
        return tools.halves(s)

    with pytest.raises(RunRefused, match="while the checks were running"):
        mil_checks(c, rules, project_dir=tmp_git_repo, id_root=ids, out_dir=tmp_path / "out",
                   tools=dataclasses.replace(tools, halves=halves_and_edit),
                   cache_dir=tmp_path / "cache", log=lambda m: None)
    assert not (tmp_path / "out" / "mil_checks.json").exists()


def test_mil_run_refuses_unless_the_dated_exposures_are_the_six_of_d10(
        cfg, rules, saved_cohorts, tmp_git_repo, tmp_path, monkeypatch):
    # The cohorts hold six exposed ASD sessions; were D10's count another, a transcript's date
    # would have changed and an exposed child could pass for an unexposed one.
    data, ids = saved_cohorts
    c = copy.deepcopy(cfg)
    c["realdata"].update(root=str(data), exclude_cases=[])
    monkeypatch.setattr(mil, "EXPOSED_ASD", 7)
    with pytest.raises(RunRefused, match="D10"):
        mil_run(c, _recorded(rules, tmp_git_repo), project_dir=tmp_git_repo, id_root=ids,
                out_dir=tmp_path / "out", tools=_untouched_tools(), cache_dir=tmp_path / "cache",
                log=lambda m: None)
    assert not (tmp_path / "out").exists()


def test_errors_while_loading_name_the_cohort_and_position_only(cfg, rules, tmp_path,
                                                                monkeypatch, tmp_git_repo):
    data, ids = tmp_path / "data", tmp_path / "ids"
    ids.mkdir()
    sc = SynthConfig(turns_per_subject=10, seed=3)
    for k in range(2):
        _save_case(data / "TD" / TD_NAMES[k], generate_subject(k, Label.CONTROL, sc))
    c = copy.deepcopy(cfg)
    c["realdata"].update(root=str(data), exclude_cases=[])

    def run():
        mil_checks(c, rules, project_dir=tmp_git_repo, id_root=ids, out_dir=tmp_path / "out",
                   tools=_untouched_tools(), cache_dir=tmp_path / "cache", log=lambda m: None)

    def shown(err):
        return "".join(traceback.format_exception(err))

    _write_maps(ids, td=[(TD_NAMES[0], "P01")])                   # the second has no pseudonym
    with pytest.raises(KeyError) as e:
        run()
    assert e.value.args == ("TD case #2 has no pseudonym",) and "假名" not in shown(e.value)

    _write_maps(ids, td=[(TD_NAMES[0], "P01"), (TD_NAMES[1], "P02")])
    transcript = data / "TD" / TD_NAMES[1] / f"{TD_NAMES[1]}_edited.json"
    good = transcript.read_text(encoding="utf-8")
    transcript.write_text("{ not json", encoding="utf-8")         # mapped, but unreadable
    with pytest.raises(RuntimeError) as e:
        run()
    assert "TD case #2 (P02) could not be loaded" in str(e.value) and "假名" not in shown(e.value)

    # the child's speaker label is a name: no child turn is left, and the case is unusable. The
    # loading report lists the unmapped labels; the error gives counts only.
    segments = json.loads(good)["segments"]
    named = [{**s, "speaker": "小乙乙" if s["speaker"] == "兒童" else s["speaker"]} for s in segments]
    transcript.write_text(json.dumps({"metadata": {}, "segments": named}, ensure_ascii=False),
                          encoding="utf-8")
    with pytest.raises(RuntimeError) as e:
        run()
    assert str(e.value).startswith("TD case #2 (P02) is unusable: ")
    assert "小乙乙" not in shown(e.value) and "假名" not in shown(e.value)

    # a value from the transcript in an error message (float() quotes what it could not read)
    broken = [{**segments[0], "start": "小乙乙"}, *segments[1:]]
    transcript.write_text(json.dumps({"metadata": {}, "segments": broken}, ensure_ascii=False),
                          encoding="utf-8")
    with pytest.raises(RuntimeError) as e:
        run()
    assert str(e.value) == "TD case #2 (P02) could not be loaded: ValueError"
    assert "小乙乙" not in shown(e.value)

    transcript.write_text(good, encoding="utf-8")
    _write_maps(ids, td=[(TD_NAMES[0], "P01"), (TD_NAMES[1], "P02"), ("假名不在", "P03")])
    with pytest.raises(RuntimeError) as e:                        # a mapped TD session is missing
        run()
    assert "P03" in str(e.value) and "假名" not in shown(e.value)

    # an error whose message holds the case's path, as a missing file's does: only the type and
    # the operating system's own text are said, and the original exception is not chained
    def lost_audio(case, *args, **kw):
        raise FileNotFoundError(2, "No such file or directory", str(case.audio))

    monkeypatch.setattr(mil, "load_subject", lost_audio)
    _write_maps(ids, td=[(TD_NAMES[0], "P01"), (TD_NAMES[1], "P02")])
    with pytest.raises(RuntimeError) as e:
        run()
    assert str(e.value) == ("TD case #1 (P01) could not be loaded: FileNotFoundError "
                            "(No such file or directory)")
    assert "假名" not in shown(e.value)


# ---------------------------------------------------------------- F2: the model profile
class _Stop(Exception):
    """Ends a command at the point where it would load its first model."""


def _smoke_cfg(cfg):
    """The committed default: models.profile is smoke, whose CREPE is "tiny"."""
    c = copy.deepcopy(cfg)
    c["models"]["profile"] = "smoke"
    return c


@pytest.mark.parametrize("command", ["cmd_mil_checks", "cmd_mil_run"])
def test_the_mil_commands_set_the_real_profile_themselves(monkeypatch, cfg, rules, command):
    from asdpipe.cli import main as cli

    monkeypatch.delenv("ASDPIPE_PROFILE", raising=False)
    for step in ("freeze_guard", "outcome_exclusions", "check_code_hash"):
        monkeypatch.setattr(mil, step, lambda *a, **k: None)
    seen = []

    def backends(c):
        seen.append(c["models"]["profile"])
        raise _Stop

    monkeypatch.setattr(cli, "_backends", backends)
    with pytest.raises(_Stop):
        getattr(cli, command)(_smoke_cfg(cfg), rules)
    assert seen == ["real"]                       # not the committed smoke profile
    monkeypatch.setenv("ASDPIPE_PROFILE", "real")           # saying so is allowed
    with pytest.raises(_Stop):
        getattr(cli, command)(_smoke_cfg(cfg), rules)
    assert seen == ["real", "real"]


@pytest.mark.parametrize("command, tag", [("cmd_mil_checks", "mil-checks"),
                                          ("cmd_mil_run", "mil-run")])
@pytest.mark.parametrize("asked", ["smoke", "Real"])
def test_the_mil_commands_refuse_another_profile_in_the_environment(
        monkeypatch, cfg, rules, capsys, command, tag, asked):
    # ASDPIPE_PROFILE used to be able to swap the pitch tracker (smoke's CREPE is "tiny") for both
    # commands, outside the freeze, and nothing recorded it
    from asdpipe.cli import main as cli

    monkeypatch.setenv("ASDPIPE_PROFILE", asked)
    monkeypatch.setattr(mil, "freeze_guard", lambda p: pytest.fail("the freeze came first"))
    monkeypatch.setattr(cli, "_backends", lambda c: pytest.fail("a model was loaded"))
    assert getattr(cli, command)(cfg, rules) == 1
    err = capsys.readouterr().err
    assert f"[{tag}] refused" in err and "ASDPIPE_PROFILE" in err and "real" in err


# ---------------------------------------------------------------- R33: features computed fresh
def _crepe_backends(tmp_path, device="cuda", **kw):
    """A ``Backends`` that loads no model: the ids are never fetched until a model is used."""
    from asdpipe.features.backends import Backends

    return Backends(sbert_id="sbert", ecapa_id="ecapa", ckip_ws_id="ckip", crepe_model="full",
                    device=device, cache_dir=str(tmp_path / "cache"), **kw)


def test_a_fresh_f0_cache_does_not_serve_the_tracks_of_an_earlier_run(tmp_path, monkeypatch):
    from pathlib import Path

    from asdpipe.features.backends import Backends

    calls = []

    def track(self, audio, sr, fmin, fmax, hop_s):             # what CREPE would have computed
        calls.append(1)
        return np.full(5, float(len(calls))), np.ones(5)

    monkeypatch.setattr(Backends, "_track_f0_uncached", track)
    b = _crepe_backends(tmp_path)
    audio = np.linspace(-1.0, 1.0, 16000, dtype=np.float32)
    args = (audio, 16000, 100.0, 600.0, 0.01)
    assert b.track_f0(*args)[0][0] == 1.0                     # an earlier run: cached in cache/f0
    assert b.track_f0(*args)[0][0] == 1.0 and len(calls) == 1  # which is how the cache works
    fresh = b.use_fresh_f0_cache(tmp_path / "mil_run")
    assert (tmp_path / "mil_run") in Path(fresh).parents
    assert os.listdir(fresh) == []                            # created now, so empty
    assert b.track_f0(*args)[0][0] == 2.0 and len(calls) == 2  # computed again, in this run
    assert b.track_f0(*args)[0][0] == 2.0 and len(calls) == 2  # and kept for the rest of it
    assert len(os.listdir(fresh)) == 1
    assert len(os.listdir(tmp_path / "cache" / "f0")) == 1     # the earlier run's track is untouched


def test_a_fresh_f0_cache_is_never_a_directory_that_already_exists(tmp_path, monkeypatch):
    monkeypatch.setattr("time.strftime", lambda fmt: "STAMP")
    b = _crepe_backends(tmp_path)
    b.use_fresh_f0_cache(tmp_path / "mil_run")
    with pytest.raises(FileExistsError):                      # it could hold another run's tracks
        b.use_fresh_f0_cache(tmp_path / "mil_run")


def test_the_crepe_settings_are_the_ones_it_runs_with(tmp_path, monkeypatch):
    import importlib.metadata

    import torch
    import torchcrepe

    seen = {}

    def predict(x, sr, **kw):
        seen.update(kw)
        return torch.zeros(1, 12), torch.ones(1, 12)

    monkeypatch.setattr(torchcrepe, "predict", predict)
    b = _crepe_backends(tmp_path)
    b._track_f0_uncached(np.zeros(16000, dtype=np.float32), 16000, 100.0, 600.0, 0.01)
    record = b.crepe_settings()
    assert record == {"model": "full", "batch_size": 1024, "device": "cuda",
                      "torchcrepe_version": importlib.metadata.version("torchcrepe")}
    # what is recorded is what CREPE was called with, and the batch size is the one that was
    # always there (ruling R33: not changed)
    assert (seen["model"], seen["batch_size"], seen["device"]) == ("full", 1024, "cuda")


def _rows_for(corpus):
    return [SubjectRow(subject_id=s.subject_id, label=Label.CONTROL, b_pooled={}, c_pooled={},
                       e_pooled={}, c_sequence=np.zeros((0, 8)), egemaps_functionals=np.zeros(88),
                       examiner_probe={}, ados={}, latents={}, quality={})
            for s in corpus.subjects]


def test_the_labelled_run_needs_cuda_and_extracts_without_the_feature_cache(
        cfg, tmp_path, monkeypatch):
    import types

    asked = []

    def extract(corpus, cfg, backends, **kw):
        asked.append(kw)
        return _rows_for(corpus)

    monkeypatch.setattr(mil, "extract_corpus", extract)
    corpus = types.SimpleNamespace(subjects=[types.SimpleNamespace(subject_id="P01")])
    aligner = object()                                        # extract is a stand-in: never used
    with pytest.raises(RunRefused, match="CUDA"):             # CREPE on the CPU is another device
        mil.real_tools(cfg, _crepe_backends(tmp_path, device="cpu"), aligner,
                       fresh_cache=tmp_path / "mil_run")
    assert not (tmp_path / "mil_run").exists()                # refused before anything was made
    b = _crepe_backends(tmp_path)
    tools = mil.real_tools(cfg, b, aligner, fresh_cache=tmp_path / "mil_run")
    assert b.f0_cache_dir.startswith(str(tmp_path / "mil_run")) and os.listdir(b.f0_cache_dir) == []
    tools.child_rows(corpus)
    assert asked[-1]["use_cache"] is False                    # neither read nor written
    # without fresh_cache (the checks, tests) the usual caches stay
    plain = _crepe_backends(tmp_path, device="cpu")
    mil.real_tools(cfg, plain, aligner).child_rows(corpus)
    assert asked[-1].get("use_cache", True) is True and plain.f0_cache_dir is None


def test_the_real_tools_describe_what_they_ran_on(cfg, tmp_path, monkeypatch):
    import torch
    from asdpipe.features.backends import Backends

    monkeypatch.setattr(Backends, "versions", lambda self: {"torch": "9.9"})
    env = mil.real_tools(cfg, _crepe_backends(tmp_path)).environment()
    assert env["packages"] == {"torch": "9.9"}
    assert env["crepe"] == _crepe_backends(tmp_path).crepe_settings()
    assert env["cuda_device"] == (torch.cuda.get_device_name() if torch.cuda.is_available()
                                  else None)


def test_the_cli_runs_only_on_cuda_with_a_fresh_cache(monkeypatch, cfg, rules, tmp_path, capsys):
    from asdpipe.cli import main as cli

    monkeypatch.delenv("ASDPIPE_PROFILE", raising=False)
    for step in ("freeze_guard", "outcome_exclusions", "check_code_hash"):
        monkeypatch.setattr(mil, step, lambda *a, **k: None)
    monkeypatch.setattr(mil, "mil_run", lambda *a, **k: pytest.fail("the run started"))
    monkeypatch.setattr(cli, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(cli, "_backends", lambda c: _crepe_backends(tmp_path, device="cpu"))
    assert cli.cmd_mil_run(cfg, rules) == 1
    assert "CUDA" in capsys.readouterr().err
    cuda = _crepe_backends(tmp_path)
    monkeypatch.setattr(cli, "_backends", lambda c: cuda)
    monkeypatch.setattr(mil, "mil_run", lambda *a, **k: (_ for _ in ()).throw(_Stop()))
    with pytest.raises(_Stop):                                # past the checks, into the run
        cli.cmd_mil_run(cfg, rules)
    assert cuda.f0_cache_dir.startswith(str(tmp_path / "cache" / "mil_run"))


# ---------------------------------------------------------------- F4: the ASD sample is pinned
@pytest.mark.parametrize("n_asd, n_unexposed", [(10, 4), (20, 14)])
def test_the_two_preregistered_asd_samples_are_allowed(n_asd, n_unexposed):
    mil.check_asd_sample(n_asd, n_unexposed)


@pytest.mark.parametrize("n_asd, n_unexposed", [
    (8, 2), (9, 3), (11, 5), (12, 6), (19, 13), (21, 15),       # a stray or a missing session
    (10, 3), (10, 5), (10, 10), (20, 13), (20, 15), (0, 0)])    # the right total, the wrong split
def test_any_other_asd_sample_is_refused_and_the_message_holds_counts_only(n_asd, n_unexposed):
    with pytest.raises(RunRefused) as e:
        mil.check_asd_sample(n_asd, n_unexposed)
    assert f"{n_asd} ASD" in str(e.value) and f"{n_unexposed} of them unexposed" in str(e.value)


def test_mil_run_refuses_a_stray_or_missing_asd_session(cfg, rules, saved_cohorts, tmp_git_repo,
                                                        tmp_path):
    # discover_cases takes any *.json of a case folder for its transcript, so a diarisation file
    # left in an untranscribed folder would load as an unexposed case; 11 or 9 are not 10 or 20
    data, ids = saved_cohorts
    c = copy.deepcopy(cfg)
    c["realdata"].update(root=str(data), exclude_cases=[ASD_NAMES[-1], ASD_NAMES[-2]])
    with pytest.raises(RunRefused, match="8 ASD sessions, 2 of them unexposed") as e:
        mil_run(c, _recorded(rules, tmp_git_repo), project_dir=tmp_git_repo, id_root=ids,
                out_dir=tmp_path / "out", tools=_untouched_tools(), cache_dir=tmp_path / "cache",
                log=lambda m: None)
    assert "假名" not in str(e.value) and not (tmp_path / "out").exists()


# ---------------------------------------------------------------- F8: a measure nobody has
def test_a_child_measure_that_no_child_has_a_value_for_is_refused(rules):
    # the extraction renamed (or lost) one of the four: _value turned that into NaN silently
    lost = "b_level_rel_examiner_db"
    kids = [_kid(i, i % 2) for i in range(10)]
    for kid in kids:
        del kid.values[lost]
    with pytest.raises(RunRefused, match=lost):
        prepare_children(kids, rules, excluded=[])
    # not when the frozen checks, or the D11 gate, had already taken the measure out
    assert lost not in prepare_children(kids, rules, excluded=[lost]).child_names
    r = copy.deepcopy(rules)
    r["convention_invariance_gate"]["outcome"]["excluded_from_confirmatory"] = [
        *confirmatory_exclusions(rules), lost]
    assert lost not in prepare_children(kids, r, excluded=[]).child_names


def test_a_measure_some_children_lack_is_not_a_missing_measure(rules):
    kids = [_kid(i, i % 2) for i in range(10)]
    del kids[3].values["b_level_rel_examiner_db"]
    prepare_children(kids, rules, excluded=[])
    # the pitch values every session's F0 check blanked are NaN only after the check, which is
    # the F0 check's business (and the group rule's), not a renamed measure
    prep = prepare_children([_kid(i, i % 2, f0_ok=False) for i in range(10)], rules, excluded=[])
    assert len(prep.blanked) == 10


# ---------------------------------------------------------------- F9: the output names
def test_the_output_names_carry_the_label_of_the_run():
    prelim, final = mil.output_files(10), mil.output_files(20)
    assert prelim["results"] == "mil_results_prelim_asd10.json"
    assert final == {"results": "mil_results_final.json", "report": "mil_report_final.md",
                     "primary": "mil_attention_final.md",
                     "transcript_only": "mil_attention_transcript_only_final.md"}
    assert not set(prelim.values()) & set(final.values())


# ---------------------------------------------------------------- F6: results as they finish
def _canned_evaluate(bags, mcfg, cv, n_perm, seed, C):
    n = len(bags.ids)
    one = {"auc": 0.7, "ci": (0.5, 0.9), "p": 0.04, "null": np.full(n_perm, 0.5),
           "oof_mean": np.full(n, 0.5)}
    return {"mil": {**one, "gates_by_fold": np.zeros((1, 5, len(bags.seg_names) +
                                                      len(bags.child_names)))}, "logreg": dict(one)}


def test_analyse_hands_over_each_variant_as_it_finishes(small_cfg, rules, bags_by_len,
                                                        monkeypatch):
    monkeypatch.setattr(mil, "evaluate", _canned_evaluate)
    monkeypatch.setattr(mil, "_final_model", lambda *a, **k: {"top_segment": {}})
    seen = []
    res = analyse(bags_by_len, small_cfg, rules, log=lambda m: None,
                  on_progress=lambda part: seen.append((list(part["variants"]),
                                                        list(part["final"]))))
    assert [len(v) for v, _ in seen] == [*range(1, len(res["variants"]) + 1), len(res["variants"]),
                                         len(res["variants"])]
    assert seen[0][0] == ["primary"] and seen[-1][1] == ["primary", "transcript_only"]
    assert seen[-1][0] == list(res["variants"])                # every variant, run or not


def test_a_crash_late_in_the_run_leaves_the_finished_variants_in_outputs(
        cfg, rules, saved_cohorts, tmp_git_repo, tmp_path, monkeypatch):
    data, ids = saved_cohorts
    c = _with_mil(cfg, repeats=1, seeds=1, n_permutations=2, epochs=10, n_boot=50)
    c["realdata"].update(root=str(data), exclude_cases=[])
    out = tmp_path / "outputs"
    kw = dict(project_dir=tmp_git_repo, id_root=ids, out_dir=out, cache_dir=tmp_path / "cache",
              log=lambda m: None, today="2026-11-02")
    outcome = _recorded(rules, tmp_git_repo)
    real_evaluate, calls = mil.evaluate, []

    def crashing(*args, **kw):
        calls.append(1)
        if len(calls) == 3:                                  # the third variant: seg_len_3
            raise _Stop("out of memory, say")
        return real_evaluate(*args, **kw)

    monkeypatch.setattr(mil, "evaluate", crashing)
    with pytest.raises(_Stop):
        mil_run(c, outcome, tools=_fake_tools(), **kw)
    partial = json.loads((out / mil.output_files(10)["results"]).read_text(encoding="utf-8"))
    assert partial["complete"] is False and partial["run_label"] == "初步（ASD n=10）"
    assert list(partial["variants"]) == ["primary", "transcript_only"]
    assert 0.0 <= partial["variants"]["primary"]["mil"]["auc"] <= 1.0
    assert partial["counts"]["asd"] == 10 and partial["environment"]["python"] == sys.executable
    assert not (out / mil.output_files(10)["report"]).exists()   # the report waits for the end
    with pytest.raises(RunRefused, match="mil_results_prelim_asd10.json"):   # F9: not replaced
        mil_run(c, outcome, tools=_untouched_tools(), **kw)
