"""realdata.exclude_cases holds ids; the folder names live only in a gitignored table.

Everything here is synthetic: folders named like the real ones ("20250101-1壹") and a table
written to a temp directory. Nothing reads or writes the real table.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import re
import traceback

import pytest

from asdpipe.eval import audio_audit
from asdpipe.eval.audio_audit import RecordingAudit, audit_cohort
from asdpipe.mil.run import id_map_hashes
from asdpipe.realdata import (EXCLUDED_ID_MAP, discover_cases, load_cohort,
                              resolve_excluded)

KEPT, DROPPED, ALSO_KEPT = "20250101-1壹", "20250102-2貳", "20250103-3參"
SEGMENTS = [{"sentence_id": 0.0, "speaker": "評估者", "start": 0.0, "end": 1.0, "text": "你喜歡什麼"},
            {"sentence_id": 1.0, "speaker": "兒童", "start": 1.2, "end": 2.0, "text": "蘋果"}]


def _cohort(root, cohort="TD"):
    """Three synthetic case folders, each with an audio file and a transcript."""
    for name in (KEPT, DROPPED, ALSO_KEPT):
        d = root / cohort / name
        d.mkdir(parents=True)
        (d / "a.mp3").write_bytes(b"")
        (d / "a_edited.json").write_text(
            json.dumps({"metadata": {}, "segments": SEGMENTS}, ensure_ascii=False),
            encoding="utf-8")
    return root


def _table(root, rows, header=("xid", "session"), encoding="utf-8"):
    """The excluded-children table in its real layout: (id, folder-name fragment) per row."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / EXCLUDED_ID_MAP).open("w", encoding=encoding, newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return root


def _shows_no_name(exc_info):
    """Neither the message nor the whole traceback (chained causes included) holds a fragment."""
    text = str(exc_info.value) + "".join(traceback.format_exception(exc_info.value))
    return not any(x in text for x in (KEPT, DROPPED, ALSO_KEPT, "壹", "貳", "參"))


def test_an_id_excludes_the_folder_its_row_names(tmp_path):
    data, ids = _cohort(tmp_path / "data"), _table(tmp_path / "ids", [("TDX01", "貳")])
    got = discover_cases(data, ["TDX01"], cohort="TD", id_root=ids)
    assert [c.case_id for c in got] == [KEPT, ALSO_KEPT]
    # several ids, listed in any order, each its own row
    ids = _table(tmp_path / "ids2", [("TDX02", "壹"), ("TDX01", "貳")])
    assert [c.case_id for c in discover_cases(data, ["TDX01", "TDX02"], cohort="TD",
                                              id_root=ids)] == [ALSO_KEPT]


def test_resolve_excluded_keeps_the_order_and_passes_plain_entries_through(tmp_path):
    ids = _table(tmp_path, [("TDX01", "貳"), ("TDX02", "參")])
    assert resolve_excluded(["TDX02", "壹", "TDX01"], ids) == ["參", "壹", "貳"]


def test_plain_entries_work_as_before_and_need_no_table(tmp_path):
    data = _cohort(tmp_path / "data")
    nowhere = tmp_path / "no_table_here"            # not even a directory: it is never opened
    assert [c.case_id for c in discover_cases(data, ["貳"], cohort="TD", id_root=nowhere)] == [
        KEPT, ALSO_KEPT]
    assert len(discover_cases(data, cohort="TD", id_root=nowhere)) == 3
    # a mixture: the plain entry is a fragment, the id comes from the table
    ids = _table(tmp_path / "ids", [("TDX01", "貳")])
    assert [c.case_id for c in discover_cases(data, ["壹", "TDX01"], cohort="TD",
                                              id_root=ids)] == [ALSO_KEPT]


def test_a_missing_id_raises_and_names_only_the_id(tmp_path):
    data, ids = _cohort(tmp_path / "data"), _table(tmp_path / "ids", [("TDX01", "貳")])
    with pytest.raises(ValueError) as e:
        discover_cases(data, ["TDX01", "TDX07"], cohort="TD", id_root=ids)
    assert "TDX07" in str(e.value) and "TDX01" not in str(e.value)
    assert EXCLUDED_ID_MAP in str(e.value)
    assert _shows_no_name(e)
    # the same when the file is not there at all (a fresh clone has no table)
    with pytest.raises(ValueError) as e:
        discover_cases(data, ["TDX01"], cohort="TD", id_root=tmp_path / "empty")
    assert "TDX01" in str(e.value) and _shows_no_name(e)


@pytest.mark.parametrize("header, rows", [
    (("id", "session"), [("TDX01", "貳")]),                      # the column is called something else
    (("xid", "session"), [("TDX01", "")]),                      # an empty fragment would match every folder
    (("xid", "session"), [("貳", "貳")]),                        # the id column holds a name
    (("xid", "session"), [("TDX01", "貳"), ("TDX01", "參")]),    # one id, two sessions
])
def test_a_malformed_table_is_refused_without_showing_a_cell(tmp_path, header, rows):
    ids = _table(tmp_path, rows, header=header)
    with pytest.raises(ValueError) as e:
        resolve_excluded(["TDX01"], ids)
    assert EXCLUDED_ID_MAP in str(e.value) and _shows_no_name(e)


def test_a_table_saved_with_a_bom_reads(tmp_path):
    ids = _table(tmp_path, [("TDX01", "貳")], encoding="utf-8-sig")
    assert resolve_excluded(["TDX01"], ids) == ["貳"]


def test_no_name_reaches_the_config_or_the_loaded_cohort(tmp_path, cfg):
    data, ids = _cohort(tmp_path / "data"), _table(tmp_path / "ids", [("TDX01", "貳")])
    c = copy.deepcopy(cfg)
    c["realdata"].update(root=str(data), cohort="TD", exclude_cases=["TDX01"])
    before = copy.deepcopy(c)
    corpus, reports = load_cohort(data, c, with_audio=False, id_root=ids)
    assert [s.subject_id for s in corpus.subjects] == [KEPT, ALSO_KEPT]
    assert c == before and c["realdata"]["exclude_cases"] == ["TDX01"]
    assert "貳" not in repr(c) + repr(corpus.config_hash) + repr([r.case_id for r in reports])


def test_the_committed_config_holds_ids_only(cfg):
    entries = cfg["realdata"]["exclude_cases"]
    not_ids = sum(not re.fullmatch(r"TDX\d+", str(x)) for x in entries)
    ok = bool(entries) and not_ids == 0           # a bool, so a failure cannot print the entries
    assert ok, "exclude_cases holds an entry that is not an id"


def test_audit_cohort_resolves_ids_too(tmp_path, cfg, monkeypatch):
    data, ids = _cohort(tmp_path / "data"), _table(tmp_path / "ids", [("TDX01", "貳")])
    audited = []

    def fake_audit(case_id, cohort, path, cfg, backends=None, cache_dir=None):
        audited.append(case_id)
        return RecordingAudit(case_id, cohort)

    monkeypatch.setattr(audio_audit, "audit_recording", fake_audit)
    out = audit_cohort(data, "TD", cfg, exclude=["TDX01"], id_root=ids)
    assert audited == [KEPT, ALSO_KEPT] and [a.case_id for a in out] == audited


def test_the_run_record_hashes_the_excluded_table(tmp_path):
    assert id_map_hashes(tmp_path) == {"_private_id_map.csv": None,
                                       "_private_id_map_asd.csv": None, EXCLUDED_ID_MAP: None}
    _table(tmp_path, [("TDX01", "貳")])
    got = id_map_hashes(tmp_path)
    assert got[EXCLUDED_ID_MAP] == hashlib.sha256(
        (tmp_path / EXCLUDED_ID_MAP).read_bytes()).hexdigest()
    assert got["_private_id_map.csv"] is None
    _table(tmp_path, [("TDX01", "壹")])                 # a changed table is a changed hash
    assert id_map_hashes(tmp_path)[EXCLUDED_ID_MAP] != got[EXCLUDED_ID_MAP]
