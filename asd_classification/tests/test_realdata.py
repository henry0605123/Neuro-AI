"""Real-cohort adapter. Fixtures are written by hand so every expected count is exact."""

from __future__ import annotations

import json

import pytest

from asdpipe.contracts import Speaker
from asdpipe.realdata import (SPEAKER_ROLES, discover_cases, is_question,
                              is_yes_no_question, load_transcript, normalise_speaker)


# ---------------------------------------------------------------- speaker roles
def test_every_role_in_the_table_is_examiner_or_child():
    assert set(SPEAKER_ROLES.values()) == {Speaker.EXAMINER, Speaker.CHILD}


def test_normalise_speaker_covers_all_four_vocabularies():
    for label in ("評估者", "測試者", "SPEAKER_00"):
        assert normalise_speaker(label) is Speaker.EXAMINER
    for label in ("兒童", "小孩", "SPEAKER_01"):
        assert normalise_speaker(label) is Speaker.CHILD


def test_third_party_labels_map_to_none():
    for label in ("個管師", "家長", "廣播", "Unknown", "路人甲"):
        assert normalise_speaker(label) is None


def test_metadata_speaker_mapping_is_applied():
    mapping = {"SPK_A": "評估者", "SPK_B": "兒童"}
    assert normalise_speaker("SPK_A", mapping) is Speaker.EXAMINER
    assert normalise_speaker("SPK_B", mapping) is Speaker.CHILD


# ---------------------------------------------------------------- question detection
@pytest.mark.parametrize("text", [
    "你今天做了什麼?",          # question mark
    "你喜歡嗎",                 # 嗎 tail without a mark
    "你最好朋友叫什麼名字",      # wh word, no punctuation
    "那你這時候都會怎麼辦",
    "你要不要吃",               # X不X, no 嗎 (added 2026-10-15)
    "你想吃嗎我去拿",           # 嗎 followed by more talk
    "那這個呢",                 # final 呢
    "你在幹嘛",
])
def test_question_detection_positives(text):
    assert is_question(text)


@pytest.mark.parametrize("text", [
    "好，換你選五個",
    "對啊",
    "我會帶去跟我同學玩",
    "我們來玩吧",               # a suggestion: final 吧 is no longer a question (D15)
    "好喔",
    "不不不",                   # a run of negations is not X不X
])
def test_question_detection_negatives(text):
    assert not is_question(text)


@pytest.mark.parametrize("text", [
    "你要不要喝水", "這個是不是紅色的", "你有沒有睡午覺", "你會不會自己穿鞋子",
    "我們要刷牙對不對？", "你說這個嗎?", "你吃不吃",
])
def test_yes_no_detection(text):
    assert is_yes_no_question(text)


def test_open_questions_are_not_flagged_as_yes_no():
    for text in ("你今天做了什麼?", "你最好朋友叫什麼名字", "那有什麼壞處?"):
        assert is_question(text) and not is_yes_no_question(text)


def test_question_mark_alone_is_not_required():
    """Punctuation is a per-transcriber habit in this cohort (0%-53% of examiner turns).

    Relying on it alone would make G1's pair count track the transcriber rather than the
    child, which is a confound. The rule therefore unions three cues.
    """
    assert is_question("你最好朋友叫什麼名字")     # no punctuation at all


# ---------------------------------------------------------------- transcript loading
def _write(tmp_path, segments, meta=None):
    case = tmp_path / "20250101-0000000測試童"
    case.mkdir()
    (case / "x.mp3").write_bytes(b"")
    payload = {"metadata": meta or {}, "segments": segments}
    p = case / "20250101-0000000測試童_edited.json"
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return p


def _seg(i, spk, a, b, text="說話內容"):
    return {"sentence_id": float(i), "speaker": spk, "start": a, "end": b, "text": text}


def test_clean_transcript_keeps_everything(tmp_path, cfg):
    segs = [_seg(0, "評估者", 0.0, 1.0, "你今天做了什麼?"),
            _seg(1, "兒童", 1.5, 2.5, "去公園"),
            _seg(2, "評估者", 3.0, 4.0, "好玩嗎"),
            _seg(3, "兒童", 4.5, 5.0, "好玩")]
    turns, rep = load_transcript(_write(tmp_path, segs), cfg)
    assert rep.n_raw == 4 and rep.n_kept == 4 and rep.n_dropped_overlap == 0
    assert [t.speaker for t in turns] == [Speaker.EXAMINER, Speaker.CHILD,
                                          Speaker.EXAMINER, Speaker.CHILD]
    assert turns[0].is_question and not turns[0].is_yes_no
    assert turns[2].is_yes_no
    assert rep.n_qa_pairs == 2 and rep.n_open_qa_pairs == 1


def test_overlapping_pair_drops_both_segments(tmp_path, cfg):
    """Segments 1 and 2 overlap by 0.4 s -> both go, leaving 2 of 4."""
    segs = [_seg(0, "評估者", 0.0, 1.0),
            _seg(1, "兒童", 1.5, 3.0),
            _seg(2, "評估者", 2.6, 4.0),
            _seg(3, "兒童", 4.5, 5.0)]
    turns, rep = load_transcript(_write(tmp_path, segs), cfg)
    assert rep.n_overlap_pairs == 1
    assert rep.n_dropped_overlap == 2
    assert rep.n_kept == 2
    assert rep.overlap_seconds == pytest.approx(0.4, abs=1e-9)


def test_overlap_below_tolerance_is_kept(tmp_path, cfg):
    """A 10 ms nick is boundary rounding, not simultaneous speech (tolerance is 20 ms)."""
    segs = [_seg(0, "評估者", 0.0, 1.0),
            _seg(1, "兒童", 0.99, 2.0)]
    _, rep = load_transcript(_write(tmp_path, segs), cfg)
    assert rep.n_overlap_pairs == 0 and rep.n_kept == 2


def test_third_party_turns_are_dropped(tmp_path, cfg):
    segs = [_seg(0, "評估者", 0.0, 1.0),
            _seg(1, "個管師", 1.2, 2.0),
            _seg(2, "兒童", 2.2, 3.0)]
    turns, rep = load_transcript(_write(tmp_path, segs), cfg)
    assert rep.n_third_party == 1 and rep.n_kept == 2
    assert all(t.speaker is not Speaker.OVERLAP for t in turns)


def test_segments_are_sorted_and_the_disorder_is_counted(tmp_path, cfg):
    """File order 5.0, 1.0, 3.0 -> one descending adjacent pair (5.0 -> 1.0).

    The counter is "adjacent pairs where the next segment starts earlier", so 1.0 -> 3.0
    does not count even though the file was unsorted overall.
    """
    segs = [_seg(0, "評估者", 5.0, 6.0),
            _seg(1, "兒童", 1.0, 2.0),
            _seg(2, "評估者", 3.0, 4.0)]
    turns, rep = load_transcript(_write(tmp_path, segs), cfg)
    assert rep.n_out_of_order == 1
    assert [t.t_start for t in turns] == [1.0, 3.0, 5.0]
    assert [t.turn_id for t in turns] == [0, 1, 2]


def test_empty_and_zero_length_segments_are_dropped(tmp_path, cfg):
    segs = [_seg(0, "評估者", 0.0, 1.0),
            _seg(1, "兒童", 1.2, 2.0, "   "),
            _seg(2, "兒童", 3.0, 3.0, "零長度"),
            _seg(3, "兒童", 4.0, 5.0)]
    _, rep = load_transcript(_write(tmp_path, segs), cfg)
    assert rep.n_empty_text == 2 and rep.n_kept == 2


def test_folder_name_wins_over_metadata_case_name(tmp_path, cfg):
    """One case in this cohort has a metadata case_name that disagrees with its folder.

    The subject id is the join key for every subject-level split, so it comes from the
    folder and the disagreement is recorded as a note rather than silently accepted.
    """
    segs = [_seg(0, "評估者", 0.0, 1.0), _seg(1, "兒童", 1.2, 2.0)]
    p = _write(tmp_path, segs, meta={"case_name": "9999-完全不同的名字"})
    _, rep = load_transcript(p, cfg, case_id=p.parent.name)
    assert rep.case_id == "20250101-0000000測試童"
    assert rep.notes and "disagrees with the folder name" in rep.notes[0]


def test_discover_cases_honours_the_exclusion_list(tmp_path, cfg):
    for name in ("20250101-1壹", "20250102-2貳"):
        d = tmp_path / name
        d.mkdir()
        (d / "a.mp3").write_bytes(b"")
        (d / "a_edited.json").write_text('{"segments": []}', encoding="utf-8")
    assert len(discover_cases(tmp_path)) == 2
    assert [c.case_id for c in discover_cases(tmp_path, exclude=["貳"])] == ["20250101-1壹"]


def test_discover_cases_prefers_the_edited_transcript(tmp_path):
    d = tmp_path / "20250101-1壹"
    d.mkdir()
    (d / "a.mp3").write_bytes(b"")
    (d / "raw.json").write_text('{"segments": []}', encoding="utf-8")
    (d / "a_edited.json").write_text('{"segments": []}', encoding="utf-8")
    assert discover_cases(tmp_path)[0].transcript.name == "a_edited.json"


def test_speech_time_accounting(tmp_path, cfg):
    """Examiner 1.0 + 1.0 s, child 0.5 + 1.5 s -> child talk ratio = 2.0 / 4.0 = 0.5."""
    segs = [_seg(0, "評估者", 0.0, 1.0),
            _seg(1, "兒童", 1.2, 1.7),
            _seg(2, "評估者", 2.0, 3.0),
            _seg(3, "兒童", 3.2, 4.7)]
    _, rep = load_transcript(_write(tmp_path, segs), cfg)
    assert rep.examiner_speech_s == pytest.approx(2.0)
    assert rep.child_speech_s == pytest.approx(2.0)
    assert rep.child_talk_ratio == pytest.approx(0.5)
    assert rep.session_s == pytest.approx(4.7)


def test_cohort_subdirectory_is_resolved_when_present(tmp_path):
    """The intake tree gained a cohort level (Fine_tune/TD/<case>) on 2026-09-12.

    The old layout returned an empty list rather than raising, and the extractor then died
    on an empty corpus several frames away -- a failure that reads as "no data" rather than
    "wrong directory". Both layouts must work.
    """
    from asdpipe.realdata import cohort_root, discover_cases

    def make_case(parent, name):
        d = parent / name
        d.mkdir(parents=True)
        (d / f"{name}.mp3").write_bytes(b"\x00")
        (d / f"{name}_edited.json").write_text("{}", encoding="utf-8")

    nested = tmp_path / "nested"
    make_case(nested / "TD", "20250101-1111甲")
    (nested / "ASD").mkdir()                       # present but empty, as on disk today
    assert cohort_root(nested, "TD") == nested / "TD"
    assert len(discover_cases(nested, cohort="TD")) == 1

    flat = tmp_path / "flat"
    make_case(flat, "20250101-2222乙")
    assert cohort_root(flat, "TD") == flat         # no TD/ level -> use the root itself
    assert len(discover_cases(flat, cohort="TD")) == 1


def test_reference_report_reads_per_subject_quality_dicts(cfg, tmp_path):
    """`SubjectTable.quality` is a list of dicts, one per subject -- not a dict of arrays.

    Written after treating it as the latter: a 27-minute extraction finished, then the
    report writer died on `.get`, so the run produced nothing. Anything the report reads out
    of `quality` needs a test, because that path only executes at the very end.
    """
    import numpy as np

    from asdpipe.assemble import build_table
    from asdpipe.contracts import Label, SubjectRow
    from asdpipe.eval.reference import write_reference_report

    rng = np.random.default_rng(0)
    rows = []
    for i in range(4):
        rows.append(SubjectRow(
            subject_id=f"S{i}", label=Label.CONTROL,
            b_pooled={"b_Loudness_mean": float(rng.normal())},
            c_pooled={"c_npvi_duration": float(45 + rng.normal())},
            e_pooled={"e_g1_similarity_median": float(rng.uniform(0.3, 0.6))},
            c_sequence=np.zeros((3, 8), dtype=np.float32),
            egemaps_functionals=np.zeros(88, dtype=np.float32),
            examiner_probe={},
            quality={"e_f7_slope_w3": -1e-4, "e_f7_slope_w5": -2e-4,
                     "e_f7_slope_w7": -3e-4, "b_ecapa_windows": 200.0,
                     "e_n_pairs": 50.0, "e_n_g1_pairs": 40.0, "c_n_syllables": 900.0},
            ados={}, latents={},
        ))
    out = write_reference_report(tmp_path / "ref.md", build_table(rows), cfg)
    text = out.read_text(encoding="utf-8")
    assert "P0-F7" in text
    assert "4 輪" not in text            # only the configured windows appear


def test_english_speaker_labels_from_the_asd_transcripts_are_mapped(cfg, tmp_path):
    """The ASD arm's transcripts use Child/Tester where the TD arm uses 兒童/測試者.

    Unmapped labels are treated as third parties and dropped, so the unfixed version of this
    produced zero child turns for every ASD case and reported it as a clean run.
    """
    import json

    from asdpipe.realdata import load_transcript

    p = tmp_path / "t.json"
    p.write_text(json.dumps({"segments": [
        {"start": 0.0, "end": 1.0, "speaker": "Tester", "text": "你叫什麼名字"},
        {"start": 1.2, "end": 2.0, "speaker": "Child", "text": "我叫小明"},
        {"start": 2.2, "end": 3.0, "speaker": "個管師", "text": "旁邊有人講話"},
    ]}, ensure_ascii=False), encoding="utf-8")
    turns, rep = load_transcript(p, cfg, case_id="X")
    assert rep.n_child == 1 and rep.n_examiner == 1
    assert rep.n_third_party == 1              # 個管師 stays a third party
    assert rep.usable


def test_a_transcript_with_no_child_turns_is_an_error_not_a_quiet_zero(cfg, tmp_path):
    """A table of zeros reads as a finding about the child; it is a finding about the file."""
    import json

    from asdpipe.realdata import load_transcript

    p = tmp_path / "t.json"
    p.write_text(json.dumps({"segments": [
        {"start": 0.0, "end": 1.0, "speaker": "Tester", "text": "你好嗎"},
        {"start": 1.2, "end": 2.0, "speaker": "小小孩", "text": "我很好"},   # not in the table
    ]}, ensure_ascii=False), encoding="utf-8")
    turns, rep = load_transcript(p, cfg, case_id="X")
    assert rep.n_child == 0
    assert not rep.usable
    assert any("no child turns" in e for e in rep.errors)


def test_video_containers_are_discovered_as_audio(tmp_path):
    """The ASD arm was delivered as .mp4. librosa decodes the audio track; a container
    extension must not make four transcribed cases invisible."""
    from asdpipe.realdata import discover_cases

    d = tmp_path / "ASD" / "case1"
    d.mkdir(parents=True)
    (d / "case1.mp4").write_bytes(b"\x00")
    (d / "case1_edited.json").write_text("{}", encoding="utf-8")
    assert len(discover_cases(tmp_path, cohort="ASD")) == 1


def test_mp3_is_preferred_when_a_session_has_both(tmp_path):
    """MP3 is the only container both arms share; the camera MP4 beside it must not win."""
    from asdpipe.realdata import discover_cases

    d = tmp_path / "TD" / "case1"
    d.mkdir(parents=True)
    for name in ("a_video.mp4", "z_audio.mp3", "case1_edited.json"):
        (d / name).write_bytes(b"\x00")
    assert discover_cases(tmp_path, cohort="TD")[0].audio.name == "z_audio.mp3"


def _write_json(path, segments):
    import json

    path.write_text(json.dumps({"segments": segments}, ensure_ascii=False), encoding="utf-8")
    return path


def test_silent_markers_are_not_turns(cfg, tmp_path):
    """[SILENT] is annotation, not speech.

    Left as ordinary text it was 48.6 minutes of "speech" across four ASD cases, 332 of it
    charged to the examiner, which inflated examiner turn duration from 2.84 s to 4.26 s and
    made the two cohorts look like they had been segmented under different conventions. The
    damage was entirely invisible: no error, no NaN, just a wrong number.
    """
    from asdpipe.realdata import load_transcript

    p = _write_json(tmp_path / "t.json", [
        {"start": 0.0, "end": 1.0, "speaker": "Tester", "text": "你好嗎"},
        {"start": 1.0, "end": 9.0, "speaker": "Tester", "text": "[SILENT]"},
        {"start": 9.0, "end": 10.0, "speaker": "Child", "text": "我很好"},
    ])
    turns, rep = load_transcript(p, cfg, case_id="X")
    assert len(turns) == 2
    assert rep.n_silent_marker == 1
    assert rep.silent_marker_s == pytest.approx(8.0)
    assert all("[SILENT]" not in t.text for t in turns)


def _unintelligible_case(tmp_path):
    return _write_json(tmp_path / "t.json", [
        {"start": 0.0, "end": 1.0, "speaker": "Tester", "text": "這是什麼"},
        {"start": 1.2, "end": 2.4, "speaker": "Child", "text": "[UNINTELLIGIBLE]"},
        {"start": 2.6, "end": 3.6, "speaker": "Child", "text": "車車"},
    ])


def test_unintelligible_turns_are_excluded_by_default_but_still_counted(cfg, tmp_path):
    """Only the ASD transcripts mark these; the TD annotators recorded the same
    vocalisations as silence. Keeping them would add child turns to one cohort and not the
    other -- a group difference manufactured by the annotation protocol. The count stays in
    the report so the size of what was discarded is visible rather than lost.
    """
    from asdpipe.realdata import load_transcript

    turns, rep = load_transcript(_unintelligible_case(tmp_path), cfg, case_id="X")
    assert rep.n_unintelligible == 1
    assert rep.n_child == 1                       # only 車車
    assert not any(t.unintelligible for t in turns)


def test_unintelligible_turns_can_be_kept_without_text(cfg, tmp_path):
    """If both arms are ever annotated to one protocol, these become usable: a vocalisation
    the annotator could not transcribe is still a vocalisation, and for a minimally verbal
    child it may be most of what there is. Kept, they carry no text, so the sentence encoder
    and the forced aligner skip them instead of being handed "[UNINTELLIGIBLE]" as speech.
    """
    from asdpipe.realdata import load_transcript

    keep = {**cfg, "realdata": {**cfg["realdata"], "drop_unintelligible": False}}
    turns, rep = load_transcript(_unintelligible_case(tmp_path), keep, case_id="X")
    assert rep.n_child == 2
    unint = [t for t in turns if t.unintelligible]
    assert len(unint) == 1
    assert unint[0].text == ""
    assert unint[0].duration == pytest.approx(1.2)


def test_silence_is_recognised_through_all_three_channels(cfg, tmp_path):
    """The ASD export marks silence three ways: the text marker, a "Silence" speaker label,
    and manual_annotation. Handling only the text left 53 segments counted as an
    unrecognised *speaker*, i.e. as a data problem rather than as silence."""
    from asdpipe.realdata import load_transcript

    p = _write_json(tmp_path / "t.json", [
        {"start": 0.0, "end": 1.0, "speaker": "Tester", "text": "你好嗎"},
        {"start": 1.0, "end": 5.0, "speaker": "Tester", "text": "[SILENT]"},
        {"start": 5.0, "end": 9.0, "speaker": "Silence", "text": ""},
        {"start": 9.0, "end": 12.0, "speaker": "Tester", "text": "", "manual_annotation": "silent"},
        {"start": 12.0, "end": 13.0, "speaker": "Child", "text": "我很好"},
    ])
    turns, rep = load_transcript(p, cfg, case_id="X")
    assert rep.n_silent_marker == 3
    assert rep.n_unmapped_labels == 0             # "Silence" is not an unknown speaker
    assert len(turns) == 2 and rep.n_child == 1


def test_a_transcript_of_nothing_but_markers_is_an_error(cfg, tmp_path):
    """The degenerate case has to be loud: markers only means no usable child speech."""
    from asdpipe.realdata import load_transcript

    p = _write_json(tmp_path / "t.json", [
        {"start": 0.0, "end": 5.0, "speaker": "Tester", "text": "[SILENT]"},
        {"start": 5.0, "end": 9.0, "speaker": "Tester", "text": "今天玩什麼"},
    ])
    turns, rep = load_transcript(p, cfg, case_id="X")
    assert rep.n_child == 0 and not rep.usable
