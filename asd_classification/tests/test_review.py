"""Review tool for the health check's step 5: what the browser may see, how answers are
kept, and how they are scored."""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

from asdpipe.contracts import SyllableSpan
from asdpipe.review.prepare import merge_boundaries, public_view, spread_pick
from asdpipe.review.score import char_edits, latest_answers, score_package


def test_adjacent_syllables_share_one_boundary():
    spans = [SyllableSpan(10.30, 10.50, 3, "我"), SyllableSpan(10.50, 10.70, 4, "要"),
             SyllableSpan(10.90, 11.10, 2, "玩")]                    # a pause before 玩
    bounds, syls = merge_boundaries(spans, t0=10.0)
    assert bounds == pytest.approx([0.30, 0.50, 0.70, 0.90, 1.10])
    assert [(s["a"], s["b"], s["text"]) for s in syls] == [(0, 1, "我"), (1, 2, "要"), (3, 4, "玩")]


def test_the_browser_never_sees_group_case_or_transcript():
    item = {"id": "ABC234", "clip": "ABC234.wav", "dur": 1.2, "group": "ASD",
            "case": "某位孩子", "label": "child", "text": "原稿"}
    assert public_view("speaker", item) == {"id": "ABC234", "clip": "ABC234.wav", "dur": 1.2}


def test_every_child_contributes_before_any_contributes_twice():
    by_case = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7]}
    picked = spread_pick(by_case, 3, random.Random(0))
    owner = {1: "a", 2: "a", 3: "a", 4: "b", 5: "b", 6: "b", 7: "c"}
    assert sorted(owner[x] for x in picked) == ["a", "b", "c"]


def test_char_edits_ignore_punctuation():
    assert char_edits("我要紅色的。", "我要紅的") == 1
    assert char_edits("你好嗎？", "你好嗎") == 0


def test_a_later_answer_replaces_an_earlier_one(tmp_path):
    p = tmp_path / "answers.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [
        {"task": "speaker", "id": "X", "answer": {"speaker": "child"}},
        {"task": "speaker", "id": "X", "answer": {"speaker": "examiner"}}]), encoding="utf-8")
    assert latest_answers(p)["speaker"]["X"] == {"speaker": "examiner"}


@pytest.fixture
def package(tmp_path):
    import soundfile as sf

    d = tmp_path / "review"
    (d / "clips").mkdir(parents=True)
    sf.write(d / "clips" / "ABC234.wav", np.zeros(1600, np.float32), 16000)
    items = {"meta": {}, "tasks": {"mfa": [], "speaker": [{"id": "ABC234", "clip": "ABC234.wav",
                                                           "dur": 0.1}],
                                   "echo": [], "a12": [], "transcribe": []}}
    key = {"meta": {}, "key": {"mfa": {}, "echo": {}, "a12": {}, "transcribe": {},
                               "speaker": {"ABC234": {"group": "TD", "case": "SECRETCASE",
                                                      "turn_id": 1, "label": "child"}}}}
    (d / "items.json").write_text(json.dumps(items), encoding="utf-8")
    (d / "key.json").write_text(json.dumps(key), encoding="utf-8")
    return d


def test_server_serves_items_and_clips_but_not_the_key(package):
    from fastapi.testclient import TestClient

    from asdpipe.review.server import create_app

    c = TestClient(create_app(package))
    assert "SECRETCASE" not in c.get("/api/items").text
    assert c.get("/clip/ABC234.wav").status_code == 200
    assert c.get("/clip/key.json").status_code == 404
    assert c.post("/api/answer", json={"task": "speaker", "id": "NOPE99",
                                       "answer": {"speaker": "child"}}).status_code == 400
    assert c.post("/api/answer", json={"task": "speaker", "id": "ABC234",
                                       "answer": {"speaker": "child"}}).status_code == 200
    assert c.get("/api/answers").json()["speaker"]["ABC234"] == {"speaker": "child"}


def test_report_scores_agreement_and_names_no_case(package):
    (package / "answers.jsonl").write_text(json.dumps(
        {"task": "speaker", "id": "ABC234", "answer": {"speaker": "child"}}) + "\n",
        encoding="utf-8")
    out, metrics = score_package(package)
    assert metrics["speaker_TD"] == {"n": 1, "agree": 1, "unsure": 0}
    assert "SECRETCASE" not in out.read_text(encoding="utf-8")


def test_agreement_interval_resamples_sentences():
    from asdpipe.review.score import agreement_ci

    point, lo, hi = agreement_ci([0, 1, 0, 2], [10, 10, 10, 10])
    assert point == pytest.approx(1 - 3 / 40)
    assert lo <= point <= hi


def test_alignment_is_scored_as_a_failure_rate_and_old_saves_count_as_right(package):
    key_path = package / "key.json"
    key = json.loads(key_path.read_text(encoding="utf-8"))
    key["key"]["mfa"] = {c: {"group": "TD", "case": "SECRETCASE", "turn_id": i}
                         for i, c in enumerate(["AAAAAA", "BBBBBB", "CCCCCC", "DDDDDD"])}
    key_path.write_text(json.dumps(key), encoding="utf-8")
    (package / "answers.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"task": "mfa", "id": "AAAAAA", "answer": {"status": "ok", "boundaries": [0.1]}},
        {"task": "mfa", "id": "BBBBBB", "answer": {"status": "aligned"}},
        {"task": "mfa", "id": "CCCCCC", "answer": {"status": "misaligned"}},
        {"task": "mfa", "id": "DDDDDD", "answer": {"status": "cannot"}}]) + "\n", encoding="utf-8")
    _, m = score_package(package)
    assert m["mfa_TD"] == {"aligned": 2, "misaligned": 1, "cannot": 1, "misaligned_rate": 1 / 3}
