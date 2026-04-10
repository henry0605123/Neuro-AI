# -*- coding: utf-8 -*-
"""
依 Playlogue clip id 與預設 CHILDES 資料夾結構，自動建立
metadata/clip_to_raw_audio.csv（id -> 原始音檔絕對路徑）。

規則（對應常見下載結構 data/audio/...）：
- ew_42ec_12007  -> EllisWeismer/42ec/12007.{mp3,wav,...}
- gleason_father_andy -> Gleason/Father/andy.{...}
- gleason_mother_katie -> Gleason/Mother/katie.{...}
- vh_brownf -> VanHouten/freeplay/brownf.{...}
- cameron_aae_* -> Cameron/AAE/*.mp3
- cameron_sae_* -> Cameron/SAE/*.mp3

連續兩列若滿足 end_time == 下一列 start_time（且 end != -1），
視為同一支長檔的兩段（例如 ew_42ec_12003 / ew_42ec_12004），
第二段沿用第一段推導出的 raw 路徑。

使用方式（在 Playlogue 根目錄或任意目錄）：
  python scripts/build_clip_to_raw_audio_map.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

AUDIO_EXTS = (".mp3", ".wav", ".flac", ".m4a", ".MP3", ".WAV")


def playlogue_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_audio_file(base: Path) -> Path | None:
    """若 base 含副檔名且存在則回傳，否則嘗試常見副檔名。"""
    if base.suffix and base.is_file():
        return base
    for ext in AUDIO_EXTS:
        p = base.with_suffix(ext)
        if p.is_file():
            return p
    return None


def infer_raw_path(clip_id: str, audio_root: Path) -> Path | None:
    """由 clip id 推導預期音檔路徑（不含副檔名時會在 resolve 時補上）。"""
    if clip_id.startswith("ew_"):
        parts = clip_id.split("_", 2)
        if len(parts) == 3 and parts[0] == "ew":
            condition, num = parts[1], parts[2]
            stem = audio_root / "EllisWeismer" / condition / num
            return resolve_audio_file(stem)
    if clip_id.startswith("gleason_father_"):
        name = clip_id.removeprefix("gleason_father_")
        stem = audio_root / "Gleason" / "Father" / name
        return resolve_audio_file(stem)
    if clip_id.startswith("gleason_mother_"):
        name = clip_id.removeprefix("gleason_mother_")
        stem = audio_root / "Gleason" / "Mother" / name
        return resolve_audio_file(stem)
    if clip_id.startswith("vh_"):
        name = clip_id.removeprefix("vh_")
        stem = audio_root / "VanHouten" / "freeplay" / name
        return resolve_audio_file(stem)
    if clip_id.startswith("cameron_aae_"):
        rest = clip_id.removeprefix("cameron_aae_")
        stem = audio_root / "Cameron" / "AAE" / rest
        return resolve_audio_file(stem)
    if clip_id.startswith("cameron_sae_"):
        rest = clip_id.removeprefix("cameron_sae_")
        stem = audio_root / "Cameron" / "SAE" / rest
        return resolve_audio_file(stem)
    return None


def ew_session_prefix(clip_id: str) -> str | None:
    """回傳 'ew_42ec' 這類前綴，非 ew 則 None。"""
    if not clip_id.startswith("ew_"):
        return None
    parts = clip_id.split("_", 2)
    if len(parts) != 3:
        return None
    return f"{parts[0]}_{parts[1]}"


def main() -> int:
    root = playlogue_root()
    audio_root = root / "data" / "audio"
    timings_path = root / "metadata" / "clip_timings.csv"
    out_path = root / "metadata" / "clip_to_raw_audio.csv"

    if not timings_path.is_file():
        print(f"找不到: {timings_path}", file=sys.stderr)
        return 1
    if not audio_root.is_dir():
        print(f"找不到音訊根目錄: {audio_root}", file=sys.stderr)
        return 1

    rows: list[dict[str, str]] = []
    with open(timings_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    paths: list[Path | None] = []
    notes: list[str] = []
    for row in rows:
        cid = row["id"].strip()
        p = infer_raw_path(cid, audio_root)
        paths.append(p)
        notes.append("")

    # 同一支長檔拆成兩列：end_time == 下一列 start_time
    for i in range(len(rows) - 1):
        try:
            end_i = int(rows[i]["end_time"])
            start_j = int(rows[i + 1]["start_time"])
        except ValueError:
            continue
        if end_i < 0:
            continue
        if end_i != start_j:
            continue
        pre_i = ew_session_prefix(rows[i]["id"].strip())
        pre_j = ew_session_prefix(rows[i + 1]["id"].strip())
        if pre_i and pre_j and pre_i == pre_j and paths[i] is not None:
            paths[i + 1] = paths[i]
            notes[i + 1] = f"same_raw_as:{rows[i]['id'].strip()}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "raw_audio_path", "note"])
        for row, p, note in zip(rows, paths, notes):
            cid = row["id"].strip()
            if p is None or not p.is_file():
                missing.append(cid)
                w.writerow([cid, "", note or "FILE_NOT_FOUND"])
            else:
                w.writerow([cid, str(p.resolve()), note])

    print(f"已寫入: {out_path}")
    print(f"共 {len(rows)} 筆 clip，缺檔: {len(missing)}")
    if missing:
        print("缺檔 id 列表:")
        for m in missing:
            print(f"  - {m}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
