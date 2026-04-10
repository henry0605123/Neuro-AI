# -*- coding: utf-8 -*-
"""
依 metadata/clip_timings.csv 與 clip_to_raw_audio.csv，將原始音訊裁成
Playlogue 對齊用的 clip，輸出為 16 kHz mono WAV（方便後續 Whisper / 訓練管線）。

時間單位與官方一致：clip_timings 為「毫秒」，相對於 raw_audio_path 該檔之時間軸。
end_time == -1 表示裁到檔案結尾（以 ffprobe 讀取時長）。

需求：系統 PATH 上有 ffmpeg、ffprobe。

使用方式（在 Playlogue 根目錄）：
  python scripts/trim_playlogue_clips.py
  python scripts/trim_playlogue_clips.py --skip-existing
  python scripts/trim_playlogue_clips.py --limit 5
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def playlogue_root_from_script() -> Path:
    return Path(__file__).resolve().parent.parent


def ffprobe_duration_sec(media_path: Path) -> float:
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(media_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe 失敗 ({media_path}): {r.stderr.strip()}")
    line = (r.stdout or "").strip().splitlines()[-1] if r.stdout else ""
    return float(line)


def trim_to_wav(
    src: Path,
    dst: Path,
    start_sec: float,
    end_sec: float,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-ss",
        f"{start_sec:.6f}",
        "-to",
        f"{end_sec:.6f}",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg 失敗: {r.stderr.strip() or r.stdout.strip()}")


def load_timings(path: Path) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row["id"].strip()
            out[cid] = (int(row["start_time"]), int(row["end_time"]))
    return out


def load_raw_map(path: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row["id"].strip()
            raw = (row.get("raw_audio_path") or "").strip()
            if raw:
                out[cid] = Path(raw)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="依 Playlogue metadata 裁切音訊 clip")
    parser.add_argument(
        "--playlogue-root",
        type=Path,
        default=playlogue_root_from_script(),
        help="Playlogue 根目錄（含 metadata/）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="輸出 WAV 目錄（預設: <root>/data/audio_trimmed）",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="若目標 WAV 已存在則跳過",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只處理前 N 筆（除錯用）",
    )
    args = parser.parse_args()

    root: Path = args.playlogue_root.resolve()
    timings_path = root / "metadata" / "clip_timings.csv"
    map_path = root / "metadata" / "clip_to_raw_audio.csv"
    out_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (root / "data" / "audio_trimmed")
    )

    if not timings_path.is_file():
        print(f"找不到: {timings_path}", file=sys.stderr)
        return 1
    if not map_path.is_file():
        print(f"找不到: {map_path}（請先執行 build_clip_to_raw_audio_map.py）", file=sys.stderr)
        return 1

    timings = load_timings(timings_path)
    raw_map = load_raw_map(map_path)

    ids = list(timings.keys())
    if args.limit is not None:
        ids = ids[: args.limit]

    ok, skipped, failed = 0, 0, 0
    for i, cid in enumerate(ids):
        start_ms, end_ms = timings[cid]
        src = raw_map.get(cid)
        if src is None or not src.is_file():
            print(f"[{i+1}/{len(ids)}] 跳過（無音檔）: {cid}")
            failed += 1
            continue

        dst = out_dir / f"{cid}.wav"
        if args.skip_existing and dst.is_file():
            skipped += 1
            continue

        try:
            dur_sec = ffprobe_duration_sec(src)
            dur_ms = int(dur_sec * 1000)
            if end_ms < 0:
                end_ms_eff = dur_ms
            else:
                end_ms_eff = end_ms
            start_sec = start_ms / 1000.0
            end_sec = end_ms_eff / 1000.0
            if start_sec >= dur_sec:
                print(f"[{i+1}/{len(ids)}] 錯誤 start 超出長度: {cid}", file=sys.stderr)
                failed += 1
                continue
            end_sec = min(end_sec, dur_sec)
            if end_sec <= start_sec:
                print(f"[{i+1}/{len(ids)}] 錯誤 區間為空: {cid}", file=sys.stderr)
                failed += 1
                continue
            trim_to_wav(src, dst, start_sec, end_sec)
            ok += 1
            print(f"[{i+1}/{len(ids)}] OK {cid} -> {dst.name}")
        except Exception as e:
            print(f"[{i+1}/{len(ids)}] 失敗 {cid}: {e}", file=sys.stderr)
            failed += 1

    print(f"\n完成: 成功 {ok}, 跳過 {skipped}, 失敗 {failed}, 輸出目錄: {out_dir}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
