# -*- coding: utf-8 -*-
"""
依 metadata/splits.csv，將裁切後 WAV 與 speaker_diarization RTTM
複製成 Fine_tune_LSTM 可用的目錄結構：

  <output_root>/
    train/<clip_id>/audio.wav
    train/<clip_id>/diarization.rttm
    val/<clip_id>/...
    test/<clip_id>/...

之後在 config 中設定：
  train_data_dir = <output_root>/train
  val_data_dir   = <output_root>/val
  test_audio_path / test_rttm_path 可指向 test 底下任一 clip。

使用方式：
  python scripts/prepare_playlogue_finetune_layout.py
  python scripts/prepare_playlogue_finetune_layout.py --output-root D:\\data\\playlogue_splits
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path


def playlogue_root_from_script() -> Path:
    return Path(__file__).resolve().parent.parent


def project_root_from_playlogue(playlogue: Path) -> Path:
    return playlogue.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="依 split 建立 Playlogue 訓練目錄")
    parser.add_argument(
        "--playlogue-root",
        type=Path,
        default=playlogue_root_from_script(),
        help="Playlogue 根目錄",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="輸出根目錄（預設：<專案>/Fine_tune_LSTM/data/playlogue_splits）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只列出將執行的動作，不複製檔案",
    )
    args = parser.parse_args()

    playlogue: Path = args.playlogue_root.resolve()
    if args.output_root is not None:
        out_root = args.output_root.resolve()
    else:
        out_root = (
            project_root_from_playlogue(playlogue)
            / "Fine_tune_LSTM"
            / "data"
            / "playlogue_splits"
        )

    splits_csv = playlogue / "metadata" / "splits.csv"
    trimmed_dir = playlogue / "data" / "audio_trimmed"
    rttm_dir = playlogue / "data" / "speaker_diarization"

    if not splits_csv.is_file():
        print(f"找不到: {splits_csv}", file=sys.stderr)
        return 1

    rows: list[tuple[str, str]] = []
    with open(splits_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row["id"].strip()
            sp = row["split"].strip().lower()
            if sp not in ("train", "val", "test"):
                print(f"警告：未知 split「{sp}」，略過 {cid}", file=sys.stderr)
                continue
            rows.append((cid, sp))

    counts: Counter[str] = Counter()
    missing_wav: list[str] = []
    missing_rttm: list[str] = []

    for clip_id, split in rows:
        wav_src = trimmed_dir / f"{clip_id}.wav"
        rttm_src = rttm_dir / f"{clip_id}.rttm"
        dst_dir = out_root / split / clip_id

        if not wav_src.is_file():
            missing_wav.append(clip_id)
            continue

        if not args.dry_run:
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(wav_src, dst_dir / "audio.wav")
        counts[split] += 1

        if rttm_src.is_file():
            if not args.dry_run:
                shutil.copy2(rttm_src, dst_dir / "diarization.rttm")
        else:
            missing_rttm.append(clip_id)

    print(f"Playlogue 根目錄: {playlogue}")
    print(f"輸出根目錄: {out_root}")
    print(f"模式: {'dry-run' if args.dry_run else '複製檔案'}")
    print(
        f"統計: train={counts.get('train', 0)}, val={counts.get('val', 0)}, "
        f"test={counts.get('test', 0)}"
    )
    if missing_wav:
        print(f"警告：缺 WAV（已略過 {len(missing_wav)} 筆）", file=sys.stderr)
    if missing_rttm:
        print(f"警告：缺 RTTM（{len(missing_rttm)} 筆無 diarization.rttm）", file=sys.stderr)

    print("\n請在 Fine_tune_LSTM/config.py 設定（路徑請依實際調整）：")
    print(f'  train_data_dir = r"{out_root / "train"}"')
    print(f'  val_data_dir   = r"{out_root / "val"}"')
    print("  test 請自選 test 子目錄內一組 clip，例如：")
    ex_test = out_root / "test"
    if ex_test.is_dir():
        subs = sorted(p for p in ex_test.iterdir() if p.is_dir())
        if subs:
            ex = subs[0]
            print(f'  test_audio_path = r"{ex / "audio.wav"}"')
            print(f'  test_rttm_path  = r"{ex / "diarization.rttm"}"')

    return 0 if not missing_wav else 2


if __name__ == "__main__":
    raise SystemExit(main())
