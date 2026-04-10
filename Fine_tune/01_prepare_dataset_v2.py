"""
Step 1 (V2): 資料準備與切片
============================
將長音檔切成小片段，大幅增加有效訓練資料量

核心改進：
1. 將音檔切成 5 秒重疊片段（而非只取 1 個隨機片段）
2. 過濾純靜音片段
3. 確保包含小孩語音的片段被充分採樣
4. 計算類別權重用於平衡訓練

執行方式：
    python 01_prepare_dataset_v2.py

輸出：
    - data_v2/sliced/*.pt  預處理的音訊片段
    - data_v2/segments.json  片段資訊
    - data_v2/class_weights.json  類別權重
"""

import os
import sys
import json
import random
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict
import numpy as np
from tqdm import tqdm

# 加入專案路徑
sys.path.insert(0, str(Path(__file__).parent))

from config_v2 import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SLICED_DATA_DIR,
    SPEAKER_MAPPING,
    VAL_RATIO,
    RANDOM_SEED,
    SEGMENT_DURATION,
    SEGMENT_OVERLAP,
    MIN_SPEECH_RATIO,
    REQUIRE_CHILD_SPEECH,
    MIN_CHILD_RATIO,
    MAX_SAMPLES_PER_FILE,
    SPEAKER_WEIGHTS,
    AUTO_COMPUTE_WEIGHTS,
    create_directories
)
from utils.srt_parser import SRTParser
from utils.rttm_converter import RTTMConverter
from utils.audio_utils import AudioUtils


@dataclass
class SegmentInfo:
    """片段資訊"""
    segment_id: str
    file_id: str
    start_time: float
    end_time: float
    audio_path: str
    rttm_path: str
    speech_ratio: float
    speaker_ratios: Dict[str, float]
    has_child_speech: bool
    child_ratio: float


def find_audio_file(folder: Path) -> Path:
    """在資料夾中尋找音訊檔案"""
    audio_extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
    for ext in audio_extensions:
        files = list(folder.glob(f"*{ext}"))
        if files:
            return files[0]
    raise FileNotFoundError(f"在 {folder} 中找不到音訊檔案")


def find_srt_file(folder: Path) -> Path:
    """在資料夾中尋找 SRT 標註檔案"""
    for name in ["Aegisub.srt", "GTruth.srt"]:
        srt = folder / name
        if srt.exists():
            return srt
    srt_files = list(folder.glob("*.srt"))
    if srt_files:
        return srt_files[0]
    raise FileNotFoundError(f"在 {folder} 中找不到 SRT 標註檔案")


def parse_rttm_file(rttm_path: Path) -> List[Tuple[float, float, str]]:
    """
    解析 RTTM 檔案
    
    Returns:
        List of (start, end, speaker)
    """
    segments = []
    with open(rttm_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8:
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                segments.append((start, start + duration, speaker))
    return segments


def compute_segment_statistics(
    segment_start: float,
    segment_end: float,
    annotations: List[Tuple[float, float, str]]
) -> Dict:
    """
    計算片段內的語音統計資訊
    
    Args:
        segment_start: 片段開始時間
        segment_end: 片段結束時間
        annotations: RTTM 標註列表
        
    Returns:
        統計資訊字典
    """
    segment_duration = segment_end - segment_start
    
    # 計算每個說話人在此片段內的時長
    speaker_durations = defaultdict(float)
    total_speech = 0.0
    
    for start, end, speaker in annotations:
        # 計算重疊部分
        overlap_start = max(segment_start, start)
        overlap_end = min(segment_end, end)
        
        if overlap_end > overlap_start:
            overlap_duration = overlap_end - overlap_start
            speaker_durations[speaker] += overlap_duration
            total_speech += overlap_duration
    
    # 計算比例
    speech_ratio = total_speech / segment_duration if segment_duration > 0 else 0
    speaker_ratios = {
        speaker: dur / segment_duration 
        for speaker, dur in speaker_durations.items()
    }
    
    # 小孩語音統計
    child_ratio = speaker_ratios.get("SPEAKER_01", 0.0)
    has_child_speech = child_ratio > 0
    
    return {
        "speech_ratio": speech_ratio,
        "speaker_ratios": speaker_ratios,
        "has_child_speech": has_child_speech,
        "child_ratio": child_ratio,
    }


def slice_audio_file(
    audio_path: Path,
    rttm_path: Path,
    file_id: str,
    output_dir: Path,
    duration: float,
    total_duration: float,
    annotations: List[Tuple[float, float, str]]
) -> List[SegmentInfo]:
    """
    將音檔切成小片段
    
    Args:
        audio_path: 音訊檔案路徑
        rttm_path: RTTM 標註路徑
        file_id: 檔案識別碼
        output_dir: 輸出目錄
        duration: 片段長度（秒）
        total_duration: 音訊總時長
        annotations: RTTM 標註
        
    Returns:
        片段資訊列表
    """
    import torch
    import torchaudio
    
    segments = []
    
    # 計算步長（考慮重疊）
    step = duration * (1 - SEGMENT_OVERLAP)
    
    # 產生所有可能的片段
    current_time = 0.0
    segment_idx = 0
    
    while current_time + duration <= total_duration:
        segment_start = current_time
        segment_end = current_time + duration
        
        # 計算此片段的統計資訊
        stats = compute_segment_statistics(segment_start, segment_end, annotations)
        
        # 過濾條件
        include_segment = True
        
        # 條件 1: 最小語音比例
        if stats["speech_ratio"] < MIN_SPEECH_RATIO:
            include_segment = False
        
        # 條件 2: 是否要求小孩語音
        if REQUIRE_CHILD_SPEECH:
            if stats["child_ratio"] < MIN_CHILD_RATIO:
                # 不完全排除，但降低採樣機率
                # 這裡我們用機率決定是否包含
                if random.random() > 0.3:  # 30% 機率保留無小孩語音的片段
                    include_segment = False
        
        if include_segment:
            segment_id = f"{file_id}_seg{segment_idx:04d}"
            
            segment_info = SegmentInfo(
                segment_id=segment_id,
                file_id=file_id,
                start_time=segment_start,
                end_time=segment_end,
                audio_path=str(audio_path),
                rttm_path=str(rttm_path),
                speech_ratio=stats["speech_ratio"],
                speaker_ratios=stats["speaker_ratios"],
                has_child_speech=stats["has_child_speech"],
                child_ratio=stats["child_ratio"],
            )
            
            segments.append(segment_info)
            segment_idx += 1
        
        current_time += step
    
    return segments


def compute_class_weights(all_segments: List[SegmentInfo]) -> Dict[str, float]:
    """
    根據實際標籤分佈計算類別權重
    
    使用逆頻率加權：出現越少的類別權重越高
    """
    # 統計各類別的總時長
    total_by_speaker = defaultdict(float)
    
    for seg in all_segments:
        duration = seg.end_time - seg.start_time
        for speaker, ratio in seg.speaker_ratios.items():
            total_by_speaker[speaker] += ratio * duration
    
    if not total_by_speaker:
        return SPEAKER_WEIGHTS.copy()
    
    # 計算權重（逆頻率）
    total_all = sum(total_by_speaker.values())
    num_classes = len(total_by_speaker)
    
    weights = {}
    for speaker, duration in total_by_speaker.items():
        if duration > 0:
            # 逆頻率加權，並加上平滑項
            freq = duration / total_all
            weights[speaker] = (1.0 / (freq * num_classes)) if freq > 0 else 1.0
        else:
            weights[speaker] = 1.0
    
    # 正規化，讓平均權重為 1
    mean_weight = sum(weights.values()) / len(weights) if weights else 1.0
    weights = {k: v / mean_weight for k, v in weights.items()}
    
    # 限制權重範圍，避免極端值
    MAX_WEIGHT = 5.0
    weights = {k: min(v, MAX_WEIGHT) for k, v in weights.items()}
    
    return weights


def process_single_sample(
    folder: Path,
    output_dir: Path,
    file_id: str
) -> Tuple[Dict, List[SegmentInfo]]:
    """
    處理單個樣本，產生所有片段
    
    Returns:
        (樣本資訊, 片段列表)
    """
    # 尋找檔案
    audio_file = find_audio_file(folder)
    srt_file = find_srt_file(folder)
    
    print(f"  音訊檔: {audio_file.name}")
    print(f"  標註檔: {srt_file.name}")
    
    # 取得音訊時長
    duration = AudioUtils.get_duration(audio_file)
    if duration is None:
        raise ValueError(f"無法取得音訊時長: {audio_file}")
    
    print(f"  時長: {duration:.2f} 秒 ({duration/60:.1f} 分鐘)")
    
    # 解析 SRT
    segments = SRTParser.parse_file(srt_file)
    print(f"  段落數: {len(segments)}")
    
    # 顯示說話人統計
    stats = SRTParser.get_speaker_statistics(segments)
    for speaker, info in stats.items():
        print(f"    {speaker}: {info['count']} 段, {info['duration']:.2f} 秒")
    
    # 轉換為 RTTM
    rttm_path = output_dir / f"{file_id}.rttm"
    rttm_lines = RTTMConverter.srt_to_rttm(
        srt_file,
        output_path=rttm_path,
        file_id=file_id,
        speaker_mapping=SPEAKER_MAPPING
    )
    print(f"  RTTM 行數: {len(rttm_lines)}")
    
    # 建立 UEM 檔案
    uem_path = output_dir / f"{file_id}.uem"
    RTTMConverter.create_uem(file_id, duration, uem_path)
    
    # 複製音訊檔
    audio_link = output_dir / f"{file_id}{audio_file.suffix}"
    if not audio_link.exists():
        shutil.copy2(audio_file, audio_link)
    
    # 解析 RTTM 取得標註
    annotations = parse_rttm_file(rttm_path)
    
    # 切片音訊
    print(f"  正在切片...")
    segment_infos = slice_audio_file(
        audio_path=audio_link,
        rttm_path=rttm_path,
        file_id=file_id,
        output_dir=SLICED_DATA_DIR,
        duration=SEGMENT_DURATION,
        total_duration=duration,
        annotations=annotations
    )
    
    # 限制每個檔案的片段數
    if len(segment_infos) > MAX_SAMPLES_PER_FILE:
        # 優先保留有小孩語音的片段
        with_child = [s for s in segment_infos if s.has_child_speech]
        without_child = [s for s in segment_infos if not s.has_child_speech]
        
        # 確保小孩語音片段佔一定比例
        min_child_count = int(MAX_SAMPLES_PER_FILE * 0.4)
        
        if len(with_child) >= min_child_count:
            # 從有小孩語音的片段中採樣
            selected_with_child = random.sample(with_child, min(len(with_child), min_child_count))
            remaining = MAX_SAMPLES_PER_FILE - len(selected_with_child)
            # 剩餘從無小孩語音的片段中採樣
            selected_without = random.sample(without_child, min(len(without_child), remaining))
            segment_infos = selected_with_child + selected_without
        else:
            # 小孩語音片段不夠，全部使用
            remaining = MAX_SAMPLES_PER_FILE - len(with_child)
            selected_without = random.sample(without_child, min(len(without_child), remaining))
            segment_infos = with_child + selected_without
    
    print(f"  切片完成: {len(segment_infos)} 個片段")
    
    # 統計有小孩語音的片段
    child_segments = sum(1 for s in segment_infos if s.has_child_speech)
    print(f"    含小孩語音: {child_segments} 個 ({child_segments/len(segment_infos)*100:.1f}%)")
    
    sample_info = {
        'file_id': file_id,
        'audio_path': str(audio_link),
        'rttm_path': str(rttm_path),
        'uem_path': str(uem_path),
        'duration': duration,
        'num_segments': len(segments),
        'num_slices': len(segment_infos),
        'speakers': list(stats.keys())
    }
    
    return sample_info, segment_infos


def main():
    """主程式"""
    print("=" * 60)
    print("Step 1 (V2): 資料準備與切片")
    print("=" * 60)
    print("\n本版本改進：")
    print("  1. 將長音檔切成重疊的小片段")
    print("  2. 確保小孩語音片段被充分採樣")
    print("  3. 自動計算類別權重")
    
    # 建立目錄
    create_directories()
    SLICED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 設定隨機種子
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # 尋找所有資料夾
    data_folders = [
        d for d in RAW_DATA_DIR.iterdir() 
        if d.is_dir() and not d.name.startswith('.')
    ]
    
    print(f"\n找到 {len(data_folders)} 個資料夾:")
    for folder in data_folders:
        print(f"  - {folder.name}")
    
    # 處理每個樣本
    all_samples = []
    all_segments = []
    
    for i, folder in enumerate(data_folders, 1):
        print(f"\n[{i}/{len(data_folders)}] 處理: {folder.name}")
        
        try:
            file_id = folder.name.replace('-', '_')
            
            sample_info, segment_infos = process_single_sample(
                folder,
                PROCESSED_DATA_DIR,
                file_id
            )
            
            all_samples.append(sample_info)
            all_segments.extend(segment_infos)
            print(f"  ✓ 處理完成")
            
        except Exception as e:
            print(f"  ✗ 處理失敗: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_samples:
        print("\n❌ 沒有成功處理任何樣本!")
        return
    
    # 統計
    print(f"\n" + "=" * 60)
    print(f"資料切片統計")
    print("=" * 60)
    print(f"原始檔案數: {len(all_samples)}")
    print(f"總切片數: {len(all_segments)}")
    
    # 計算類別權重
    print(f"\n計算類別權重...")
    if AUTO_COMPUTE_WEIGHTS:
        class_weights = compute_class_weights(all_segments)
        print(f"  自動計算的權重:")
    else:
        class_weights = SPEAKER_WEIGHTS.copy()
        print(f"  使用預設權重:")
    
    from config_v2 import SPEAKER_REVERSE_MAPPING
    for speaker, weight in class_weights.items():
        name = SPEAKER_REVERSE_MAPPING.get(speaker, speaker)
        print(f"    {name}: {weight:.3f}")
    
    # 分割訓練集和驗證集（以片段為單位，但保持同一檔案的片段在同一集合）
    random.shuffle(all_samples)
    
    val_size = max(1, int(len(all_samples) * VAL_RATIO))
    val_sample_ids = set(s['file_id'] for s in all_samples[:val_size])
    
    train_segments = [s for s in all_segments if s.file_id not in val_sample_ids]
    val_segments = [s for s in all_segments if s.file_id in val_sample_ids]
    
    # 如果樣本太少，兩個集合都使用全部
    if len(all_samples) <= 2:
        train_segments = all_segments
        val_segments = all_segments
    
    print(f"\n資料集分割:")
    print(f"  訓練集: {len(train_segments)} 個片段")
    print(f"  驗證集: {len(val_segments)} 個片段")
    
    # 儲存片段資訊
    segments_json = PROCESSED_DATA_DIR / "segments_v2.json"
    with open(segments_json, 'w', encoding='utf-8') as f:
        json.dump({
            'train': [asdict(s) for s in train_segments],
            'val': [asdict(s) for s in val_segments],
            'config': {
                'segment_duration': SEGMENT_DURATION,
                'segment_overlap': SEGMENT_OVERLAP,
                'min_speech_ratio': MIN_SPEECH_RATIO,
                'require_child_speech': REQUIRE_CHILD_SPEECH,
                'min_child_ratio': MIN_CHILD_RATIO,
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 已儲存片段資訊: {segments_json}")
    
    # 儲存類別權重
    weights_json = PROCESSED_DATA_DIR / "class_weights.json"
    with open(weights_json, 'w', encoding='utf-8') as f:
        json.dump(class_weights, f, ensure_ascii=False, indent=2)
    print(f"✓ 已儲存類別權重: {weights_json}")
    
    # 儲存樣本清單（相容舊版）
    samples_json = PROCESSED_DATA_DIR / "samples_v2.json"
    with open(samples_json, 'w', encoding='utf-8') as f:
        json.dump({
            'train': [s for s in all_samples if s['file_id'] not in val_sample_ids],
            'val': [s for s in all_samples if s['file_id'] in val_sample_ids],
            'total_train_segments': len(train_segments),
            'total_val_segments': len(val_segments),
        }, f, ensure_ascii=False, indent=2)
    print(f"✓ 已儲存樣本清單: {samples_json}")
    
    # 最終統計
    total_duration = sum(s['duration'] for s in all_samples)
    train_child_count = sum(1 for s in train_segments if s.has_child_speech)
    val_child_count = sum(1 for s in val_segments if s.has_child_speech)
    
    print(f"\n" + "=" * 60)
    print(f"✅ 資料準備完成!")
    print("=" * 60)
    print(f"  原始檔案: {len(all_samples)} 個")
    print(f"  原始總時長: {total_duration / 60:.1f} 分鐘")
    print(f"  ")
    print(f"  訓練片段: {len(train_segments)} 個")
    print(f"    - 含小孩語音: {train_child_count} ({train_child_count/len(train_segments)*100:.1f}%)")
    print(f"  驗證片段: {len(val_segments)} 個")
    print(f"    - 含小孩語音: {val_child_count} ({val_child_count/len(val_segments)*100:.1f}%)")
    print(f"  ")
    print(f"  有效訓練資料量提升: {len(train_segments) / len(all_samples):.1f} 倍")
    print(f"  ")
    print(f"  輸出目錄: {PROCESSED_DATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
