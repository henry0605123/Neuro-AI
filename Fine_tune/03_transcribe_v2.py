"""
Step 3 (V2): 使用微調模型進行轉錄
==================================
結合 WhisperX 語音轉錄和優化後的 pyannote 語者分離

執行方式:
    python 03_transcribe_v2.py --audio <音檔路徑> --output <輸出路徑>

可選參數:
    --audio: 音檔路徑
    --output: 輸出 SRT 路徑
    --model: 模型路徑（預設使用 V2 優化模型）
    --use-original: 使用原始模型進行比較
    --skip-alignment: 跳過時間對齊步驟（避免網路問題）
    --compare: 同時輸出原始和微調模型的結果
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import torchaudio
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore', category=UserWarning)

# 加入專案路徑
sys.path.insert(0, str(Path(__file__).parent))

from config_v2 import (
    MODEL_DIR,
    OUTPUT_DIR,
    HF_TOKEN,
    TEST_AUDIO_FILE,
    WHISPER_MODEL,
    WHISPER_BATCH_SIZE,
    WHISPER_COMPUTE_TYPE,
    MIN_SPEAKERS,
    MAX_SPEAKERS,
    SPEAKER_REVERSE_MAPPING,
    PRETRAINED_DIARIZATION_MODEL,
    PRETRAINED_SEGMENTATION_MODEL,
    create_directories
)


def format_to_srt_time(seconds: float) -> str:
    """將秒數轉換為 SRT 時間格式"""
    if seconds is None or seconds < 0:
        seconds = 0
    
    millis = int((seconds - int(seconds)) * 1000)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def infer_speaker_mapping(diarize_segments) -> dict:
    """
    根據說話時長推斷說話人身份
    
    假設：
    1. 測試者通常說話時間最長（問問題、引導對話）
    2. 小孩說話時間次長（回答問題）
    3. Unknown 說話時間最短（背景聲音、其他人）
    
    Returns:
        dict: 從 pyannote 標籤到實際身份的映射
    """
    from collections import defaultdict
    
    # 計算每個說話人的總時長
    speaker_duration = defaultdict(float)
    for segment, _, speaker in diarize_segments.itertracks(yield_label=True):
        duration = segment.end - segment.start
        speaker_duration[speaker] += duration
    
    # 按時長排序（降序）
    sorted_speakers = sorted(speaker_duration.items(), key=lambda x: -x[1])
    
    print(f"\n  說話人時長分析:")
    for speaker, duration in sorted_speakers:
        print(f"    {speaker}: {duration:.1f} 秒")
    
    # 建立映射
    # 說話最多的 = 測試者，次多的 = 小孩，其他 = Unknown
    mapping = {}
    role_order = ["測試者", "小孩", "Unknown"]
    
    for i, (speaker, duration) in enumerate(sorted_speakers):
        if i < len(role_order):
            mapping[speaker] = role_order[i]
        else:
            mapping[speaker] = "Unknown"
    
    print(f"  自動映射結果:")
    for speaker, role in mapping.items():
        print(f"    {speaker} -> {role}")
    
    return mapping


def map_speaker_label(speaker: str, dynamic_mapping: dict = None) -> str:
    """將 pyannote 的說話人標籤映射為易讀標籤"""
    if dynamic_mapping and speaker in dynamic_mapping:
        return dynamic_mapping[speaker]
    return SPEAKER_REVERSE_MAPPING.get(speaker, speaker)


def load_whisper_model(device: str):
    """載入 WhisperX 模型"""
    import whisperx
    
    print(f"正在載入 Whisper 模型: {WHISPER_MODEL}")
    
    model = whisperx.load_model(
        WHISPER_MODEL,
        device=device,
        compute_type=WHISPER_COMPUTE_TYPE,
        language="zh"
    )
    
    print("✓ Whisper 模型載入完成")
    return model


def load_diarization_pipeline(
    device: str,
    model_path: Path = None,
    use_finetuned: bool = True
):
    """
    載入語者分離 pipeline
    
    Args:
        device: 運算裝置
        model_path: 微調模型路徑（None 則使用預設）
        use_finetuned: 是否使用微調模型
    """
    from pyannote.audio import Pipeline, Model
    
    print(f"正在載入語者分離模型...")
    
    # 載入基礎 pipeline
    pipeline = Pipeline.from_pretrained(
        PRETRAINED_DIARIZATION_MODEL,
        use_auth_token=HF_TOKEN
    )
    
    if use_finetuned:
        # 尋找模型檔案
        if model_path is None:
            # 優先使用 V2 模型
            v2_model = MODEL_DIR / "finetuned_segmentation_v2" / "best_model.pt"
            v1_model = MODEL_DIR / "finetuned_segmentation" / "best_model.pt"
            
            if v2_model.exists():
                model_path = v2_model
                print(f"  使用 V2 優化模型")
            elif v1_model.exists():
                model_path = v1_model
                print(f"  使用 V1 模型")
            else:
                print(f"  ⚠️ 找不到微調模型，使用原始模型")
                use_finetuned = False
        
        if use_finetuned and model_path and model_path.exists():
            print(f"  載入: {model_path.name}")
            
            try:
                # 載入預訓練模型架構
                segmentation_model = Model.from_pretrained(
                    PRETRAINED_SEGMENTATION_MODEL,
                    use_auth_token=HF_TOKEN
                )
                
                # 載入微調權重
                checkpoint = torch.load(model_path, map_location=device)
                
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    epoch = checkpoint.get('epoch', '?')
                    val_loss = checkpoint.get('val_loss', '?')
                    metrics = checkpoint.get('metrics', {})
                    
                    print(f"  Epoch {epoch}, Val Loss: {val_loss:.4f}")
                    
                    # 顯示指標
                    if metrics:
                        for speaker, m in metrics.items():
                            name = SPEAKER_REVERSE_MAPPING.get(speaker, speaker)
                            print(f"    {name}: P={m['precision']:.2%}, R={m['recall']:.2%}")
                else:
                    state_dict = checkpoint
                
                segmentation_model.load_state_dict(state_dict)
                segmentation_model = segmentation_model.to(torch.device(device))
                segmentation_model.eval()
                
                # 替換 pipeline 中的模型
                pipeline._segmentation.model = segmentation_model
                
                print("  ✓ 已載入微調模型")
                
            except Exception as e:
                print(f"  ⚠️ 載入失敗: {e}")
                print("  使用原始模型...")
    else:
        print("  使用原始模型（未微調）")
    
    pipeline = pipeline.to(torch.device(device))
    
    print("✓ 語者分離模型載入完成")
    return pipeline


def transcribe_with_diarization(
    audio_path: str,
    output_path: str,
    device: str = "cuda",
    model_path: Path = None,
    use_finetuned: bool = True,
    skip_alignment: bool = False
):
    """
    執行語音轉錄和語者分離
    """
    import whisperx
    import pandas as pd
    
    audio_path = Path(audio_path)
    output_path = Path(output_path)
    
    if not audio_path.exists():
        raise FileNotFoundError(f"找不到音檔: {audio_path}")
    
    print(f"\n音檔: {audio_path}")
    print(f"輸出: {output_path}")
    print("-" * 60)
    
    # Step 1: 語音轉錄
    print("\n[1/4] 語音轉錄...")
    whisper_model = load_whisper_model(device)
    
    optimized_batch_size = 32 if device == "cuda" else WHISPER_BATCH_SIZE
    result = whisper_model.transcribe(
        str(audio_path),
        batch_size=optimized_batch_size
    )
    print(f"✓ 轉錄完成，共 {len(result['segments'])} 個段落")
    
    # Step 2: 時間對齊（可選）
    if skip_alignment:
        print("\n[2/4] 跳過時間對齊...")
        result_aligned = result
    else:
        print("\n[2/4] 時間對齊...")
        try:
            align_model, metadata = whisperx.load_align_model(
                language_code=result["language"],
                device=device
            )
            result_aligned = whisperx.align(
                result["segments"],
                align_model,
                metadata,
                str(audio_path),
                device,
                return_char_alignments=False
            )
            del align_model
            print("✓ 時間對齊完成")
        except Exception as e:
            print(f"⚠️ 時間對齊失敗: {e}")
            result_aligned = result
    
    # 釋放記憶體
    del whisper_model
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # Step 3: 語者分離
    print("\n[3/4] 語者分離...")
    diarization_pipeline = load_diarization_pipeline(
        device, model_path, use_finetuned
    )
    
    # 載入音訊
    waveform, sample_rate = torchaudio.load(str(audio_path))
    
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    audio_input = {
        "waveform": waveform,
        "sample_rate": 16000
    }
    
    # 執行語者分離
    try:
        diarize_segments = diarization_pipeline(
            audio_input,
            min_speakers=MIN_SPEAKERS,
            max_speakers=MAX_SPEAKERS
        )
    except RuntimeError as e:
        if "Sizes of tensors must match" in str(e):
            print("  ⚠️ 調整批次大小...")
            if hasattr(diarization_pipeline, '_embedding'):
                diarization_pipeline._embedding.batch_size = 1
            
            # 補零處理
            segment_samples = 10 * 16000
            current_length = waveform.shape[-1]
            padding_needed = segment_samples - (current_length % segment_samples)
            
            if padding_needed < segment_samples:
                waveform = F.pad(waveform, (0, padding_needed))
            
            audio_input = {"waveform": waveform, "sample_rate": 16000}
            
            diarize_segments = diarization_pipeline(
                audio_input,
                min_speakers=MIN_SPEAKERS,
                max_speakers=MAX_SPEAKERS
            )
        else:
            raise
    
    del waveform
    torch.cuda.empty_cache() if device == "cuda" else None
    
    print("✓ 語者分離完成")
    
    # 轉換為 DataFrame
    diarize_list = []
    for segment, _, speaker in diarize_segments.itertracks(yield_label=True):
        diarize_list.append({
            'start': segment.start,
            'end': segment.end,
            'speaker': speaker
        })
    
    diarize_df = pd.DataFrame(diarize_list)
    
    # 根據說話時長推斷說話人身份
    dynamic_mapping = infer_speaker_mapping(diarize_segments)
    
    # Step 4: 合併結果
    print("\n[4/4] 合併說話人標籤...")
    result_with_speaker = whisperx.assign_word_speakers(
        diarize_df,
        result_aligned
    )
    print("✓ 合併完成")
    
    # 輸出 SRT 檔案
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n正在寫入 SRT 檔案...")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(result_with_speaker["segments"], start=1):
            start_time = seg.get("start", 0)
            end_time = seg.get("end", 0)
            text = seg["text"].strip()
            speaker = seg.get("speaker", "Unknown")
            
            # 使用動態映射
            speaker_label = map_speaker_label(speaker, dynamic_mapping)
            
            f.write(f"{i}\n")
            f.write(f"{format_to_srt_time(start_time)} --> {format_to_srt_time(end_time)}\n")
            f.write(f"{speaker_label}: {text}\n")
            f.write("\n")
    
    print(f"✓ SRT 檔案已儲存: {output_path}")
    
    # 統計說話人分佈
    print(f"\n說話人統計:")
    speaker_stats = {}
    for seg in result_with_speaker["segments"]:
        speaker = seg.get("speaker", "Unknown")
        speaker_label = map_speaker_label(speaker, dynamic_mapping)
        duration = seg.get("end", 0) - seg.get("start", 0)
        
        if speaker_label not in speaker_stats:
            speaker_stats[speaker_label] = {"count": 0, "duration": 0}
        speaker_stats[speaker_label]["count"] += 1
        speaker_stats[speaker_label]["duration"] += duration
    
    for speaker, stats in speaker_stats.items():
        print(f"  {speaker}: {stats['count']} 段, {stats['duration']:.1f} 秒")
    
    return result_with_speaker


def main():
    """主程式"""
    parser = argparse.ArgumentParser(
        description="使用微調模型進行語音轉錄和語者分離"
    )
    
    parser.add_argument(
        "--audio",
        type=str,
        default=str(TEST_AUDIO_FILE),
        help="音檔路徑"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="輸出 SRT 路徑"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="微調模型路徑"
    )
    
    parser.add_argument(
        "--use-original",
        action="store_true",
        help="使用原始模型（不使用微調模型）"
    )
    
    parser.add_argument(
        "--skip-alignment",
        action="store_true",
        help="跳過時間對齊步驟"
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="同時輸出原始和微調模型結果"
    )
    
    args = parser.parse_args()
    
    # 建立目錄
    create_directories()
    
    # 設定路徑
    audio_path = Path(args.audio)
    
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"{audio_path.stem}_{timestamp}.srt"
    
    model_path = Path(args.model) if args.model else None
    
    # 檢查裝置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("Step 3 (V2): 使用微調模型進行轉錄")
    print("=" * 60)
    print(f"裝置: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    if args.skip_alignment:
        print("⚠️ 跳過時間對齊步驟")
    
    if args.compare:
        # 比較模式
        print("\n📊 比較模式")
        
        # 原始模型
        print("\n" + "=" * 60)
        print("使用原始模型")
        print("=" * 60)
        original_output = output_path.with_stem(f"{output_path.stem}_original")
        transcribe_with_diarization(
            audio_path,
            original_output,
            device,
            model_path=None,
            use_finetuned=False,
            skip_alignment=args.skip_alignment
        )
        
        # 微調模型
        print("\n" + "=" * 60)
        print("使用微調模型")
        print("=" * 60)
        finetuned_output = output_path.with_stem(f"{output_path.stem}_finetuned")
        transcribe_with_diarization(
            audio_path,
            finetuned_output,
            device,
            model_path=model_path,
            use_finetuned=True,
            skip_alignment=args.skip_alignment
        )
        
        print("\n" + "=" * 60)
        print("✅ 比較完成!")
        print(f"  原始模型: {original_output}")
        print(f"  微調模型: {finetuned_output}")
        print("=" * 60)
        
    else:
        # 單一模式
        use_finetuned = not args.use_original
        
        transcribe_with_diarization(
            audio_path,
            output_path,
            device,
            model_path=model_path,
            use_finetuned=use_finetuned,
            skip_alignment=args.skip_alignment
        )
        
        print("\n" + "=" * 60)
        print("✅ 轉錄完成!")
        print(f"  輸出檔案: {output_path}")
        print("=" * 60)


if __name__ == "__main__":
    main()
