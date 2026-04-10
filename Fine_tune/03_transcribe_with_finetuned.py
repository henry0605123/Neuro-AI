"""
Step 3: 使用微調後的模型進行轉錄
================================
結合 WhisperX 語音轉錄和微調後的 pyannote 語者分離

執行方式:
    python 03_transcribe_with_finetuned.py

可選參數:
    --audio: 指定音檔路徑（預設使用 config.py 中的測試音檔）
    --output: 指定輸出路徑
    --use-original: 使用原始模型（不使用微調模型）進行比較
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import warnings
from pathlib import Path
from datetime import datetime

# 過濾 torchaudio 弃用警告
warnings.filterwarnings('ignore', category=UserWarning, module='torchaudio')
warnings.filterwarnings('ignore', category=UserWarning, module='pyannote')

# 加入專案路徑
sys.path.insert(0, str(Path(__file__).parent))

from config import (
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


def run_diarization_with_padding(pipeline, audio_input, min_speakers, max_speakers):
    """
    執行語者分離，處理不同長度片段的問題
    
    透過修改 pipeline 參數來避免批次處理中的長度不匹配
    """
    # 修改 embedding 模型的批次大小為 1，強制逐個處理
    if hasattr(pipeline, '_embedding') and hasattr(pipeline._embedding, 'batch_size'):
        pipeline._embedding.batch_size = 1
    
    # 修改 segmentation 模型的批次大小
    if hasattr(pipeline, '_segmentation') and hasattr(pipeline._segmentation, 'batch_size'):
        pipeline._segmentation.batch_size = 1
    
    # 嘗試使用修改後的 pipeline
    try:
        return pipeline(
            audio_input,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
    except RuntimeError as e:
        if "Sizes of tensors must match" in str(e):
            # 如果還是失敗，嘗試直接處理音訊
            print("  ⚠️ 使用備用方法處理...")
            
            # 將音訊補零到 10 秒的倍數
            waveform = audio_input["waveform"]
            sample_rate = audio_input["sample_rate"]
            
            # 計算需要補多少零（補到 10 秒的倍數）
            segment_samples = 10 * sample_rate  # 10 秒
            current_length = waveform.shape[-1]
            padding_needed = segment_samples - (current_length % segment_samples)
            
            if padding_needed < segment_samples:
                waveform = F.pad(waveform, (0, padding_needed))
            
            padded_input = {
                "waveform": waveform,
                "sample_rate": sample_rate
            }
            
            return pipeline(
                padded_input,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
        else:
            raise


def format_to_srt_time(seconds: float) -> str:
    """將秒數轉換為 SRT 時間格式 HH:MM:SS,mmm"""
    if seconds is None or seconds < 0:
        seconds = 0
    
    millis = int((seconds - int(seconds)) * 1000)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def map_speaker_label(speaker: str) -> str:
    """
    將 pyannote 的說話人標籤映射為易讀標籤
    
    Args:
        speaker: 原始標籤（如 SPEAKER_00）
        
    Returns:
        映射後的標籤（如 測試者）
    """
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


def load_diarization_pipeline(device: str, use_finetuned: bool = True):
    """
    載入語者分離 pipeline
    
    Args:
        device: 運算裝置
        use_finetuned: 是否使用微調模型
        
    Returns:
        語者分離 pipeline
    """
    from pyannote.audio import Pipeline
    from pyannote.audio import Model
    
    print(f"正在載入語者分離模型...")
    
    # 載入基礎 pipeline
    pipeline = Pipeline.from_pretrained(
        PRETRAINED_DIARIZATION_MODEL,
        use_auth_token=HF_TOKEN
    )
    
    # 如果使用微調模型，替換 segmentation 模型
    if use_finetuned:
        finetuned_model_path = MODEL_DIR / "finetuned_segmentation" / "best_model.pt"
        
        if finetuned_model_path.exists():
            print(f"載入微調模型: {finetuned_model_path}")
            
            # 載入微調後的 segmentation 模型
            try:
                from pyannote.audio import Model
                
                # 先載入預訓練模型的架構
                segmentation_model = Model.from_pretrained(
                    PRETRAINED_SEGMENTATION_MODEL,
                    use_auth_token=HF_TOKEN
                )
                
                # 載入微調後的權重
                checkpoint = torch.load(finetuned_model_path, map_location=device)
                
                # 從 checkpoint 中提取模型權重
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    epoch = checkpoint.get('epoch', '?')
                    val_loss = checkpoint.get('val_loss', '?')
                    print(f"  載入 Epoch {epoch}, Val Loss: {val_loss:.4f}")
                else:
                    state_dict = checkpoint
                
                # 載入權重
                segmentation_model.load_state_dict(state_dict)
                segmentation_model = segmentation_model.to(torch.device(device))
                segmentation_model.eval()
                
                # 替換 pipeline 中的 segmentation 模型
                pipeline._segmentation.model = segmentation_model
                
                print("✓ 已載入微調模型")
                
            except Exception as e:
                print(f"⚠️ 載入微調模型失敗: {e}")
                import traceback
                traceback.print_exc()
                print("使用原始模型繼續...")
        else:
            print(f"⚠️ 找不到微調模型: {finetuned_model_path}")
            print("使用原始模型...")
    else:
        print("使用原始模型（未微調）")
    
    pipeline = pipeline.to(torch.device(device))
    
    print("✓ 語者分離模型載入完成")
    return pipeline


def transcribe_with_diarization(
    audio_path: str,
    output_path: str,
    device: str = "cuda",
    use_finetuned: bool = True,
    skip_alignment: bool = False
):
    """
    執行語音轉錄和語者分離
    
    Args:
        audio_path: 音檔路徑
        output_path: 輸出 SRT 路徑
        device: 運算裝置
        use_finetuned: 是否使用微調模型
        skip_alignment: 是否跳過時間對齊（避免網路下載問題）
    """
    import whisperx
    
    audio_path = Path(audio_path)
    output_path = Path(output_path)
    
    if not audio_path.exists():
        raise FileNotFoundError(f"找不到音檔: {audio_path}")
    
    print(f"\n音檔: {audio_path}")
    print(f"輸出: {output_path}")
    print(f"裝置: {device}")
    print("-" * 60)
    
    # Step 1: 語音轉錄（優化批次大小）
    print("\n[1/4] 語音轉錄...")
    whisper_model = load_whisper_model(device)
    # RTX 5070 Ti 可使用更大的批次大小
    optimized_batch_size = 32 if device == "cuda" else WHISPER_BATCH_SIZE
    # Windows 需要 num_workers=0 避免多進程序列化問題
    result = whisper_model.transcribe(
        str(audio_path), 
        batch_size=optimized_batch_size
    )
    print(f"✓ 轉錄完成，共 {len(result['segments'])} 個段落")
    
    # Step 2: 時間對齊（可選）
    if skip_alignment:
        print("\n[2/4] 跳過時間對齊...")
        result_aligned = result
        print("✓ 使用原始時間戳")
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
            print("  使用原始時間戳繼續...")
            result_aligned = result
    
    # 釋放記憶體
    del whisper_model
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # Step 3: 語者分離（優化效能）
    print("\n[3/4] 語者分離...")
    diarization_pipeline = load_diarization_pipeline(device, use_finetuned)
    
    # 載入音訊並確保正確的格式
    import torchaudio
    waveform, sample_rate = torchaudio.load(str(audio_path))
    
    # 確保是單聲道
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # 重採樣到 16kHz（pyannote 標準）
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # 建立 pyannote 輸入格式
    audio_input = {
        "waveform": waveform,
        "sample_rate": 16000
    }
    
    # 執行語者分離（使用 waveform 輸入避免批次大小問題）
    try:
        diarize_segments = diarization_pipeline(
            audio_input,
            min_speakers=MIN_SPEAKERS,
            max_speakers=MAX_SPEAKERS
        )
    except RuntimeError as e:
        if "Sizes of tensors must match" in str(e):
            print("  ⚠️ 遇到批次大小問題，切換到逐段處理模式...")
            # 修改 embedding 批次大小為 1
            if hasattr(diarization_pipeline, '_embedding'):
                diarization_pipeline._embedding.batch_size = 1
            # 使用 hook 來處理不同長度的片段
            diarize_segments = run_diarization_with_padding(
                diarization_pipeline, 
                audio_input,
                MIN_SPEAKERS,
                MAX_SPEAKERS
            )
        else:
            raise
    
    # 清理記憶體
    del waveform
    torch.cuda.empty_cache() if device == "cuda" else None
    
    print("✓ 語者分離完成")
    
    # 將 pyannote Annotation 轉換為 whisperx 格式
    # whisperx 期望的格式是 pandas DataFrame
    import pandas as pd
    
    diarize_list = []
    for segment, _, speaker in diarize_segments.itertracks(yield_label=True):
        diarize_list.append({
            'start': segment.start,
            'end': segment.end,
            'speaker': speaker
        })
    
    diarize_df = pd.DataFrame(diarize_list)
    
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
            
            # 映射說話人標籤
            speaker_label = map_speaker_label(speaker)
            
            # 寫入 SRT 格式
            f.write(f"{i}\n")
            f.write(f"{format_to_srt_time(start_time)} --> {format_to_srt_time(end_time)}\n")
            f.write(f"{speaker_label}: {text}\n")
            f.write("\n")
    
    print(f"✓ SRT 檔案已儲存: {output_path}")
    
    return result_with_speaker


def main():
    """主程式"""
    parser = argparse.ArgumentParser(description="使用微調模型進行語音轉錄和語者分離")
    
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
        "--use-original",
        action="store_true",
        help="使用原始模型（不使用微調模型）"
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="同時輸出原始模型和微調模型的結果進行比較"
    )
    
    parser.add_argument(
        "--skip-alignment",
        action="store_true",
        help="跳過時間對齊步驟（避免網路下載問題）"
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
        output_path = OUTPUT_DIR / f"{audio_path.stem}_finetuned_{timestamp}.srt"
    
    # 檢查裝置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("Step 3: 使用微調模型進行轉錄")
    print("=" * 60)
    print(f"裝置: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    if args.skip_alignment:
        print("⚠️ 跳過時間對齊步驟")
    
    if args.compare:
        # 比較模式：同時輸出兩種結果
        print("\n📊 比較模式：將同時輸出原始模型和微調模型的結果")
        
        # 原始模型
        print("\n" + "=" * 60)
        print("使用原始模型")
        print("=" * 60)
        original_output = output_path.with_stem(f"{output_path.stem}_original")
        transcribe_with_diarization(
            audio_path,
            original_output,
            device,
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
            use_finetuned=True,
            skip_alignment=args.skip_alignment
        )
        
        print("\n" + "=" * 60)
        print("✅ 比較完成!")
        print(f"  原始模型結果: {original_output}")
        print(f"  微調模型結果: {finetuned_output}")
        print("=" * 60)
        
    else:
        # 單一模式
        use_finetuned = not args.use_original
        
        transcribe_with_diarization(
            audio_path,
            output_path,
            device,
            use_finetuned=use_finetuned,
            skip_alignment=args.skip_alignment
        )
        
        print("\n" + "=" * 60)
        print("✅ 轉錄完成!")
        print(f"  輸出檔案: {output_path}")
        print("=" * 60)


if __name__ == "__main__":
    main()
