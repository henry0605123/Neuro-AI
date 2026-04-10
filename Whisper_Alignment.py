import os
import torch
import whisperx
import gc
from pathlib import Path
from datetime import timedelta
from pyannote.audio import Pipeline
import torchaudio


def format_timestamp(seconds):
    """將秒數轉換為 SRT 格式的時間戳記 (HH:MM:SS,mmm)"""
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60
    millis = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def load_audio_for_vad(audio_path):
    """載入音訊供 pyannote.audio 使用"""
    waveform, sample_rate = torchaudio.load(str(audio_path))
    # 轉換為單聲道
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    # 重新採樣到 16kHz（如果需要）
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    return waveform, 16000


def advanced_vad(audio_path, hf_token):
    """
    使用 pyannote.audio SOTA VAD 模型進行語音活動偵測

    模型: pyannote/segmentation-3.0 (SOTA)
    """
    print("執行 SOTA VAD（pyannote.audio segmentation-3.0）...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 載入 pyannote.audio VAD pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/voice-activity-detection",
        use_auth_token=hf_token
    )
    pipeline.to(torch.device(device))

    # 執行 VAD
    vad_result = pipeline(str(audio_path))

    # 轉換為時間區間列表
    speech_segments = []
    for segment in vad_result.get_timeline().support():
        speech_segments.append({
            'start': segment.start,
            'end': segment.end
        })

    print(f"✓ VAD 偵測到 {len(speech_segments)} 個語音區段")

    return speech_segments


def transcribe_with_whisperx_sota(audio_path, model_name="large-v3", language="zh", hf_token=None,
                                  use_advanced_vad=False):
    """
    使用 SOTA 模型組合進行語音轉錄

    轉錄模型: OpenAI Whisper large-v3 (SOTA)
    VAD 模型: pyannote.audio segmentation-3.0 (SOTA) 或 Silero VAD
    對齊模型: wav2vec2 (SOTA for alignment)

    參數:
        audio_path: 音訊檔案路徑
        model_name: Whisper 模型 (large-v3 為 SOTA)
        language: 語言代碼
        hf_token: Hugging Face Token (pyannote VAD 需要)
        use_advanced_vad: 是否使用 pyannote.audio SOTA VAD
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"\n{'=' * 60}")
    print(f"使用裝置: {device}, 計算類型: {compute_type}")
    print(f"{'=' * 60}\n")

    # ========== Step 1: SOTA VAD（可選） ==========
    vad_segments = None
    if use_advanced_vad and hf_token:
        vad_segments = advanced_vad(audio_path, hf_token)

    # ========== Step 2: 載入 Whisper large-v3 (SOTA 轉錄) ==========
    print(f"載入 Whisper {model_name} 模型（SOTA 轉錄模型）...")

    # 設定 VAD 選項（句子級分段）
    vad_options = None
    if not use_advanced_vad:
        vad_options = {
            "vad_onset": 0.500,  # 語音開始閾值
            "vad_offset": 0.363,  # 語音結束閾值
            "min_silence_duration_ms": 10,  # 最小靜音時長 300ms（句子間停頓）
            "min_speech_duration_ms": 250,  # 最小語音片段 250ms
            "speech_pad_ms": 10  # 語音片段前後填充 30ms
        }

    model = whisperx.load_model(
        model_name,
        device,
        compute_type=compute_type,
        language=language,
        asr_options={"suppress_numerals": True},
        vad_options=vad_options
    )

    # ========== Step 3: 載入音訊 ==========
    print(f"載入音訊: {audio_path}")
    audio = whisperx.load_audio(str(audio_path))

    # ========== Step 4: 執行轉錄 ==========
    print("執行語音轉錄（Whisper large-v3）...")
    result = model.transcribe(
        audio,
        batch_size=16,
        language=language
    )

    print(f"✓ 轉錄完成，偵測到 {len(result['segments'])} 個語音片段")

    # 清理記憶體
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ========== Step 5: 載入 SOTA Forced Alignment 模型 ==========
    print(f"\n載入 Forced Alignment 模型（wav2vec2 for {language}）...")
    model_a, metadata = whisperx.load_align_model(
        language_code=language,
        device=device
    )

    # ========== Step 6: 執行 SOTA Forced Alignment ==========
    print("執行 Forced Alignment（精確詞級對齊）...")
    aligned_result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False
    )

    print("✓ Forced Alignment 完成！")

    # 清理記憶體
    del model_a
    gc.collect()
    torch.cuda.empty_cache()

    return aligned_result


def create_srt(segments, output_path):
    """
    將對齊結果轉換為 SRT 字幕格式（一句話一個序號）
    """

    with open(output_path, 'w', encoding='utf-8') as f:
        seq_num = 1

        for segment in segments:
            if 'start' not in segment or 'end' not in segment:
                continue

            # SRT 序號
            f.write(f"{seq_num}\n")

            # 時間軸
            start_time = format_timestamp(segment['start'])
            end_time = format_timestamp(segment['end'])
            f.write(f"{start_time} --> {end_time}\n")

            # 字幕內容
            text = segment['text'].strip()
            f.write(f"{text}\n\n")

            seq_num += 1

    print(f"✓ SRT 字幕已儲存: {output_path}")


def display_word_level_alignment(segments, num_segments=2, num_words=10):
    """顯示詞級對齊資訊"""
    print(f"\n{'=' * 60}")
    print("詞級對齊範例（前 {} 個片段）".format(num_segments))
    print(f"{'=' * 60}")

    for i, segment in enumerate(segments[:num_segments], 1):
        print(f"\n片段 {i}: {segment['text']}")
        print("-" * 60)

        if 'words' in segment:
            for word_info in segment['words'][:num_words]:
                word = word_info.get('word', 'N/A')
                start = word_info.get('start', 0)
                end = word_info.get('end', 0)
                score = word_info.get('score', 0)
                print(f"  '{word}' -> [{start:.3f}s - {end:.3f}s] (信心分數: {score:.3f})")


def main():
    """主程式"""

    # ========== 設定區 ==========
    # 音訊檔案路徑
    audio_path = r"C:\自閉症\data\TD\20250324-20054665陳芮晞\20250324-20054665陳芮晞.mp3"  # 請修改

    # 輸出檔案
    output_srt = r"C:\自閉症\data\TD\20250324-20054665陳芮晞\Whisper_alignment.srt"

    # 模型選擇（SOTA 配置）
    model_name = "large-v3"  # Whisper SOTA 模型
    language = "zh"  # zh=中文, en=英文

    # Hugging Face Token（必須，用於 pyannote.audio）；請設定環境變數 HF_TOKEN
    # 取得方式: https://huggingface.co/settings/tokens
    # 並接受使用條款: https://huggingface.co/pyannote/segmentation-3.0
    hf_token = os.environ.get("HF_TOKEN")

    # 是否使用 pyannote.audio SOTA VAD（需要 HF token）
    use_advanced_vad = False  # 設為 True 使用 pyannote VAD
    # ============================

    # 檢查檔案
    if not Path(audio_path).exists():
        print(f"❌ 錯誤: 找不到音訊檔案 '{audio_path}'")
        return

    # 檢查 HF token
    if use_advanced_vad and not hf_token:
        print("⚠️  警告: 請設定您的 Hugging Face Token")
        print("取得方式: https://huggingface.co/settings/tokens")
        print("並接受使用條款: https://huggingface.co/pyannote/segmentation-3.0")
        return

    print("\n" + "=" * 60)
    print("SOTA 語音處理流程")
    print("=" * 60)
    print(f"轉錄模型: OpenAI Whisper {model_name}")
    print(f"VAD 模型: {'pyannote.audio segmentation-3.0 (SOTA)' if use_advanced_vad else 'Silero VAD (內建)'}")
    print(f"對齊模型: wav2vec2-based forced alignment (SOTA)")
    print("=" * 60 + "\n")

    # 執行 SOTA 轉錄 + 對齊
    aligned_result = transcribe_with_whisperx_sota(
        audio_path,
        model_name=model_name,
        language=language,
        hf_token=hf_token if use_advanced_vad else None,
        use_advanced_vad=use_advanced_vad
    )

    segments = aligned_result['segments']
    print(f"\n✓ 總共對齊 {len(segments)} 個語音片段")

    # 產生 SRT 字幕
    create_srt(segments, output_srt)

    # 顯示轉錄預覽
    print(f"\n{'=' * 60}")
    print("轉錄預覽（前 3 個片段）")
    print(f"{'=' * 60}")
    for i, seg in enumerate(segments[:3], 1):
        if 'start' in seg and 'end' in seg:
            print(f"\n{i}. [{seg['start']:.2f}s - {seg['end']:.2f}s]")
            print(f"   {seg['text']}")

    # 顯示詞級對齊
    display_word_level_alignment(segments, num_segments=2, num_words=10)

    print(f"\n{'=' * 60}")
    print("✅ 處理完成！")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()