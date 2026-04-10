import torch
import whisperx
from whisperx.diarize import DiarizationPipeline
import os
import gc
import torchaudio
from df.enhance import enhance, init_df, save_audio
from df.io import resample

# ---------------------------------------------------------
# 設定區 (請修改這裡的路徑與 Token)
# ---------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")
input_audio_path = r"C:\自閉症\data\20250106\20250106-蔡oo\2025106-蔡oo-拼圖.mp3"
output_srt_path = r"C:\自閉症\data\20250106\20250106-蔡oo\蔡oo-enhance.srt"
enhanced_audio_path = input_audio_path.replace(".mp3", "_enhanced.wav")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🎧 主要運算裝置 (Whisper): {device}")


# 定義一個清理記憶體的函式
def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------
# 1️⃣ 語音增強 (分段處理 + GPU 加速)
# ---------------------------------------------------------
if not os.path.exists(enhanced_audio_path):
    print("\n🚀 步驟 1/5: 語音增強 (使用 GPU 分段處理)...")
    try:
        # 初始化模型並移至 GPU
        model_df, df_state, _ = init_df()
        model_df = model_df.to(device)
        print(f"   DeepFilterNet 模型已載入至 {device}")

        # 載入完整音訊以獲取資訊 (先在 CPU 上處理)
        audio_full, sr_orig = torchaudio.load(input_audio_path)

        # 重採樣到模型需要的採樣率 (在 CPU 上進行)
        target_sr = df_state.sr()
        if sr_orig != target_sr:
            print(f"   重採樣: {sr_orig} Hz -> {target_sr} Hz")
            audio_full = resample(audio_full, sr_orig, target_sr)

        # 轉換為單聲道（如果是立體聲）
        if audio_full.shape[0] > 1:
            audio_full = audio_full.mean(dim=0, keepdim=True)
            print("   已轉換為單聲道")

        # 預處理完成後再移至 GPU
        audio_full = audio_full.to(device)
        print(f"   音訊已準備完成並移至 {device}")

        # 計算分段參數 (每段30秒，重疊5秒)
        segment_duration = 30  # 秒
        overlap_duration = 5  # 秒
        segment_samples = segment_duration * target_sr
        overlap_samples = overlap_duration * target_sr
        hop_samples = segment_samples - overlap_samples

        total_samples = audio_full.shape[1]
        num_segments = (total_samples - overlap_samples + hop_samples - 1) // hop_samples

        print(f"   音訊總長度: {total_samples / target_sr:.1f} 秒")
        print(f"   將分為 {num_segments} 段處理...")

        enhanced_segments = []

        for i in range(num_segments):
            start_idx = i * hop_samples
            end_idx = min(start_idx + segment_samples, total_samples)

            # 提取音訊片段並移到 CPU (因為 df_features 需要 numpy)
            audio_segment = audio_full[:, start_idx:end_idx].cpu()

            # 增強處理 (內部會自動處理 CPU/GPU 轉換)
            with torch.no_grad():
                enhanced_segment = enhance(model_df, df_state, audio_segment)

            # 確保結果在 CPU 上
            if enhanced_segment.device.type != 'cpu':
                enhanced_segment = enhanced_segment.cpu()

            # 如果有重疊區域，使用淡入淡出混合
            if i > 0:
                # 對當前段的開頭部分應用淡入
                fade_samples = overlap_samples // 2
                fade_in = torch.linspace(0, 1, fade_samples).unsqueeze(0).to(enhanced_segment.device)
                enhanced_segment[:, :fade_samples] *= fade_in

                # 對前一段的結尾部分應用淡出並混合
                fade_out = torch.linspace(1, 0, fade_samples).unsqueeze(0).to(enhanced_segment.device)
                overlap_part = enhanced_segments[-1][:, -fade_samples:]
                overlap_part *= fade_out

                # 混合重疊區域
                enhanced_segment[:, :fade_samples] += overlap_part

                # 移除前一段的重疊部分
                enhanced_segments[-1] = enhanced_segments[-1][:, :-fade_samples]

            enhanced_segments.append(enhanced_segment)

            print(f"   進度: {i + 1}/{num_segments} ({(i + 1) / num_segments * 100:.1f}%)")

            # 清理記憶體
            del audio_segment, enhanced_segment
            clean_memory()

        # 合併所有片段
        print("   合併音訊片段...")
        enhanced_audio_full = torch.cat(enhanced_segments, dim=1)

        # 儲存
        save_audio(enhanced_audio_path, enhanced_audio_full, target_sr)
        print(f"✨ 語音增強完成: {enhanced_audio_path}")

        # 清理
        del model_df, df_state, audio_full, enhanced_segments, enhanced_audio_full
        clean_memory()

    except Exception as e:
        print(f"❌ 語音增強失敗: {e}")
        import traceback

        traceback.print_exc()
        enhanced_audio_path = input_audio_path
else:
    print(f"\nℹ️ 跳過增強步驟，使用現有檔案: {enhanced_audio_path}")

# ---------------------------------------------------------
# 2️⃣ WhisperX 轉錄
# ---------------------------------------------------------
print("\n🚀 步驟 2/5: 載入 Whisper 模型 (Large-v3)...")
compute_type = "float16"
batch_size = 8

try:
    model = whisperx.load_model("large-v3", device=device, compute_type=compute_type, language="zh")

    print(f"   開始轉錄...")
    result = model.transcribe(enhanced_audio_path, batch_size=batch_size)
    print("   初步轉錄完成。")

    del model
    clean_memory()
    print("   Whisper 模型已卸載，VRAM 已釋放。")

except Exception as e:
    print(f"❌ 轉錄階段錯誤: {e}")
    import traceback

    traceback.print_exc()
    exit()

# ---------------------------------------------------------
# 3️⃣ 強制對齊 (Alignment)
# ---------------------------------------------------------
print("\n🚀 步驟 3/5: 時間對齊 (Forced Alignment)...")
try:
    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )

    result_aligned = whisperx.align(result["segments"], align_model, metadata, enhanced_audio_path, device,
                                    return_char_alignments=False)
    print("   時間對齊完成。")

    del align_model, metadata
    clean_memory()

except Exception as e:
    print(f"❌ 對齊階段錯誤: {e}")
    import traceback

    traceback.print_exc()
    result_aligned = result

# ---------------------------------------------------------
# 4️⃣ 說話人分離 (Diarization)
# ---------------------------------------------------------
print("\n🚀 步驟 4/5: 說話人分離 (Diarization)...")
try:
    diarize_model = DiarizationPipeline(
        use_auth_token=HF_TOKEN,
        device=device
    )

    diarize_segments = diarize_model(enhanced_audio_path, min_speakers=2, max_speakers=2)
    print("   分離完成。")

    del diarize_model
    clean_memory()

    result_with_speaker = whisperx.assign_word_speakers(diarize_segments, result_aligned)

except Exception as e:
    print(f"❌ 分離階段錯誤: {e}")
    import traceback

    traceback.print_exc()
    exit()


# ---------------------------------------------------------
# 5️⃣ 輸出 SRT
# ---------------------------------------------------------
def format_to_srt_time(seconds):
    if seconds is None: return "00:00:00,000"
    millis = int((seconds - int(seconds)) * 1000)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


print(f"\n🚀 步驟 5/5: 寫入 SRT 至 {output_srt_path}")

with open(output_srt_path, "w", encoding="utf-8") as f:
    for i, seg in enumerate(result_with_speaker["segments"], start=1):
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        text = seg["text"].strip()
        speaker = seg.get("speaker", "Unknown")

        f.write(f"{i}\n")
        f.write(f"{format_to_srt_time(start)} --> {format_to_srt_time(end)}\n")
        f.write(f"{speaker}: {text}\n\n")

print("\n✅ 所有工作已完成！")