import os
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline
import math

# 檢查設備
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🎧 使用裝置: {device}")

# 1️⃣ 載入 Whisper 模型
# 根據您的顯卡記憶體，可以調整 batch_size
batch_size = 16
compute_type = "float16"  # 如果是較舊的顯卡（如 GTX 10系列）可能需要改成 "int8"

print("正在載入 Whisper 模型...")
model = whisperx.load_model("large-v3", device=device, compute_type=compute_type, language="zh")

# 設定檔案路徑
audio_file = r"D:\自閉症\data\TD\20250602-20086013陳弈安\20250602-20086013陳弈安.mp3"

# 2️⃣ 語音轉錄
print(f"開始轉錄檔案: {audio_file} ...")
result = model.transcribe(audio_file, batch_size=batch_size)
print("初步轉錄完成。")

# 3️⃣ 對齊文字 (Forced Alignment)
# 為了獲得更精確的時間戳記
print("正在載入對齊模型...")
align_model, metadata = whisperx.load_align_model(
    language_code=result["language"], device=device
)
print("開始時間對齊...")
result_aligned = whisperx.align(result["segments"], align_model, metadata, audio_file, device,
                                return_char_alignments=False)
print("時間對齊完成。")

# 4️⃣ 說話人分離 (Diarization)
print("開始說話人分離 (這可能需要一點時間)...")
# 請確保 token 是正確且有權限的
diarize_model = DiarizationPipeline(
    use_auth_token=os.environ.get("HF_TOKEN"),
    device=device
)
diarize_segments = diarize_model(audio_file)
print("說話人分離完成。")

# 5️⃣ 合併說話人與文字
print("正在合併說話人標籤...")
result_with_speaker = whisperx.assign_word_speakers(diarize_segments, result_aligned)

# ---------------------------------------------------------
# 6️⃣ 輸出符合標準 SRT 格式的檔案 (修改部分)
# ---------------------------------------------------------

output_path = r"D:\自閉症\data\TD\20250602-20086013陳弈安\陳弈安-whisperX.srt"


def format_to_srt_time(seconds):
    """將秒數轉換為 SRT 時間格式 HH:MM:SS,mmm"""
    if seconds is None:
        return "00:00:00,000"

    millis = int((seconds - int(seconds)) * 1000)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


print(f"正在寫入 SRT 檔案至: {output_path}")

with open(output_path, "w", encoding="utf-8") as f:
    # 使用 enumerate 來產生序號 (index)，從 1 開始
    for i, seg in enumerate(result_with_speaker["segments"], start=1):
        # 取得資訊
        start_time = seg.get("start", 0)
        end_time = seg.get("end", 0)
        text = seg["text"].strip()
        speaker = seg.get("speaker", "Unknown")  # 如果沒抓到語者，預設顯示 Unknown

        # 格式化時間
        srt_start = format_to_srt_time(start_time)
        srt_end = format_to_srt_time(end_time)

        # 寫入 SRT 格式
        # 1. 序號
        f.write(f"{i}\n")
        # 2. 時間軸 (00:00:00,000 --> 00:00:05,000)
        f.write(f"{srt_start} --> {srt_end}\n")
        # 3. 內容 (SPEAKER_00: 文字內容)
        f.write(f"{speaker}: {text}\n")
        # 4. 空行 (區隔下一段)
        f.write("\n")

print(f"✅ 全部完成! 結果已儲存。")