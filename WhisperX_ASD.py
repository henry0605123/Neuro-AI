import torch
import whisperx
from faster_whisper import WhisperModel
import warnings
import gc
import numpy as np
import torchaudio
import tempfile
import os
from collections import Counter

# 關閉警告
warnings.filterwarnings("ignore", category=UserWarning)

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
print(f"🎧 使用裝置: {device}")

audio_file = r"C:\自閉症\data\TD\20250210-19683409吳苡恩\20250210-19683409吳苡恩.mp3"
my_prompt = "這是一段兩個人的對話"
print(f"🧬 使用提示詞: {my_prompt}")

# -------------------------------
# 1️⃣ Faster-Whisper 模型
# -------------------------------
print("載入 Faster-Whisper 模型...")
model = WhisperModel("large-v3", device=device, compute_type=compute_type, language="zh")
print("模型載入完成。")

# -------------------------------
# 2️⃣ 使用 Silero VAD 模型
# -------------------------------
print("🧩 載入 Silero VAD 模型並進行語音活動偵測...")
silero_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    verbose=False,
    trust_repo=True
)

get_speech_timestamps_fn = utils[0]

# 載入音訊
wav, sr = torchaudio.load(audio_file)
if sr != 16000:
    wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)
    sr = 16000

# 偵測語音段落並合併接近的片段
timestamps = get_speech_timestamps_fn(
    wav[0],
    silero_model,
    sampling_rate=sr,
    threshold=0.5,
    min_silence_duration_ms=500
)
print(f"✅ 偵測到 {len(timestamps)} 段語音片段。")

# -------------------------------
# 3️⃣ 分段轉錄（改進版）
# -------------------------------
all_segments = []
temp_dir = tempfile.mkdtemp()


def detect_repetition(text, max_repeat=3):
    """檢測並移除重複文字"""
    words = text.split()
    if len(words) < 2:
        return text

    # 檢測連續重複的字詞
    cleaned_words = []
    i = 0
    while i < len(words):
        word = words[i]
        count = 1
        # 計算連續重複次數
        while i + count < len(words) and words[i + count] == word:
            count += 1
        # 只保留最多 max_repeat 次
        cleaned_words.extend([word] * min(count, max_repeat))
        i += count

    return ' '.join(cleaned_words)


try:
    for i, seg in enumerate(timestamps):
        start_sample = seg["start"]
        end_sample = seg["end"]
        start_sec = start_sample / sr
        end_sec = end_sample / sr
        duration = end_sec - start_sec

        # 🔥 跳過太短的片段（可能是噪音）
        if duration < 0.5:
            print(f"⏭️ 跳過太短片段 {i + 1}: {duration:.2f}s")
            continue

        # 🔥 限制單段最長時間，避免幻覺
        max_segment_duration = 30  # 最長 30 秒
        if duration > max_segment_duration:
            print(f"⚠️ 片段 {i + 1} 太長 ({duration:.2f}s)，將分割處理")
            # 可以選擇分割或跳過
            continue

        print(f"🎙️ 轉錄第 {i + 1}/{len(timestamps)} 段: {start_sec:.2f}s ~ {end_sec:.2f}s ({duration:.2f}s)")

        # 切割音訊片段
        audio_segment = wav[:, start_sample:end_sample]

        # 儲存為臨時檔案
        temp_audio_path = os.path.join(temp_dir, f"segment_{i}.wav")
        torchaudio.save(temp_audio_path, audio_segment, sr)

        # 🔥 改進的轉錄參數
        segment_result, _ = model.transcribe(
            temp_audio_path,
            language="zh",
            initial_prompt=my_prompt,
            condition_on_previous_text=False,
            vad_filter=False,
            beam_size=5,  # 增加 beam size 提高準確度
            best_of=5,
            temperature=0.0,  # 降低隨機性
            compression_ratio_threshold=2.4,  # 檢測重複內容
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6
        )

        # 將轉錄結果加上原始時間戳記並處理文字
        for s in segment_result:
            text = s.text.strip()

            # 🔥 檢測並移除重複
            text = detect_repetition(text, max_repeat=2)

            # 🔥 過濾太短或空白的結果
            if len(text) > 0 and not text.isspace():
                all_segments.append({
                    "start": start_sec + s.start,
                    "end": start_sec + s.end,
                    "text": text
                })

        # 刪除臨時檔案
        os.remove(temp_audio_path)

finally:
    # 清理臨時目錄
    try:
        os.rmdir(temp_dir)
    except:
        pass

print(f"📝 所有分段轉錄完成,共 {len(all_segments)} 個片段。")

# -------------------------------
# 4️⃣ 說話人分離
# -------------------------------
print("🗣️ 開始說話人分離...")
from whisperx.diarize import DiarizationPipeline

diarize_model = DiarizationPipeline(
    use_auth_token=os.environ.get("HF_TOKEN"),
    device=device
)

# 強制指定為 2 個說話人
diarize_segments = diarize_model(
    audio_file,
    min_speakers=2,
    max_speakers=2
)
print("✅ 說話人分離完成。")

# -------------------------------
# 5️⃣ 改進的說話人分配
# -------------------------------
print("🔗 合併說話人資訊...")

# 將 diarization DataFrame 轉成 timeline list
speaker_timeline = []
for _, row in diarize_segments.iterrows():
    speaker_timeline.append({
        "start": row["start"],
        "end": row["end"],
        "speaker": row["speaker"]
    })

# 統計說話人標籤
all_speakers = [sp["speaker"] for sp in speaker_timeline]
speaker_counts = Counter(all_speakers)
top_2_speakers = [sp for sp, _ in speaker_counts.most_common(2)]

print(f"📊 檢測到的說話人: {speaker_counts}")
print(f"✅ 保留前兩位說話人: {top_2_speakers}")

# 重新映射說話人標籤
speaker_map = {}
if len(top_2_speakers) >= 1:
    speaker_map[top_2_speakers[0]] = "SPEAKER_00"
if len(top_2_speakers) >= 2:
    speaker_map[top_2_speakers[1]] = "SPEAKER_01"

# 過濾並重新標記說話人時間軸
filtered_timeline = []
for sp in speaker_timeline:
    if sp["speaker"] in top_2_speakers:
        filtered_timeline.append({
            "start": sp["start"],
            "end": sp["end"],
            "speaker": speaker_map[sp["speaker"]]
        })

# 🔥 改進的說話人分配算法
for seg in all_segments:
    assigned_speaker = "Unknown"
    max_overlap = 0

    # 計算與每個說話人片段的重疊時間
    for sp in filtered_timeline:
        overlap_start = max(seg["start"], sp["start"])
        overlap_end = min(seg["end"], sp["end"])
        overlap = max(0, overlap_end - overlap_start)

        if overlap > max_overlap:
            max_overlap = overlap
            assigned_speaker = sp["speaker"]

    # 🔥 如果重疊時間太少，使用最近的說話人
    if max_overlap < 0.1:  # 重疊少於 0.1 秒
        seg_mid = (seg["start"] + seg["end"]) / 2
        min_distance = float('inf')

        for sp in filtered_timeline:
            sp_mid = (sp["start"] + sp["end"]) / 2
            distance = abs(seg_mid - sp_mid)

            if distance < min_distance:
                min_distance = distance
                assigned_speaker = sp["speaker"]

    seg["speaker"] = assigned_speaker

# -------------------------------
# 6️⃣ 輸出 SRT 檔案
# -------------------------------
output_path = r"C:\自閉症\data\TD\20250210-19683409吳苡恩\吳苡恩-whisperX.srt"


def format_srt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


with open(output_path, "w", encoding="utf-8") as f:
    for idx, seg in enumerate(all_segments, start=1):
        speaker = seg.get("speaker", "Unknown")
        text = seg["text"].strip()
        start = seg.get("start", 0)
        end = seg.get("end", 0)

        f.write(f"{idx}\n")
        f.write(f"{format_srt_time(start)} --> {format_srt_time(end)}\n")
        f.write(f"{speaker}: {text}\n\n")

print(f"✅ 完成! 結果輸出至: {output_path}")

# -------------------------------
# 7️⃣ 清理記憶體
# -------------------------------
del model, silero_model, diarize_model
gc.collect()
torch.cuda.empty_cache()
print("🧹 記憶體清理完成。")