import whisperx
import torch
import re
import csv
import os
import gc

# ================= RTX 5070 Ti 極限優化設定 =================
if torch.cuda.is_available():
    # 啟用 TF32 加速 (Ampere/Ada 架構核心功能)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 啟用 CUDA 圖形優化
    torch.backends.cudnn.benchmark = True

    # 設定記憶體管理策略
    torch.cuda.empty_cache()

    print("🚀 GPU 加速已啟用")
    print(f"   裝置: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    print(f"   TF32: Enabled")
    print(f"   CuDNN Benchmark: Enabled")


def parse_srt_time(time_str):
    """高效能時間解析"""
    time_str = time_str.strip()
    if ',' in time_str:
        time_part, ms_part = time_str.split(',')
        ms = int(ms_part)
    else:
        time_part = time_str
        ms = 0

    h, m, s = map(int, time_part.split(':'))
    return h * 3600 + m * 60 + s + ms / 1000.0


def seconds_to_srt_time(seconds):
    """快速時間格式轉換"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def load_srt_fast(srt_path):
    """優化的 SRT 載入器 - 減少記憶體分配"""
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    segments = []
    blocks = re.split(r'\n\s*\n', content.strip())

    for block in blocks:
        lines = [line.strip() for line in block.split('\n') if line.strip()]

        if len(lines) < 3:
            continue

        # 找時間軸
        time_line = lines[1]
        if '-->' not in time_line:
            continue

        # 快速解析時間
        time_parts = time_line.split('-->')
        start = parse_srt_time(time_parts[0])
        end = parse_srt_time(time_parts[1])

        # 處理文字行
        for text in lines[2:]:
            if text:
                segments.append({
                    'text': text,
                    'start': start,
                    'end': end
                })

    return segments


def batch_segments(segments, max_batch_duration=30.0):
    """
    智能批次分組 - 關鍵優化
    將相鄰片段合併成批次，充分利用 GPU 並行處理能力
    """
    if not segments:
        return []

    batches = []
    current_batch = []
    current_duration = 0

    for seg in segments:
        seg_duration = seg['end'] - seg['start']

        # 如果加入這個片段會超過批次上限，則開始新批次
        if current_duration + seg_duration > max_batch_duration and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_duration = 0

        current_batch.append(seg)
        current_duration += seg_duration

    if current_batch:
        batches.append(current_batch)

    return batches


def forced_alignment_optimized(audio_path, srt_path, output_srt_path, output_csv_path=None):
    """
    RTX 5070 Ti 優化版 Forced Alignment
    主要優化：
    1. 批次處理 - 減少 GPU 啟動開銷
    2. TF32 加速 - 利用 Tensor Core
    3. 混合精度 - FP16 運算
    4. 記憶體優化 - 減少 CPU-GPU 傳輸
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        print("⚠️ 未偵測到 CUDA，將使用 CPU（速度較慢）")

    print("\n" + "=" * 60)
    print("階段 1/5: 載入 SRT 檔案")
    print("=" * 60)

    gt_segments = load_srt_fast(srt_path)
    print(f"✓ 載入 {len(gt_segments)} 個文字片段")

    if len(gt_segments) == 0:
        print("❌ 錯誤: SRT 檔案無有效內容")
        return None

    # 顯示前 3 個範例
    print("\n前 3 個片段:")
    for i, seg in enumerate(gt_segments[:3], 1):
        print(f"  [{i}] {seg['text'][:60]}...")

    print("\n" + "=" * 60)
    print("階段 2/5: 載入音訊")
    print("=" * 60)

    audio = whisperx.load_audio(audio_path)
    audio_duration = audio.shape[0] / 16000
    print(f"✓ 音訊長度: {audio_duration:.2f} 秒 ({audio_duration / 60:.1f} 分鐘)")

    print("\n" + "=" * 60)
    print("階段 3/5: 載入對齊模型")
    print("=" * 60)

    # 使用混合精度載入模型
    with torch.cuda.amp.autocast(enabled=(device == "cuda")):
        model_a, metadata = whisperx.load_align_model(
            language_code="zh",
            device=device
        )
    print("✓ Wav2Vec2 中文模型已載入")

    print("\n" + "=" * 60)
    print("階段 4/5: 批次對齊處理 (GPU 加速)")
    print("=" * 60)

    # 準備對齊輸入
    align_segments = [
        {"text": seg['text'], "start": seg['start'], "end": seg['end']}
        for seg in gt_segments
    ]

    # 批次分組 (關鍵優化)
    batches = batch_segments(align_segments, max_batch_duration=30.0)
    print(f"✓ 已分成 {len(batches)} 個批次進行處理")

    all_aligned = []

    # 批次處理
    for batch_idx, batch in enumerate(batches, 1):
        print(f"  處理批次 {batch_idx}/{len(batches)} ({len(batch)} 個片段)...", end=' ')

        try:
            # GPU 批次對齊
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                result = whisperx.align(
                    batch,
                    model_a,
                    metadata,
                    audio,
                    device,
                    return_char_alignments=False  # 關鍵：關閉字元對齊加速
                )

            all_aligned.extend(result["segments"])
            print("✓")

        except Exception as e:
            print(f"✗ 錯誤: {e}")
            # 失敗時保留原始時間
            all_aligned.extend(batch)

    print(f"✓ 總共對齊 {len(all_aligned)} 個片段")

    print("\n" + "=" * 60)
    print("階段 5/5: 輸出結果")
    print("=" * 60)

    # 寫入 SRT
    with open(output_srt_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(all_aligned, 1):
            f.write(f"{i}\n")
            f.write(f"{seconds_to_srt_time(seg['start'])} --> {seconds_to_srt_time(seg['end'])}\n")
            f.write(f"{seg['text'].strip()}\n\n")

    print(f"✓ SRT: {output_srt_path}")

    # 寫入 CSV
    if output_csv_path:
        with open(output_csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['序號', '開始(秒)', '結束(秒)', '開始(SRT)', '結束(SRT)', '時長', '內容'])

            for i, seg in enumerate(all_aligned, 1):
                duration = seg['end'] - seg['start']
                writer.writerow([
                    i,
                    f"{seg['start']:.3f}",
                    f"{seg['end']:.3f}",
                    seconds_to_srt_time(seg['start']),
                    seconds_to_srt_time(seg['end']),
                    f"{duration:.3f}",
                    seg['text'].strip()
                ])

        print(f"✓ CSV: {output_csv_path}")

    # 統計資訊
    print("\n" + "=" * 60)
    print("📊 統計資訊")
    print("=" * 60)
    print(f"  輸入: {len(gt_segments)} 個片段")
    print(f"  輸出: {len(all_aligned)} 個片段")
    print(f"  音訊: {audio_duration:.2f} 秒")

    if all_aligned:
        avg_dur = sum(s['end'] - s['start'] for s in all_aligned) / len(all_aligned)
        print(f"  平均片段長度: {avg_dur:.2f} 秒")

    print("=" * 60)

    # 顯示前 5 個結果
    print("\n✅ 前 5 個對齊結果:")
    for i, seg in enumerate(all_aligned[:5], 1):
        dur = seg['end'] - seg['start']
        print(f"\n[{i}] {seconds_to_srt_time(seg['start'])} --> {seconds_to_srt_time(seg['end'])} ({dur:.2f}s)")
        print(f"    {seg['text'][:70]}{'...' if len(seg['text']) > 70 else ''}")

    # 清理 GPU 記憶體
    del model_a, audio
    gc.collect()
    torch.cuda.empty_cache()

    print("\n✅ 完成！GPU 記憶體已釋放")

    return all_aligned


# ============ 主程式 ============
if __name__ == "__main__":
    import time

    # 設定檔案路徑
    AUDIO_FILE = r"C:\自閉症\data\TD\20250210-19683409吳苡恩\20250210-19683409吳苡恩.mp3"
    SRT_FILE = r"C:\自閉症\data\TD\20250210-19683409吳苡恩\GTruth.srt"
    OUTPUT_SRT = r"C:\自閉症\data\TD\20250210-19683409吳苡恩\Aegisub.srt"
    OUTPUT_CSV = r"C:\自閉症\data\TD\20250210-19683409吳苡恩\Aegisub.csv"

    print("=" * 60)
    print("WhisperX Forced Alignment - RTX 5070 Ti 優化版")
    print("=" * 60)

    # 檢查檔案
    if not os.path.exists(AUDIO_FILE):
        print(f"❌ 找不到: {AUDIO_FILE}")
        exit(1)

    if not os.path.exists(SRT_FILE):
        print(f"❌ 找不到: {SRT_FILE}")
        exit(1)

    print(f"📁 音訊: {os.path.basename(AUDIO_FILE)}")
    print(f"📁 SRT: {os.path.basename(SRT_FILE)}")

    # 開始計時
    start_time = time.time()

    try:
        results = forced_alignment_optimized(
            audio_path=AUDIO_FILE,
            srt_path=SRT_FILE,
            output_srt_path=OUTPUT_SRT,
            output_csv_path=OUTPUT_CSV
        )

        # 計算耗時
        elapsed = time.time() - start_time

        if results:
            print("\n" + "=" * 60)
            print(f"⚡ 總耗時: {elapsed:.2f} 秒 ({elapsed / 60:.1f} 分鐘)")
            print("=" * 60)

    except Exception as e:
        print(f"\n❌ 執行失敗: {e}")
        import traceback

        traceback.print_exc()