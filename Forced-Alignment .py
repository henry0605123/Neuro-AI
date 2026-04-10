import whisperx
import torch
import re
import csv
import os


def parse_srt_time(time_str):
    """解析 SRT 時間格式 (HH:MM:SS,mmm 或 HH:MM:SS)"""
    time_str = time_str.strip()
    # 處理可能的格式: HH:MM:SS,mmm 或 HH:MM:SS
    if ',' in time_str:
        time_part, ms_part = time_str.split(',')
        ms = int(ms_part)
    else:
        time_part = time_str
        ms = 0

    parts = time_part.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = int(parts[2])

    return hours * 3600 + minutes * 60 + seconds + ms / 1000.0


def seconds_to_srt_time(seconds):
    """將秒數轉換為 SRT 時間格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def load_ground_truth_srt_manual(srt_path):
    """手動解析 SRT 檔案，處理多行對話的情況"""
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 分割成各個片段（以連續兩個換行符號分割）
    blocks = re.split(r'\n\s*\n', content.strip())

    segments = []
    for block in blocks:
        lines = [line.strip() for line in block.split('\n') if line.strip()]

        if len(lines) < 3:
            continue

        # 第一行是序號
        seq_num = lines[0]

        # 第二行是時間軸
        time_line = lines[1]
        if '-->' not in time_line:
            continue

        # 解析時間
        time_parts = time_line.split('-->')
        start_time = parse_srt_time(time_parts[0])
        end_time = parse_srt_time(time_parts[1])

        # 剩下的所有行都是文字內容
        text_lines = lines[2:]

        # 將所有對話合併成一個文字片段
        full_text = '\n'.join(text_lines)

        # 也可以選擇逐行分割對話
        for text_line in text_lines:
            if text_line:  # 確保不是空行
                segments.append({
                    'text': text_line,
                    'original_start': start_time,
                    'original_end': end_time,
                    'original_seq': seq_num
                })

    return segments


def forced_alignment(audio_path, srt_path, output_srt_path, output_csv_path=None,
                     combine_by_timeblock=False):
    """
    執行 Forced Alignment

    Args:
        audio_path: 音訊檔案路徑
        srt_path: Ground Truth SRT 檔案路徑
        output_srt_path: 輸出的對齊後 SRT 檔案路徑
        output_csv_path: 輸出的 CSV 檔案路徑（可選）
        combine_by_timeblock: 是否按照原始時間區塊合併對話（False=逐句對齊）
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    print(f"使用裝置: {device}")
    print(f"計算類型: {compute_type}")

    # 1. 載入 Ground Truth 文字
    print("\n[步驟 1] 載入 Ground Truth SRT 檔案...")
    gt_segments = load_ground_truth_srt_manual(srt_path)
    print(f"載入了 {len(gt_segments)} 個文字片段")

    if len(gt_segments) == 0:
        print("❌ 錯誤: 沒有載入任何文字片段，請檢查 SRT 檔案格式")
        return None

    # 顯示前幾個片段供檢查
    print("\n前 5 個載入的片段範例:")
    for i, seg in enumerate(gt_segments[:5], 1):
        print(f"  [{i}] {seg['text'][:50]}{'...' if len(seg['text']) > 50 else ''}")

    # 2. 載入音訊
    print("\n[步驟 2] 載入音訊檔案...")
    audio = whisperx.load_audio(audio_path)
    audio_duration = audio.shape[0] / 16000
    print(f"音訊長度: {audio_duration:.2f} 秒")

    # 3. 載入對齊模型
    print("\n[步驟 3] 載入對齊模型...")
    model_a, metadata = whisperx.load_align_model(
        language_code="zh",
        device=device
    )

    # 4. 準備轉錄結果格式
    print("\n[步驟 4] 準備 Ground Truth 文字進行對齊...")

    if combine_by_timeblock:
        # 選項 A: 按原始時間區塊合併
        print("  模式: 按原始時間區塊合併對話")
        timeblock_dict = {}
        for seg in gt_segments:
            key = (seg['original_start'], seg['original_end'])
            if key not in timeblock_dict:
                timeblock_dict[key] = []
            timeblock_dict[key].append(seg['text'])

        transcript_segments = []
        for (start, end), texts in sorted(timeblock_dict.items()):
            combined_text = ' '.join(texts)
            transcript_segments.append({
                "text": combined_text,
                "start": start,
                "end": end
            })
    else:
        # 選項 B: 逐句對齊（每個對話一個片段）
        print("  模式: 逐句對齊（每句話獨立對齊）")
        transcript_segments = []
        for seg in gt_segments:
            transcript_segments.append({
                "text": seg['text'],
                "start": seg['original_start'],
                "end": seg['original_end']
            })

    print(f"  準備了 {len(transcript_segments)} 個片段進行對齊")

    # 5. 執行強制對齊
    print("\n[步驟 5] 執行強制對齊...")
    print("  這可能需要幾分鐘，請稍候...")

    result = whisperx.align(
        transcript_segments,
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False
    )

    # 6. 輸出為 SRT 格式
    print("\n[步驟 6] 輸出對齊結果...")
    aligned_segments = result["segments"]

    # 生成 SRT 檔案
    srt_output = []
    for i, seg in enumerate(aligned_segments, 1):
        start_time = seconds_to_srt_time(seg['start'])
        end_time = seconds_to_srt_time(seg['end'])
        text = seg['text'].strip()

        srt_output.append(f"{i}")
        srt_output.append(f"{start_time} --> {end_time}")
        srt_output.append(text)
        srt_output.append("")

    with open(output_srt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(srt_output))

    print(f"✓ SRT 檔案已儲存至: {output_srt_path}")

    # 7. 輸出 CSV
    if output_csv_path:
        with open(output_csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['序號', '開始時間(秒)', '結束時間(秒)',
                             '開始時間(SRT)', '結束時間(SRT)', '時長(秒)', '文字內容'])

            for i, seg in enumerate(aligned_segments, 1):
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

        print(f"✓ CSV 檔案已儲存至: {output_csv_path}")

    # 8. 顯示統計資訊
    print("\n" + "=" * 60)
    print("對齊統計:")
    print(f"  輸入片段數: {len(gt_segments)}")
    print(f"  輸出片段數: {len(aligned_segments)}")
    print(f"  音訊總長度: {audio_duration:.2f} 秒")

    # 計算平均片段長度
    avg_duration = sum(seg['end'] - seg['start'] for seg in aligned_segments) / len(aligned_segments)
    print(f"  平均片段長度: {avg_duration:.2f} 秒")
    print("=" * 60)

    # 顯示結果範例
    print("\n前 5 個對齊結果範例:")
    for i, seg in enumerate(aligned_segments[:5], 1):
        duration = seg['end'] - seg['start']
        print(f"\n[{i}] {seconds_to_srt_time(seg['start'])} --> {seconds_to_srt_time(seg['end'])} ({duration:.2f}秒)")
        print(f"    {seg['text'][:80]}{'...' if len(seg['text']) > 80 else ''}")

    return aligned_segments


# ============ 主程式 ============
if __name__ == "__main__":
    # 設定檔案路徑
    AUDIO_FILE = r"C:\自閉症\data\TD\20250210-19683438吳忻芸\20250210-19683438吳忻芸.mp3"  # 修改為你的音訊檔案路徑
    SRT_FILE = r"C:\自閉症\data\TD\20250210-19683438吳忻芸\GTruth.srt" # 修改為你的 SRT 檔案路徑
    OUTPUT_SRT = r"C:\自閉症\data\TD\20250210-19683438吳忻芸\Aegisub.srt"  # 輸出的對齊後 SRT 檔案
    OUTPUT_CSV = r"C:\自閉症\data\TD\20250210-19683438吳忻芸\Aegisub.csv"

    # 對齊模式選擇
    # False = 每句話獨立對齊（推薦，更精確）
    # True = 按原始時間區塊合併後對齊
    COMBINE_MODE = False

    # 檢查檔案
    if not os.path.exists(AUDIO_FILE):
        print(f"❌ 錯誤: 找不到音訊檔案 '{AUDIO_FILE}'")
        exit(1)

    if not os.path.exists(SRT_FILE):
        print(f"❌ 錯誤: 找不到 SRT 檔案 '{SRT_FILE}'")
        exit(1)

    print("開始執行 Forced Alignment...")
    print("=" * 60)

    try:
        results = forced_alignment(
            audio_path=AUDIO_FILE,
            srt_path=SRT_FILE,
            output_srt_path=OUTPUT_SRT,
            output_csv_path=OUTPUT_CSV,
            combine_by_timeblock=COMBINE_MODE
        )

        if results:
            print("\n" + "=" * 60)
            print("✓ 對齊完成!")
            print("=" * 60)

    except Exception as e:
        print(f"\n❌ 執行過程發生錯誤: {str(e)}")
        import traceback

        traceback.print_exc()
