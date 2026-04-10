import re
import os

# ================= 設定區 =================
# 請輸入你的 SRT 檔案名稱
input_file = r"C:\自閉症\data\ASD\20250106\20250106-蔡oo\GroundTruth.srt"
# 輸出後的檔案名稱
output_file = r"C:\自閉症\data\ASD\20250106\20250106-蔡oo\fixed_output.srt"


# ==========================================

def standardize_timestamp(time_str):
    """
    將各種不完整的時間格式轉為標準 SRT 格式 (HH:MM:SS,mmm)
    輸入範例: "0:00:12" 或 "00:00:12"
    輸出範例: "00:00:12,000"
    """
    # 移除前後空白
    time_str = time_str.strip()

    # 檢查是否已經有毫秒 (判斷是否包含 . 或 ,)
    if ',' in time_str or '.' in time_str:
        # 如果已經有毫秒，我們只修正分隔符號為逗號
        return time_str.replace('.', ',')

    # 如果完全沒有毫秒，進行補零處理
    parts = time_str.split(':')

    # 防呆：確保至少有時分秒三段
    if len(parts) == 3:
        h, m, s = parts
        # zfill(2) 會把 '1' 變成 '01'，'10' 保持 '10'
        return f"{h.zfill(2)}:{m.zfill(2)}:{s.zfill(2)},000"

    # 如果格式太奇怪 (例如只有 分:秒)，就原樣回傳以免誤改
    return time_str


def fix_srt_timestamps(input_path, output_path):
    print(f"正在讀取並修復: {input_path}")

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        print("編碼偵測失敗，嘗試使用 cp950...")
        with open(input_path, 'r', encoding='cp950') as f:
            content = f.read()

    # 正規表達式策略：
    # 尋找符合 "數字:數字:數字 --> 數字:數字:數字" 的結構
    # 這樣可以避免誤改到字幕內文的時間

    pattern = re.compile(r'(\d{1,2}:\d{1,2}:\d{1,2})\s*-->\s*(\d{1,2}:\d{1,2}:\d{1,2})')

    def replacement_handler(match):
        start_time = match.group(1)
        end_time = match.group(2)

        # 呼叫標準化函數
        new_start = standardize_timestamp(start_time)
        new_end = standardize_timestamp(end_time)

        return f"{new_start} --> {new_end}"

    # 執行取代
    new_content = pattern.sub(replacement_handler, content)

    # 寫入檔案
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"✅ 修復完成！")
    print(f"   原格式範例: 00:00:05 --> 00:00:10")
    print(f"   新格式範例: 00:00:05,000 --> 00:00:10,000")
    print(f"👉 請使用新檔案：{output_path}")


if __name__ == "__main__":
    if os.path.exists(input_file):
        fix_srt_timestamps(input_file, output_file)
    else:
        print(f"❌ 找不到檔案：{input_file}")