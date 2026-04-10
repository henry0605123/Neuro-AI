import os
import re
import numpy as np
import torch
import torchaudio
from pyannote.audio import Inference, Model
from pyannote.core import Segment
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# =================設定區=================
# 請設定環境變數 HF_TOKEN（勿寫入版控）
HF_TOKEN = os.environ.get("HF_TOKEN")

# 檔案路徑 (請確認這裡指到正確的 .wav 與 .srt)
AUDIO_PATH = r"C:\自閉症\data\ASD\20250106\20250106-蔡oo\20250106-蔡oo.mp3"
GT_SRT_PATH = r"C:\自閉症\data\TD\20250324-20054665陳芮晞\GTruth.srt"

# 設定中文字型 (以免圖表亂碼)
# 嘗試抓 Windows 常見字型
font_paths = [
    r"C:\Windows\Fonts\msjh.ttc",  # 微軟正黑體
    r"C:\Windows\Fonts\simsun.ttc", # 新宋體
    r"C:\Windows\Fonts\kaiu.ttf"    # 標楷體
]
my_font = None
for path in font_paths:
    if os.path.exists(path):
        my_font = fm.FontProperties(fname=path)
        print(f"字型載入成功: {path}")
        break
if my_font is None:
    print("警告：找不到中文字型，圖表中文可能會顯示為方塊。")
# =======================================

class EmbeddingVisualizer:
    def __init__(self, audio_path, token):
        self.audio_path = audio_path
        print("正在載入 Pyannote Wespeaker 模型 (第一次執行會下載)...")
        try:
            # 改用更新的 wespeaker 模型，效果較好
            self.model = Inference("pyannote/wespeaker-voxceleb-resnet34-LM", use_auth_token=token)
        except Exception as e:
            print("無法載入 'pyannote/wespeaker-voxceleb-resnet34-LM'。嘗試載入舊版 'pyannote/embedding'...")
            try:
                self.model = Inference("pyannote/embedding", use_auth_token=token)
            except Exception as e2:
                print(f"\n嚴重錯誤: 模型載入失敗。請確認：\n1. Token 正確\n2. 已在 HuggingFace 網頁上同意使用條款 (Gated Model)\n錯誤訊息: {e2}")
                exit()

    def parse_srt(self, srt_path):
        """針對你的 SRT 格式優化的解析器"""
        segments = []
        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(srt_path, 'r', encoding='utf-8-sig') as f:
                content = f.read()

        # 正規表達式抓取區塊
        pattern = re.compile(
            r'(\d{1,2}:\d{2}:\d{2}[,.]\d{3}) --> (\d{1,2}:\d{2}:\d{2}[,.]\d{3})\n(.*?)(?=\n\d+\n|\Z)',
            re.DOTALL
        )
        matches = pattern.findall(content)
        print(f"DEBUG: 原始 SRT 區塊數量: {len(matches)}")

        for m in matches:
            start = self._time_to_sec(m[0])
            end = self._time_to_sec(m[1])
            text = m[2].strip()
            duration = end - start

            # 針對你的 SRT 格式： "測試者: (吸氣)"
            # 支援半形冒號(:) 與 全形冒號(：)
            match = re.match(r'^([^\s:：]+)[:：]\s*(.*)', text, re.DOTALL)
            
            if match:
                speaker = match.group(1).strip()
                # 過濾掉太短的片段 (小於 0.3 秒通常提取不出有效聲紋)
                # 你的第一個片段是 0.49 秒，這裡放寬到 0.3 秒以免漏掉
                if duration >= 0.3:
                    segments.append((start, end, speaker))
            else:
                # 可能是沒有標註語者的雜訊，暫時忽略
                pass
        
        print(f"DEBUG: 解析後有效語音片段: {len(segments)}")
        return sorted(segments, key=lambda x: x[0])

    def _time_to_sec(self, time_str):
        time_str = time_str.replace(',', '.')
        p = time_str.split(':')
        return float(p[0])*3600 + float(p[1])*60 + float(p[2])

    def extract_noise_segments(self, speech_segments, num_samples=30):
        """提取空白處作為噪音樣本"""
        noise_segs = []
        last_end = 0.0
        
        for start, end, _ in speech_segments:
            gap = start - last_end
            if gap > 2.0:
                mid = last_end + (gap / 2)
                # 確保不超出邊界
                noise_segs.append((mid - 0.5, mid + 0.5, "Noise (背景/靜音)"))
            last_end = end
            
        # 隨機取樣
        if len(noise_segs) > num_samples:
            indices = np.random.choice(len(noise_segs), num_samples, replace=False)
            noise_segs = [noise_segs[i] for i in indices]
            
        print(f"DEBUG: 提取噪音樣本數量: {len(noise_segs)}")
        return noise_segs

    def get_embeddings(self, segments):
        """輸入時間段，輸出聲紋向量與標籤 (修正版)"""
        vectors = []
        labels = []
        
        print(f"正在提取聲紋 (共 {len(segments)} 個片段)...")
        
        error_count = 0
        
        for i, (start, end, label) in enumerate(segments):
            try:
                # 1. 取得 SlidingWindowFeature 物件
                result = self.model.crop(self.audio_path, Segment(start, end))
                
                # 2. === 關鍵修正 ===
                # 我們必須從 result.data 取得真正的數值矩陣
                embedding_data = result.data 
                
                # 3. 處理維度
                # 如果這段時間很長，模型切了好幾個 window，形狀會是 (N, 512) -> 取平均變 (512,)
                # 如果只有一個 window，形狀可能是 (1, 512) -> 取平均變 (512,)
                if len(embedding_data.shape) == 2:
                    vec = np.mean(embedding_data, axis=0)
                else:
                    vec = embedding_data
                    
                vectors.append(vec) 
                labels.append(label)

                # 成功提取第一個時，印出形狀確認
                if len(vectors) == 1:
                    print(f"DEBUG: 成功提取第一個向量，原始形狀: {embedding_data.shape} -> 平均後: {vec.shape}")

            except Exception as e:
                error_count += 1
                if error_count == 1:
                    print(f"\n[錯誤偵測] 第 1 筆提取失敗！")
                    print(f"時間段: {start} - {end}")
                    print(f"錯誤原因: {e}")
                elif error_count == 10:
                    print("錯誤過多，暫停顯示後續錯誤訊息...")
                continue
                
        if error_count == len(segments):
            print(f"\n所有 {len(segments)} 個片段全部失敗。")
        else:
            print(f"成功提取 {len(vectors)} / {len(segments)} 個片段。")
                
        return np.array(vectors), labels

    def plot_tsne(self, vectors, labels):
        n_samples = len(vectors)
        if n_samples < 2:
            print("錯誤: 有效樣本數不足 2 個，無法繪製 t-SNE。請檢查 SRT 解析是否正確。")
            return

        # 動態調整 perplexity，防止樣本太少報錯
        # 根據 sklearn 文件，perplexity 必須小於 n_samples
        perp = min(30, n_samples - 1)
        if perp < 5: perp = max(1, n_samples - 1)

        print(f"正在執行 t-SNE (樣本數: {n_samples}, Perplexity: {perp})...")
        
        # 修正: 移除 n_iter 參數以相容新版 sklearn，加入 init='random'
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='random', learning_rate='auto')
        
        try:
            v_2d = tsne.fit_transform(vectors)
        except Exception as e:
            print(f"t-SNE 計算失敗: {e}")
            return
        
        plt.figure(figsize=(12, 10))
        unique_labels = sorted(list(set(labels)))
        
        # 簡單配色
        colors = plt.cm.get_cmap('tab10', len(unique_labels))
        
        for i, label in enumerate(unique_labels):
            idxs = [idx for idx, l in enumerate(labels) if l == label]
            points = v_2d[idxs]
            
            # 根據不同標籤給不同形狀，增加辨識度
            marker = 'o'
            if "Noise" in label: marker = 'x'
            
            plt.scatter(points[:, 0], points[:, 1], 
                        c=[colors(i)], label=label, s=60, alpha=0.7, edgecolors='w', marker=marker)

        plt.title("語者聲紋分佈 (t-SNE)", fontproperties=my_font, fontsize=16)
        plt.legend(prop=my_font)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    viz = EmbeddingVisualizer(AUDIO_PATH, HF_TOKEN)
    
    # 1. 解析 SRT
    speech_segs = viz.parse_srt(GT_SRT_PATH)
    
    if not speech_segs:
        print("錯誤: 沒有解析到任何語音片段。請檢查 SRT 內容或時間格式。")
    else:
        # 2. 提取噪音
        noise_segs = viz.extract_noise_segments(speech_segs, num_samples=30)
        
        # 3. 合併
        all_segs = speech_segs + noise_segs
        
        # 4. 提取聲紋
        vectors, labels = viz.get_embeddings(all_segs)
        
        # 5. 畫圖
        if len(vectors) > 0:
            viz.plot_tsne(vectors, labels)
        else:
            print("沒有成功提取任何向量。")