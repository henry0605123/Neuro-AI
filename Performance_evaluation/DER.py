import re
import numpy as np
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class Utterance:
    speaker: str
    text: str
    start: float
    end: float


class DiarizationEvaluator:
    def __init__(self):
        # 設定關注的目標語者 (GT 中的名字)
        self.target_filter = ["小孩", "測試者"]
        # 時間解析度 (秒)，越小越準確，建議 0.1
        self.time_step = 0.1

    def parse_srt_with_speaker(self, srt_path: str, is_ground_truth: bool = False) -> List[Utterance]:
        """讀取 SRT 並提取 '語者' 與 '時間'"""
        utterances = []
        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(srt_path, 'r', encoding='utf-8-sig') as f:
                content = f.read()

        # 正規表達式提取 SRT 區塊
        pattern = re.compile(
            r'(\d+)\n(\d{1,2}:\d{2}:\d{2}(?:,\d{3})?) --> (\d{1,2}:\d{2}:\d{2}(?:,\d{3})?)\n(.*?)(?=\n\d+\n|\Z)',
            re.DOTALL
        )

        matches = pattern.findall(content)

        for match in matches:
            start = self._time_to_sec(match[1])
            end = self._time_to_sec(match[2])
            raw_content = match[3].strip()

            lines = raw_content.split('\n')
            for line in lines:
                line = line.strip()
                if not line: continue

                # 嘗試解析 "Speaker: Text" 格式
                speaker_match = re.match(r'^([^\s:：]+)[:：]\s*(.*)', line)

                if speaker_match:
                    speaker = speaker_match.group(1)
                    text = speaker_match.group(2)
                else:
                    speaker = "Unknown"
                    text = line

                speaker = speaker.strip()
                if text.strip():
                    utterances.append(Utterance(speaker, text, start, end))

        return utterances

    def _time_to_sec(self, time_str: str) -> float:
        time_str = time_str.replace(',', '.')
        try:
            parts = time_str.split(':')
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        except:
            return 0.0

    def map_speakers(self, gt_data: List[Utterance], pred_data: List[Utterance]) -> Dict[str, str]:
        """
        建立 Pred 語者到 GT 語者的映射 (使用最大重疊時間貪婪算法)
        """
        overlap_matrix = {}

        # 計算重疊時間
        for gt in gt_data:
            if gt.speaker not in self.target_filter: continue
            for pred in pred_data:
                overlap_start = max(gt.start, pred.start)
                overlap_end = min(gt.end, pred.end)
                duration = max(0, overlap_end - overlap_start)

                if duration > 0:
                    key = (gt.speaker, pred.speaker)
                    overlap_matrix[key] = overlap_matrix.get(key, 0) + duration

        mapping = {}
        used_gt = set()

        # 根據重疊時間長度排序 (由大到小)
        sorted_overlaps = sorted(overlap_matrix.items(), key=lambda x: x[1], reverse=True)

        print("\n【語者映射與對齊】")
        for (gt_spk, pred_spk), duration in sorted_overlaps:
            if pred_spk not in mapping and gt_spk not in used_gt:
                print(f"  預測 '{pred_spk}' -> 對齊 GT '{gt_spk}' (重疊 {duration:.2f}秒)")
                mapping[pred_spk] = gt_spk
                used_gt.add(gt_spk)

        # 剩下的未匹配語者標記為 False Alarm Source
        pred_speakers = set(u.speaker for u in pred_data)
        for p in pred_speakers:
            if p not in mapping:
                mapping[p] = None  # 表示無法映射到目標語者

        return mapping

    def evaluate(self, gt_path: str, pred_path: str):
        print("=" * 60)
        print("標準語者分離評估 (Standard DER Calculation)")
        print("=" * 60)

        # 1. 讀取資料
        gt_data = self.parse_srt_with_speaker(gt_path, is_ground_truth=True)
        pred_data = self.parse_srt_with_speaker(pred_path, is_ground_truth=False)

        # 過濾 GT：只保留目標語者
        gt_data = [u for u in gt_data if u.speaker in self.target_filter]

        if not gt_data:
            print("錯誤: GT 資料中找不到目標語者。")
            return

        # 2. 建立映射
        mapping = self.map_speakers(gt_data, pred_data)

        # 3. 準備時間軸 (Timeline)
        max_time = max(
            max((u.end for u in gt_data), default=0),
            max((u.end for u in pred_data), default=0)
        )

        num_steps = int(np.ceil(max_time / self.time_step))

        # GT Timeline: 0=無人, 1,2...=對應 target_filter 的 index+1
        gt_timeline = np.zeros(num_steps, dtype=int)
        # Pred Timeline: 0=無人, 1,2...=映射後的 ID, -1=未映射噪音
        pred_timeline = np.zeros(num_steps, dtype=int)

        # 建立 ID 查找表
        speaker_to_id = {name: i + 1 for i, name in enumerate(self.target_filter)}

        # 填入 GT 時間軸
        for u in gt_data:
            start_idx = int(u.start / self.time_step)
            end_idx = int(u.end / self.time_step)
            spk_id = speaker_to_id.get(u.speaker)
            if spk_id:
                gt_timeline[start_idx:end_idx] = spk_id

        # 填入 Pred 時間軸 (使用映射後的 ID)
        for u in pred_data:
            mapped_name = mapping.get(u.speaker)
            start_idx = int(u.start / self.time_step)
            end_idx = int(u.end / self.time_step)

            if mapped_name:  # 如果有映射到目標
                spk_id = speaker_to_id.get(mapped_name)
                if spk_id:
                    pred_timeline[start_idx:end_idx] = spk_id
            else:
                # 只有當該位置原本是空的時候才填入，避免覆蓋已映射的
                mask = pred_timeline[start_idx:end_idx] == 0
                pred_timeline[start_idx:end_idx][mask] = -1

        # 4. 計算 DER 各項指標
        miss_count = np.sum((gt_timeline > 0) & (pred_timeline == 0))
        fa_count = np.sum((gt_timeline == 0) & (pred_timeline != 0))
        conf_count = np.sum((gt_timeline > 0) & (pred_timeline != 0) & (gt_timeline != pred_timeline))
        correct_count = np.sum((gt_timeline > 0) & (gt_timeline == pred_timeline))

        # 轉換回秒數
        miss_dur = miss_count * self.time_step
        fa_dur = fa_count * self.time_step
        conf_dur = conf_count * self.time_step
        correct_dur = correct_count * self.time_step
        total_gt_dur = np.sum(gt_timeline > 0) * self.time_step

        if total_gt_dur == 0:
            print("錯誤: 總 GT 時間為 0")
            return

        der = (miss_dur + fa_dur + conf_dur) / total_gt_dur

        # 顯示結果
        print("\n【評估結果詳情】")
        print(f"總 GT 語音長度: {total_gt_dur:.2f} 秒")
        print("-" * 40)
        print(f"正確識別 (Correct): {correct_dur:.2f} 秒 ({correct_dur / total_gt_dur * 100:.2f}%)")
        print(f"漏檢 (Miss):       {miss_dur:.2f} 秒 ({miss_dur / total_gt_dur * 100:.2f}%)")
        print(f"誤檢 (False Alarm):{fa_dur:.2f} 秒 ({fa_dur / total_gt_dur * 100:.2f}%)")
        print(f"混淆 (Confusion):  {conf_dur:.2f} 秒 ({conf_dur / total_gt_dur * 100:.2f}%)")
        print("-" * 40)
        print(f"Diarization Error Rate (DER): {der * 100:.2f}%")

        # 5. 顯示混淆矩陣
        self._print_confusion_matrix(gt_timeline, pred_timeline, speaker_to_id)

        # 6. 計算並顯示 Precision, Recall, F1-Score
        self._print_detailed_metrics(gt_timeline, pred_timeline, speaker_to_id)

    def _print_confusion_matrix(self, gt_timeline, pred_timeline, speaker_to_id):
        """
        計算並印出混淆矩陣
        Rows (行): Ground Truth
        Cols (列): Predicted
        """
        print("\n" + "=" * 60)
        print("混淆矩陣 (Confusion Matrix) - 單位: 秒")
        print("說明: 行(Row)為 GT 真實語者，列(Col)為模型預測語者")
        print("-" * 60)

        num_speakers = len(self.target_filter)

        matrix = np.zeros((num_speakers, num_speakers))

        for t in range(len(gt_timeline)):
            gt_id = gt_timeline[t]
            pred_id = pred_timeline[t]

            if gt_id > 0 and pred_id > 0:
                if pred_id <= num_speakers:
                    matrix[gt_id - 1][pred_id - 1] += self.time_step

        # 修正 f-string 反斜線問題，先定義變數
        row_header = "GT \\ Pred"
        header = f"{row_header:>12} |"

        for name in self.target_filter:
            header += f" {name:>10} |"
        print(header)
        print("-" * len(header))

        for i, gt_name in enumerate(self.target_filter):
            row_str = f"{gt_name:>12} |"
            for j in range(num_speakers):
                val = matrix[i][j]
                if i == j:
                    row_str += f" *{val:>8.2f} |"
                else:
                    row_str += f" {val:>10.2f} |"
            print(row_str)
        print("=" * 60)

    def _print_detailed_metrics(self, gt_timeline, pred_timeline, speaker_to_id):
        """
        計算每個語者的 Precision, Recall, F1-Score
        """
        print("\n" + "=" * 60)
        print("詳細語者指標 (Detailed Metrics)")
        print("Precision = TP / (TP + FP)")
        print("Recall    = TP / (TP + FN)")
        print("F1-Score  = 2 * P * R / (P + R)")
        print("-" * 60)

        header = f"{'Speaker':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}"
        print(header)
        print("-" * len(header))

        for name in self.target_filter:
            spk_id = speaker_to_id[name]

            # True Positive (TP): GT是這個人，預測也是這個人
            tp = np.sum((gt_timeline == spk_id) & (pred_timeline == spk_id)) * self.time_step

            # Ground Truth Total (TP + FN): GT是這個人的總時間 (包含被漏檢或混淆成別人的)
            gt_total = np.sum(gt_timeline == spk_id) * self.time_step

            # Predicted Total (TP + FP): 預測是這個人的總時間 (包含誤檢噪音或混淆別人的)
            pred_total = np.sum(pred_timeline == spk_id) * self.time_step

            # 計算指標
            precision = (tp / pred_total) if pred_total > 0 else 0.0
            recall = (tp / gt_total) if gt_total > 0 else 0.0

            if (precision + recall) > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0

            print(f"{name:<10} | {precision * 100:>9.2f}% | {recall * 100:>9.2f}% | {f1 * 100:>9.2f}%")

        print("=" * 60)


if __name__ == "__main__":
    # 請修改這裡的路徑
    gt_path = r"C:\自閉症\data\ASD\20250106\20250106-蔡oo\GroundTruth.srt"
    pred_path = r"C:\自閉症\data\ASD\20250106\20250106-蔡oo\Aegisub.srt"

    evaluator = DiarizationEvaluator()
    evaluator.evaluate(gt_path, pred_path)