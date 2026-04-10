"""
彈性 NLP 評估系統
處理人工標註與自動轉錄分段不一致的問題
提供三種評估模式
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import json
import re

try:
    from sentence_transformers import SentenceTransformer, util
    from bert_score import score as bert_score
    import jieba
except ImportError as e:
    print(f"警告: 某些套件未安裝 - {e}")
    print("請執行: pip install sentence-transformers bert-score jieba")

@dataclass
class FlexibleNLPResult:
    """彈性 NLP 評估結果"""
    # 整體評估
    overall_similarity: float
    overall_bert_f1: float
    overall_bleu: float
    overall_rouge_l: float

    # 文本統計
    gt_total_length: int
    pred_total_length: int
    gt_segment_count: int
    pred_segment_count: int

    # 詳細結果
    mode: str  # 評估模式
    detailed_info: Dict


class FlexibleNLPEvaluator:
    """彈性 NLP 評估器 - 處理分段不一致"""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """初始化評估器"""
        print(f"載入 Sentence Transformer 模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("模型載入完成!")

    def normalize_text(self, text: str) -> str:
        """正規化文本"""
        # 移除語者標籤
        text = re.sub(r'^[A-Z_0-9]+:\s*', '', text)
        text = re.sub(r'^(測試者|小孩|家長):\s*', '', text)

        # 移除標點符號
        punctuation = r'[，。！？、；：""''《》（）【】『』「」…—～·\,\.\!\?\;\:\"\'\(\)\[\]\{\}]'
        text = re.sub(punctuation, '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def merge_texts(self, texts: List[str]) -> str:
        """合併多個文本片段為一個完整文本"""
        normalized = [self.normalize_text(t) for t in texts]
        return ' '.join(normalized)

    def evaluate_mode_1_whole_text(self,
                                   ground_truth: List[str],
                                   predictions: List[str]) -> FlexibleNLPResult:
        """
        模式 1: 整體文本比較
        將所有片段合併成一個完整文本進行比較

        優點: 不受分段影響,評估整體品質
        適用: 總體評估,不在意分段準確性
        """

        print("\n【模式 1: 整體文本比較】")
        print("將所有片段合併為完整文本...")

        # 合併所有文本
        gt_full = self.merge_texts(ground_truth)
        pred_full = self.merge_texts(predictions)

        print(f"Ground Truth 總長度: {len(gt_full)} 字")
        print(f"Prediction 總長度: {len(pred_full)} 字")

        # 計算語義相似度
        print("計算語義相似度...")
        emb1 = self.model.encode(gt_full, convert_to_tensor=True)
        emb2 = self.model.encode(pred_full, convert_to_tensor=True)
        similarity = util.cos_sim(emb1, emb2).item()

        # 計算 BERTScore
        print("計算 BERTScore...")
        P, R, F1 = bert_score(
            [pred_full],
            [gt_full],
            lang="zh",
            verbose=False
        )
        bert_f1 = F1.mean().item()

        # 計算 BLEU
        print("計算 BLEU...")
        bleu = self._calculate_bleu(gt_full, pred_full)

        # 計算 ROUGE-L
        rouge = self._calculate_rouge(gt_full, pred_full)

        print("評估完成!")

        return FlexibleNLPResult(
            overall_similarity=similarity,
            overall_bert_f1=bert_f1,
            overall_bleu=bleu,
            overall_rouge_l=rouge['rouge_l'],
            gt_total_length=len(gt_full),
            pred_total_length=len(pred_full),
            gt_segment_count=len(ground_truth),
            pred_segment_count=len(predictions),
            mode="整體文本比較",
            detailed_info={
                'ground_truth_full': gt_full[:200] + "...",
                'prediction_full': pred_full[:200] + "...",
                'bert_precision': P.mean().item(),
                'bert_recall': R.mean().item(),
                'rouge_1': rouge['rouge_1'],
                'rouge_2': rouge['rouge_2']
            }
        )

    def evaluate_mode_2_time_aligned(self,
                                    ground_truth: List[Tuple[str, float, float]],
                                    predictions: List[Tuple[str, float, float]],
                                    tolerance: float = 2.0) -> FlexibleNLPResult:
        """
        模式 2: 時間對齊比較
        根據時間軸對齊片段進行比較

        參數:
            ground_truth: [(text, start_time, end_time), ...]
            predictions: [(text, start_time, end_time), ...]
            tolerance: 時間容差(秒)

        優點: 考慮時間對齊,更精確
        適用: SRT 檔案比較
        """

        print("\n【模式 2: 時間對齊比較】")
        print(f"時間容差: ±{tolerance} 秒")

        # 建立時間對齊的配對
        aligned_pairs = []
        used_pred_indices = set()

        for gt_text, gt_start, gt_end in ground_truth:
            gt_mid = (gt_start + gt_end) / 2

            # 找出時間最接近的預測片段
            best_match = None
            best_distance = float('inf')
            best_idx = -1

            for i, (pred_text, pred_start, pred_end) in enumerate(predictions):
                if i in used_pred_indices:
                    continue

                pred_mid = (pred_start + pred_end) / 2
                distance = abs(gt_mid - pred_mid)

                if distance <= tolerance and distance < best_distance:
                    best_distance = distance
                    best_match = pred_text
                    best_idx = i

            if best_match:
                aligned_pairs.append((gt_text, best_match))
                used_pred_indices.add(best_idx)
            else:
                # 沒有匹配的,用空字串
                aligned_pairs.append((gt_text, ""))

        print(f"成功對齊: {len(aligned_pairs)} 個片段")
        print(f"無法對齊的 GT: {len(ground_truth) - len(aligned_pairs)}")
        print(f"未使用的 Pred: {len(predictions) - len(used_pred_indices)}")

        # 計算評估指標
        gt_texts = [self.normalize_text(pair[0]) for pair in aligned_pairs]
        pred_texts = [self.normalize_text(pair[1]) for pair in aligned_pairs]

        # 合併後計算
        gt_full = ' '.join(gt_texts)
        pred_full = ' '.join(pred_texts)

        # 計算指標
        emb1 = self.model.encode(gt_full, convert_to_tensor=True)
        emb2 = self.model.encode(pred_full, convert_to_tensor=True)
        similarity = util.cos_sim(emb1, emb2).item()

        P, R, F1 = bert_score([pred_full], [gt_full], lang="zh", verbose=False)
        bert_f1 = F1.mean().item()

        bleu = self._calculate_bleu(gt_full, pred_full)
        rouge = self._calculate_rouge(gt_full, pred_full)

        return FlexibleNLPResult(
            overall_similarity=similarity,
            overall_bert_f1=bert_f1,
            overall_bleu=bleu,
            overall_rouge_l=rouge['rouge_l'],
            gt_total_length=len(gt_full),
            pred_total_length=len(pred_full),
            gt_segment_count=len(ground_truth),
            pred_segment_count=len(predictions),
            mode="時間對齊比較",
            detailed_info={
                'aligned_count': len(aligned_pairs),
                'unaligned_gt': len(ground_truth) - len(aligned_pairs),
                'unused_pred': len(predictions) - len(used_pred_indices),
                'coverage': len(aligned_pairs) / len(ground_truth) if ground_truth else 0
            }
        )

    def evaluate_mode_3_sliding_window(self,
                                      ground_truth: List[str],
                                      predictions: List[str],
                                      window_size: int = 5) -> FlexibleNLPResult:
        """
        模式 3: 滑動窗口比較
        將較多的片段分組後與較少的片段比較

        參數:
            window_size: 窗口大小(自動調整)

        優點: 適應不同分段策略
        適用: 分段策略完全不同的情況
        """

        print("\n【模式 3: 滑動窗口比較】")

        # 確定哪個是較多片段的
        if len(ground_truth) > len(predictions):
            many_segments = ground_truth
            few_segments = predictions
            is_gt_many = True
        else:
            many_segments = predictions
            few_segments = ground_truth
            is_gt_many = False

        # 計算合適的窗口大小
        ratio = len(many_segments) / len(few_segments)
        window = max(1, int(ratio))

        print(f"GT 片段數: {len(ground_truth)}")
        print(f"Pred 片段數: {len(predictions)}")
        print(f"窗口大小: {window}")

        # 分組
        grouped_segments = []
        for i in range(0, len(many_segments), window):
            group = many_segments[i:i+window]
            merged = self.merge_texts(group)
            grouped_segments.append(merged)

        # 確保數量一致
        min_len = min(len(grouped_segments), len(few_segments))
        grouped_segments = grouped_segments[:min_len]
        few_segments = few_segments[:min_len]

        print(f"分組後數量: {min_len}")

        # 正規化
        if is_gt_many:
            gt_texts = grouped_segments
            pred_texts = [self.normalize_text(t) for t in few_segments]
        else:
            gt_texts = [self.normalize_text(t) for t in few_segments]
            pred_texts = grouped_segments

        # 合併後評估
        gt_full = ' '.join(gt_texts)
        pred_full = ' '.join(pred_texts)

        # 計算指標
        emb1 = self.model.encode(gt_full, convert_to_tensor=True)
        emb2 = self.model.encode(pred_full, convert_to_tensor=True)
        similarity = util.cos_sim(emb1, emb2).item()

        P, R, F1 = bert_score([pred_full], [gt_full], lang="zh", verbose=False)
        bert_f1 = F1.mean().item()

        bleu = self._calculate_bleu(gt_full, pred_full)
        rouge = self._calculate_rouge(gt_full, pred_full)

        return FlexibleNLPResult(
            overall_similarity=similarity,
            overall_bert_f1=bert_f1,
            overall_bleu=bleu,
            overall_rouge_l=rouge['rouge_l'],
            gt_total_length=len(gt_full),
            pred_total_length=len(pred_full),
            gt_segment_count=len(ground_truth),
            pred_segment_count=len(predictions),
            mode="滑動窗口比較",
            detailed_info={
                'window_size': window,
                'grouped_count': min_len,
                'original_gt_count': len(ground_truth),
                'original_pred_count': len(predictions)
            }
        )

    def print_report(self, result: FlexibleNLPResult):
        """列印評估報告"""
        print("\n" + "="*60)
        print(f"NLP 評估報告 - {result.mode}")
        print("="*60)

        print("\n【文本統計】")
        print(f"  Ground Truth 片段數:     {result.gt_segment_count}")
        print(f"  Prediction 片段數:       {result.pred_segment_count}")
        print(f"  Ground Truth 總字數:     {result.gt_total_length}")
        print(f"  Prediction 總字數:       {result.pred_total_length}")
        print(f"  字數差異:               {abs(result.gt_total_length - result.pred_total_length)} 字")

        print("\n【NLP 評估指標】")
        print(f"  語義相似度:             {result.overall_similarity*100:.2f}%")
        print(f"  BERTScore F1:          {result.overall_bert_f1*100:.2f}%")
        print(f"  BLEU Score:            {result.overall_bleu*100:.2f}%")
        print(f"  ROUGE-L:               {result.overall_rouge_l*100:.2f}%")

        # 語義保留率
        semantic_preservation = (result.overall_similarity + result.overall_bert_f1) / 2
        print(f"  語義保留率 (綜合):      {semantic_preservation*100:.2f}%")

        # 整體評價
        print("\n【整體評價】")
        if semantic_preservation >= 0.9:
            grade = "優秀 (A)"
            comment = "語義保留度極高,轉錄品質優良"
        elif semantic_preservation >= 0.8:
            grade = "良好 (B)"
            comment = "語義大致正確,適合實際應用"
        elif semantic_preservation >= 0.7:
            grade = "尚可 (C)"
            comment = "語義部分保留,建議人工校對"
        elif semantic_preservation >= 0.6:
            grade = "一般 (D)"
            comment = "語義遺失較多,需要改進"
        else:
            grade = "較差 (F)"
            comment = "語義保留不足,建議重新轉錄"

        print(f"  等級: {grade}")
        print(f"  評語: {comment}")

        # 額外資訊
        if result.detailed_info:
            print("\n【詳細資訊】")
            for key, value in result.detailed_info.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

        print("="*60)

    def export_report(self, result: FlexibleNLPResult, output_file: str):
        """匯出評估報告"""
        report = {
            'mode': result.mode,
            'metrics': {
                'semantic_similarity': f"{result.overall_similarity*100:.2f}%",
                'bert_f1': f"{result.overall_bert_f1*100:.2f}%",
                'bleu_score': f"{result.overall_bleu*100:.2f}%",
                'rouge_l': f"{result.overall_rouge_l*100:.2f}%",
                'semantic_preservation': f"{(result.overall_similarity + result.overall_bert_f1)/2*100:.2f}%"
            },
            'statistics': {
                'gt_segment_count': result.gt_segment_count,
                'pred_segment_count': result.pred_segment_count,
                'gt_total_length': result.gt_total_length,
                'pred_total_length': result.pred_total_length,
                'length_difference': abs(result.gt_total_length - result.pred_total_length)
            },
            'detailed_info': result.detailed_info
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n評估報告已儲存至: {output_file}")

    # 輔助函數
    def _calculate_bleu(self, reference: str, hypothesis: str) -> float:
        """計算 BLEU Score"""
        ref_tokens = list(jieba.cut(reference))
        hyp_tokens = list(jieba.cut(hypothesis))

        if len(hyp_tokens) == 0:
            return 0.0

        precisions = []
        for n in range(1, 5):
            ref_ngrams = self._get_ngrams(ref_tokens, n)
            hyp_ngrams = self._get_ngrams(hyp_tokens, n)

            if len(hyp_ngrams) == 0:
                precisions.append(0)
                continue

            matches = sum(min(ref_ngrams.get(ng, 0), hyp_ngrams.get(ng, 0))
                         for ng in hyp_ngrams)
            precision = matches / len(hyp_ngrams)
            precisions.append(precision)

        if any(p == 0 for p in precisions):
            return 0.0

        bleu = np.exp(np.mean([np.log(p) for p in precisions]))
        bp = min(1.0, np.exp(1 - len(ref_tokens) / len(hyp_tokens)))

        return bleu * bp

    def _calculate_rouge(self, reference: str, prediction: str) -> Dict[str, float]:
        """計算 ROUGE Scores"""
        ref_tokens = list(jieba.cut(reference))
        pred_tokens = list(jieba.cut(prediction))

        ref_unigrams = set(ref_tokens)
        pred_unigrams = set(pred_tokens)
        overlap = len(ref_unigrams & pred_unigrams)
        rouge_1 = overlap / len(ref_unigrams) if len(ref_unigrams) > 0 else 0

        ref_bigrams = set(self._get_ngrams(ref_tokens, 2).keys())
        pred_bigrams = set(self._get_ngrams(pred_tokens, 2).keys())
        overlap = len(ref_bigrams & pred_bigrams)
        rouge_2 = overlap / len(ref_bigrams) if len(ref_bigrams) > 0 else 0

        lcs_length = self._lcs_length(ref_tokens, pred_tokens)
        rouge_l = lcs_length / len(ref_tokens) if len(ref_tokens) > 0 else 0

        return {
            'rouge_1': rouge_1,
            'rouge_2': rouge_2,
            'rouge_l': rouge_l
        }

    def _get_ngrams(self, tokens: List[str], n: int) -> Dict[tuple, int]:
        """取得 n-grams"""
        ngrams = {}
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams

    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """計算最長公共子序列長度"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]


def load_srt_file(srt_path: str) -> List[str]:
    """讀取 SRT 檔案(只提取文本)"""
    texts = []

    with open(srt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_text = []
    for line in lines:
        line = line.strip()

        if line.isdigit() or '-->' in line or not line:
            if current_text:
                text = ' '.join(current_text)
                texts.append(text)
                current_text = []
        else:
            current_text.append(line)

    if current_text:
        text = ' '.join(current_text)
        texts.append(text)

    return texts


def load_srt_with_time(srt_path: str) -> List[Tuple[str, float, float]]:
    """讀取 SRT 檔案(包含時間資訊)"""
    segments = []

    with open(srt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # 跳過序號
        if line.isdigit():
            i += 1
            if i >= len(lines):
                break

            # 解析時間
            time_line = lines[i].strip()
            if '-->' in time_line:
                times = time_line.split('-->')
                start = parse_srt_time(times[0].strip())
                end = parse_srt_time(times[1].strip())

                i += 1
                # 收集文本
                text_lines = []
                while i < len(lines) and lines[i].strip() and not lines[i].strip().isdigit():
                    text_lines.append(lines[i].strip())
                    i += 1

                if text_lines:
                    text = ' '.join(text_lines)
                    segments.append((text, start, end))

        i += 1

    return segments


def parse_srt_time(time_str: str) -> float:
    """解析 SRT 時間格式為秒數"""
    # 格式: 00:00:00,000 或 00:00:00
    time_str = time_str.replace(',', '.')
    parts = time_str.split(':')

    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds

    return 0.0


def main():
    """主程式"""

    print("="*60)
    print("彈性 NLP 評估系統")
    print("="*60)

    # 載入資料
    print("\n載入檔案...")

    gt_srt_path = r"C:\自閉症\data\ASD\20250106\20250106-蔡oo\GroundTruth.srt"
    pred_srt_path = r"C:\Users\User\PycharmProjects\PythonProject1\Fine_tune\outputs\test.srt"

    # 載入文本
    ground_truth_texts = load_srt_file(gt_srt_path)
    prediction_texts = load_srt_file(pred_srt_path)

    print(f"Ground Truth: {len(ground_truth_texts)} 個句子")
    print(f"Predictions: {len(prediction_texts)} 個句子")

    # 建立評估器
    evaluator = FlexibleNLPEvaluator()

    # ===== 模式 1: 整體文本比較 (推薦) =====
    print("\n" + "="*60)
    result1 = evaluator.evaluate_mode_1_whole_text(
        ground_truth_texts,
        prediction_texts
    )
    evaluator.print_report(result1)

    # 匯出報告
    output1 = r"C:\自閉症\data\ASD\20250106\20250106-蔡oo\nlp_report_mode1.json"
    evaluator.export_report(result1, output1)

    # ===== 模式 2: 時間對齊比較 (如果有完整時間資訊) =====
    try:
        print("\n" + "="*60)
        print("嘗試載入時間資訊...")
        gt_with_time = load_srt_with_time(gt_srt_path)
        pred_with_time = load_srt_with_time(pred_srt_path)

        if gt_with_time and pred_with_time:
            result2 = evaluator.evaluate_mode_2_time_aligned(
                gt_with_time,
                pred_with_time,
                tolerance=3.0  # 3秒容差
            )
            evaluator.print_report(result2)

            output2 = r"C:\自閉症\data\ASD\20250106\20250106-蔡oo\nlp_report_mode2.json"
            evaluator.export_report(result2, output2)
    except Exception as e:
        print(f"模式 2 評估失敗: {e}")

    # ===== 模式 3: 滑動窗口比較 =====
    print("\n" + "="*60)
    result3 = evaluator.evaluate_mode_3_sliding_window(
        ground_truth_texts,
        prediction_texts
    )
    evaluator.print_report(result3)

    output3 = r"C:\自閉症\data\ASD\20250106\20250106-蔡oo\nlp_report_mode3.json"
    evaluator.export_report(result3, output3)

    # ===== 總結 =====
    print("\n" + "="*60)
    print("評估完成總結")
    print("="*60)
    print("\n【三種模式比較】")
    print(f"模式 1 (整體文本):  語義相似度 {result1.overall_similarity*100:.2f}%")
    print(f"模式 3 (滑動窗口):  語義相似度 {result3.overall_similarity*100:.2f}%")

    print("\n【建議】")
    print("✓ 模式 1 最適合評估整體轉錄品質")
    print("✓ 模式 3 可作為參考")
    print("✓ 建議以模式 1 的結果為主要評估依據")


if __name__ == "__main__":
    main()