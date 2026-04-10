"""
評估工具
========
評估語者分離的準確度，比較模型效果

功能:
1. 計算 DER (Diarization Error Rate)
2. 比較不同模型的結果
3. 視覺化分析
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.srt_parser import SRTParser, SRTSegment


@dataclass
class EvaluationResult:
    """評估結果"""
    der: float  # Diarization Error Rate
    miss: float  # Miss Rate
    false_alarm: float  # False Alarm Rate
    confusion: float  # Speaker Confusion Rate
    total_duration: float  # 總時長
    correct_duration: float  # 正確識別的時長


class DiarizationEvaluator:
    """語者分離評估器"""
    
    @staticmethod
    def time_overlap(seg1: Tuple[float, float], seg2: Tuple[float, float]) -> float:
        """
        計算兩個時間段的重疊時長
        
        Args:
            seg1: (開始時間, 結束時間)
            seg2: (開始時間, 結束時間)
            
        Returns:
            重疊時長（秒）
        """
        start = max(seg1[0], seg2[0])
        end = min(seg1[1], seg2[1])
        return max(0, end - start)
    
    @classmethod
    def evaluate_srt(
        cls,
        reference_path: str,
        hypothesis_path: str,
        collar: float = 0.25
    ) -> EvaluationResult:
        """
        評估 SRT 檔案的語者分離準確度
        
        Args:
            reference_path: 參考標註（Ground Truth）
            hypothesis_path: 系統輸出（預測結果）
            collar: 容錯邊界（秒）
            
        Returns:
            評估結果
        """
        # 解析 SRT 檔案
        ref_segments = SRTParser.parse_file(reference_path)
        hyp_segments = SRTParser.parse_file(hypothesis_path)
        
        # 取得說話人時間段
        ref_speaker_segs = cls._get_speaker_timeline(ref_segments)
        hyp_speaker_segs = cls._get_speaker_timeline(hyp_segments)
        
        # 計算總時長
        total_duration = max(
            SRTParser.get_total_duration(ref_segments),
            SRTParser.get_total_duration(hyp_segments)
        )
        
        # 計算各項錯誤
        miss = 0.0  # 漏檢（參考有但預測沒有）
        false_alarm = 0.0  # 誤檢（預測有但參考沒有）
        confusion = 0.0  # 說話人混淆
        
        # 計算每個時間點的說話人
        time_points = set()
        for segs in ref_speaker_segs.values():
            for start, end in segs:
                time_points.add(start)
                time_points.add(end)
        for segs in hyp_speaker_segs.values():
            for start, end in segs:
                time_points.add(start)
                time_points.add(end)
        
        time_points = sorted(time_points)
        
        # 逐段計算
        for i in range(len(time_points) - 1):
            t_start = time_points[i]
            t_end = time_points[i + 1]
            duration = t_end - t_start
            
            if duration <= 0:
                continue
            
            # 取得該時間段的說話人
            ref_speakers = cls._get_speakers_at_time(ref_speaker_segs, t_start, t_end)
            hyp_speakers = cls._get_speakers_at_time(hyp_speaker_segs, t_start, t_end)
            
            # 計算錯誤
            if ref_speakers and not hyp_speakers:
                miss += duration
            elif hyp_speakers and not ref_speakers:
                false_alarm += duration
            elif ref_speakers and hyp_speakers:
                # 簡化處理：比較說話人標籤
                ref_normalized = cls._normalize_speakers(ref_speakers)
                hyp_normalized = cls._normalize_speakers(hyp_speakers)
                
                if ref_normalized != hyp_normalized:
                    confusion += duration
        
        # 計算 DER
        total_ref_duration = sum(
            end - start 
            for segs in ref_speaker_segs.values() 
            for start, end in segs
        )
        
        if total_ref_duration > 0:
            der = (miss + false_alarm + confusion) / total_ref_duration
        else:
            der = 0.0
        
        correct_duration = total_ref_duration - miss - confusion
        
        return EvaluationResult(
            der=der,
            miss=miss / total_ref_duration if total_ref_duration > 0 else 0,
            false_alarm=false_alarm / total_ref_duration if total_ref_duration > 0 else 0,
            confusion=confusion / total_ref_duration if total_ref_duration > 0 else 0,
            total_duration=total_duration,
            correct_duration=max(0, correct_duration)
        )
    
    @staticmethod
    def _get_speaker_timeline(segments: List[SRTSegment]) -> Dict[str, List[Tuple[float, float]]]:
        """取得每個說話人的時間軸"""
        timeline = {}
        
        for seg in segments:
            speaker = seg.speaker
            if speaker not in timeline:
                timeline[speaker] = []
            timeline[speaker].append((seg.start_time, seg.end_time))
        
        return timeline
    
    @staticmethod
    def _get_speakers_at_time(
        speaker_segs: Dict[str, List[Tuple[float, float]]],
        t_start: float,
        t_end: float
    ) -> set:
        """取得特定時間段的說話人"""
        speakers = set()
        
        for speaker, segs in speaker_segs.items():
            for start, end in segs:
                # 檢查是否有重疊
                if start < t_end and end > t_start:
                    speakers.add(speaker)
                    break
        
        return speakers
    
    @staticmethod
    def _normalize_speakers(speakers: set) -> set:
        """標準化說話人標籤（用於比較）"""
        normalized = set()
        
        for s in speakers:
            s_lower = s.lower()
            if '測試' in s or 'speaker_00' in s_lower:
                normalized.add('tester')
            elif '小孩' in s or 'speaker_01' in s_lower:
                normalized.add('child')
            else:
                normalized.add(s)
        
        return normalized
    
    @classmethod
    def compare_models(
        cls,
        reference_path: str,
        original_path: str,
        finetuned_path: str
    ) -> Dict:
        """
        比較原始模型和微調模型的效果
        
        Args:
            reference_path: Ground Truth
            original_path: 原始模型輸出
            finetuned_path: 微調模型輸出
            
        Returns:
            比較結果
        """
        original_result = cls.evaluate_srt(reference_path, original_path)
        finetuned_result = cls.evaluate_srt(reference_path, finetuned_path)
        
        improvement = {
            'der': original_result.der - finetuned_result.der,
            'miss': original_result.miss - finetuned_result.miss,
            'false_alarm': original_result.false_alarm - finetuned_result.false_alarm,
            'confusion': original_result.confusion - finetuned_result.confusion,
        }
        
        return {
            'original': original_result,
            'finetuned': finetuned_result,
            'improvement': improvement
        }
    
    @staticmethod
    def print_result(result: EvaluationResult, title: str = "評估結果"):
        """列印評估結果"""
        print(f"\n{title}")
        print("-" * 40)
        print(f"  DER (Diarization Error Rate): {result.der:.2%}")
        print(f"  - Miss Rate: {result.miss:.2%}")
        print(f"  - False Alarm Rate: {result.false_alarm:.2%}")
        print(f"  - Confusion Rate: {result.confusion:.2%}")
        print(f"  總時長: {result.total_duration:.1f} 秒")
        print(f"  正確識別: {result.correct_duration:.1f} 秒")


def main():
    """評估範例"""
    import argparse
    
    parser = argparse.ArgumentParser(description="評估語者分離準確度")
    parser.add_argument("--reference", "-r", required=True, help="Ground Truth SRT")
    parser.add_argument("--hypothesis", "-h", required=True, help="預測結果 SRT")
    
    args = parser.parse_args()
    
    evaluator = DiarizationEvaluator()
    result = evaluator.evaluate_srt(args.reference, args.hypothesis)
    evaluator.print_result(result)


if __name__ == "__main__":
    main()
