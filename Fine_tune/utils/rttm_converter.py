"""
RTTM 格式轉換工具
=================
將 SRT 標註轉換為 pyannote 訓練所需的 RTTM 格式
"""

from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from .srt_parser import SRTParser, SRTSegment


@dataclass
class RTTMLine:
    """RTTM 格式的單行資料"""
    type: str = "SPEAKER"
    file_id: str = ""
    channel: int = 1
    start_time: float = 0.0
    duration: float = 0.0
    ortho: str = "<NA>"
    stype: str = "<NA>"
    speaker: str = ""
    conf: str = "<NA>"
    slat: str = "<NA>"
    
    def to_string(self) -> str:
        """轉換為 RTTM 格式字串"""
        return (f"{self.type} {self.file_id} {self.channel} "
                f"{self.start_time:.3f} {self.duration:.3f} "
                f"{self.ortho} {self.stype} {self.speaker} "
                f"{self.conf} {self.slat}")


class RTTMConverter:
    """RTTM 格式轉換器"""
    
    @staticmethod
    def normalize_speaker(speaker: str, mapping: Dict[str, str] = None) -> str:
        """
        標準化說話人標籤
        
        Args:
            speaker: 原始說話人標籤
            mapping: 說話人映射表
            
        Returns:
            標準化後的說話人標籤
        """
        if mapping is None:
            mapping = {
                "測試者": "SPEAKER_00",
                "小孩": "SPEAKER_01",
                "Unknown": "SPEAKER_02",
            }
        
        return mapping.get(speaker, speaker)
    
    @classmethod
    def segments_to_rttm(
        cls,
        segments: List[SRTSegment],
        file_id: str,
        speaker_mapping: Dict[str, str] = None,
        merge_threshold: float = 0.3
    ) -> List[RTTMLine]:
        """
        將 SRT 段落轉換為 RTTM 格式
        
        Args:
            segments: SRT 段落列表
            file_id: 檔案識別碼
            speaker_mapping: 說話人映射表
            merge_threshold: 同說話人連續段落合併閾值（秒）
            
        Returns:
            RTTMLine 列表
        """
        if not segments:
            return []
        
        # 按時間排序
        sorted_segments = sorted(segments, key=lambda x: x.start_time)
        
        # 合併連續的同說話人段落
        merged_segments = []
        current = None
        
        for seg in sorted_segments:
            speaker = cls.normalize_speaker(seg.speaker, speaker_mapping)
            
            if current is None:
                current = {
                    'speaker': speaker,
                    'start': seg.start_time,
                    'end': seg.end_time
                }
            elif (current['speaker'] == speaker and 
                  seg.start_time - current['end'] <= merge_threshold):
                # 合併連續段落
                current['end'] = max(current['end'], seg.end_time)
            else:
                merged_segments.append(current)
                current = {
                    'speaker': speaker,
                    'start': seg.start_time,
                    'end': seg.end_time
                }
        
        if current is not None:
            merged_segments.append(current)
        
        # 轉換為 RTTM 格式
        rttm_lines = []
        for seg in merged_segments:
            duration = seg['end'] - seg['start']
            if duration > 0:
                rttm_lines.append(RTTMLine(
                    file_id=file_id,
                    start_time=seg['start'],
                    duration=duration,
                    speaker=seg['speaker']
                ))
        
        return rttm_lines
    
    @classmethod
    def srt_to_rttm(
        cls,
        srt_path: str | Path,
        output_path: str | Path = None,
        file_id: str = None,
        speaker_mapping: Dict[str, str] = None
    ) -> List[RTTMLine]:
        """
        將 SRT 檔案轉換為 RTTM 檔案
        
        Args:
            srt_path: SRT 檔案路徑
            output_path: RTTM 輸出路徑（可選）
            file_id: 檔案識別碼（預設使用檔名）
            speaker_mapping: 說話人映射表
            
        Returns:
            RTTMLine 列表
        """
        srt_path = Path(srt_path)
        
        if file_id is None:
            file_id = srt_path.stem
        
        # 解析 SRT
        segments = SRTParser.parse_file(srt_path)
        
        # 轉換為 RTTM
        rttm_lines = cls.segments_to_rttm(segments, file_id, speaker_mapping)
        
        # 如果指定了輸出路徑，寫入檔案
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for line in rttm_lines:
                    f.write(line.to_string() + '\n')
        
        return rttm_lines
    
    @classmethod
    def create_uem(
        cls,
        file_id: str,
        duration: float,
        output_path: str | Path = None
    ) -> str:
        """
        建立 UEM (Un-partitioned Evaluation Map) 檔案
        
        UEM 檔案指定音檔中需要評估的區間
        
        Args:
            file_id: 檔案識別碼
            duration: 音檔總長度（秒）
            output_path: 輸出路徑（可選）
            
        Returns:
            UEM 格式字串
        """
        uem_line = f"{file_id} 1 0.000 {duration:.3f}"
        
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(uem_line + '\n')
        
        return uem_line
    
    @classmethod
    def create_lst_file(
        cls,
        file_ids: List[str],
        output_path: str | Path
    ) -> None:
        """
        建立 LST 檔案（檔案清單）
        
        Args:
            file_ids: 檔案識別碼列表
            output_path: 輸出路徑
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for file_id in file_ids:
                f.write(file_id + '\n')


if __name__ == "__main__":
    # 測試
    test_srt = r"C:\自閉症\data\Fine_tune\20250210-19683409吳苡恩\Aegisub.srt"
    
    rttm_lines = RTTMConverter.srt_to_rttm(test_srt, file_id="test_audio")
    
    print(f"產生 {len(rttm_lines)} 行 RTTM")
    print("\n前 5 行:")
    for line in rttm_lines[:5]:
        print(f"  {line.to_string()}")
