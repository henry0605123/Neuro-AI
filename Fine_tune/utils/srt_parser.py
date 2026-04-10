"""
SRT 檔案解析工具
================
解析 SRT 格式的字幕檔案，提取說話人標籤和時間戳
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SRTSegment:
    """SRT 段落資料結構"""
    index: int
    start_time: float  # 秒
    end_time: float    # 秒
    speaker: str
    text: str


class SRTParser:
    """SRT 檔案解析器"""
    
    # 時間格式正則表達式
    # 支援 00:00:00,000 和 00:00:00 兩種格式
    TIME_PATTERN = re.compile(
        r'(\d{1,2}):(\d{2}):?(\d{2})?[,.]?(\d{0,3})?'
    )
    
    # 時間軸正則表達式
    TIMELINE_PATTERN = re.compile(
        r'(\d{1,2}:\d{2}(?::\d{2})?(?:[,.]\d{1,3})?)\s*-->\s*(\d{1,2}:\d{2}(?::\d{2})?(?:[,.]\d{1,3})?)'
    )
    
    # 說話人標籤正則表達式
    # 支援「測試者:」、「小孩：」等格式
    SPEAKER_PATTERN = re.compile(r'^(測試者|小孩|Unknown|SPEAKER_\d+)[：:]?\s*(.*)$')
    
    @classmethod
    def parse_time(cls, time_str: str) -> float:
        """
        將時間字串轉換為秒數
        
        支援格式：
        - 00:00:00,000
        - 00:00:00
        - 00:00
        """
        time_str = time_str.strip()
        match = cls.TIME_PATTERN.match(time_str)
        
        if not match:
            raise ValueError(f"無法解析時間格式: {time_str}")
        
        parts = match.groups()
        
        # 處理不同的時間格式
        if parts[2] is not None:
            # 完整格式: HH:MM:SS,mmm
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            milliseconds = int(parts[3]) if parts[3] else 0
        else:
            # 簡短格式: MM:SS 或 MM:SS,mmm
            hours = 0
            minutes = int(parts[0])
            seconds = int(parts[1])
            milliseconds = int(parts[3]) if parts[3] else 0
        
        total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
        return total_seconds
    
    @classmethod
    def format_time(cls, seconds: float) -> str:
        """將秒數轉換為 SRT 時間格式"""
        if seconds < 0:
            seconds = 0
            
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    @classmethod
    def parse_file(cls, file_path: str | Path) -> List[SRTSegment]:
        """
        解析 SRT 檔案
        
        Args:
            file_path: SRT 檔案路徑
            
        Returns:
            SRTSegment 列表
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"找不到檔案: {file_path}")
        
        # 嘗試不同編碼
        encodings = ['utf-8', 'utf-8-sig', 'big5', 'gbk']
        content = None
        
        for encoding in encodings:
            try:
                content = file_path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            raise ValueError(f"無法讀取檔案: {file_path}")
        
        return cls.parse_content(content)
    
    @classmethod
    def parse_content(cls, content: str) -> List[SRTSegment]:
        """
        解析 SRT 內容
        
        Args:
            content: SRT 檔案內容
            
        Returns:
            SRTSegment 列表
        """
        segments = []
        
        # 以空行分割段落
        blocks = re.split(r'\n\s*\n', content.strip())
        
        for block in blocks:
            lines = block.strip().split('\n')
            
            if len(lines) < 2:
                continue
            
            try:
                # 第一行是序號
                index = int(lines[0].strip())
                
                # 第二行是時間軸
                timeline_match = cls.TIMELINE_PATTERN.match(lines[1].strip())
                if not timeline_match:
                    continue
                
                start_time = cls.parse_time(timeline_match.group(1))
                end_time = cls.parse_time(timeline_match.group(2))
                
                # 剩餘行是文字內容（可能有多行）
                text_lines = lines[2:]
                
                # 解析每一行的說話人和文字
                for text_line in text_lines:
                    text_line = text_line.strip()
                    if not text_line:
                        continue
                    
                    # 嘗試匹配說話人
                    speaker_match = cls.SPEAKER_PATTERN.match(text_line)
                    
                    if speaker_match:
                        speaker = speaker_match.group(1)
                        text = speaker_match.group(2).strip()
                    else:
                        speaker = "Unknown"
                        text = text_line
                    
                    if text:  # 只保留有內容的段落
                        segments.append(SRTSegment(
                            index=index,
                            start_time=start_time,
                            end_time=end_time,
                            speaker=speaker,
                            text=text
                        ))
                
            except (ValueError, IndexError) as e:
                # 跳過格式錯誤的段落
                continue
        
        return segments
    
    @classmethod
    def get_speaker_segments(cls, segments: List[SRTSegment]) -> Dict[str, List[Tuple[float, float]]]:
        """
        按說話人分組時間段
        
        Args:
            segments: SRTSegment 列表
            
        Returns:
            {說話人: [(開始時間, 結束時間), ...]}
        """
        speaker_segments = {}
        
        for seg in segments:
            if seg.speaker not in speaker_segments:
                speaker_segments[seg.speaker] = []
            speaker_segments[seg.speaker].append((seg.start_time, seg.end_time))
        
        return speaker_segments
    
    @classmethod
    def get_total_duration(cls, segments: List[SRTSegment]) -> float:
        """取得總時長"""
        if not segments:
            return 0.0
        return max(seg.end_time for seg in segments)
    
    @classmethod
    def get_speaker_statistics(cls, segments: List[SRTSegment]) -> Dict[str, Dict]:
        """
        取得說話人統計資訊
        
        Returns:
            {說話人: {count: 段落數, duration: 總時長}}
        """
        stats = {}
        
        for seg in segments:
            if seg.speaker not in stats:
                stats[seg.speaker] = {'count': 0, 'duration': 0.0}
            
            stats[seg.speaker]['count'] += 1
            stats[seg.speaker]['duration'] += seg.end_time - seg.start_time
        
        return stats


if __name__ == "__main__":
    # 測試
    test_file = r"C:\自閉症\data\Fine_tune\20250210-19683409吳苡恩\Aegisub.srt"
    
    segments = SRTParser.parse_file(test_file)
    
    print(f"總共 {len(segments)} 個段落")
    print(f"總時長: {SRTParser.get_total_duration(segments):.2f} 秒")
    
    print("\n說話人統計:")
    stats = SRTParser.get_speaker_statistics(segments)
    for speaker, info in stats.items():
        print(f"  {speaker}: {info['count']} 段, {info['duration']:.2f} 秒")
    
    print("\n前 5 個段落:")
    for seg in segments[:5]:
        print(f"  [{seg.start_time:.2f}-{seg.end_time:.2f}] {seg.speaker}: {seg.text[:30]}...")
