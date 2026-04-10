# -*- coding: utf-8 -*-
"""
論文復現：Exploring Speech Foundation Models for Speaker Diarization in Child-Adult Dyadic Interactions
資料集模組：SRT 解析、Frame-level 標籤生成、音訊切片與 DataLoader

作者：AI Engineering Team
日期：2026-01-19
修訂：2026-01-19 - 改為 3 分類任務 (Child=0, Adult=1, Silence=2)
             - 修復 BOM 解析問題
             - 移除 Overlap，改用 Last-Write-Wins 策略
             - 2026-03-24 - 支援 RTTM（Playlogue / NIST）；與 SRT 並存時優先 RTTM（對齊 Fine_tune_LSTM）
"""

import os
import re
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import torchaudio
from tqdm import tqdm

from config import Config


# ==================== SRT 解析器 ====================

@dataclass
class SRTSegment:
    """SRT 字幕片段資料結構"""
    index: int
    start_time: float  # 秒
    end_time: float    # 秒
    speaker: str       # "小孩", "測試者", "overlap" 等
    text: str          # 字幕內容


class SRTParser:
    """
    SRT 字幕檔解析器
    ================
    
    支援格式：
    1. 標準格式：
       測試者: 內容文字
       小孩: 內容文字
    
    2. 可處理多語者同時說話（Overlap）
    """
    
    # 語者標籤對應
    SPEAKER_MAPPING = {
        "小孩": 0,      # Child
        "兒童": 0,
        "child": 0,
        "測試者": 1,    # Adult
        "大人": 1,
        "成人": 1,
        "adult": 1,
        "examiner": 1,
        "tester": 1,
    }
    
    def __init__(self):
        self.segments: List[SRTSegment] = []
    
    @staticmethod
    def parse_time(time_str: str) -> float:
        """
        解析 SRT 時間格式，轉換為秒數
        
        支援格式：
        - 00:00:02,475 (標準 SRT 格式)
        - 00:01:11 (簡化格式)
        - 00:01:11.500 (帶毫秒的簡化格式)
        
        Args:
            time_str: 時間字串
            
        Returns:
            float: 秒數
        """
        time_str = time_str.strip()
        
        # 處理逗號（標準 SRT 格式使用逗號分隔毫秒）
        time_str = time_str.replace(",", ".")
        
        parts = time_str.split(":")
        
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            
            # 處理秒和毫秒
            sec_parts = parts[2].split(".")
            seconds = int(sec_parts[0])
            milliseconds = int(sec_parts[1]) if len(sec_parts) > 1 else 0
            
            # 如果毫秒是三位數，需要正確轉換
            if len(sec_parts) > 1:
                ms_str = sec_parts[1]
                if len(ms_str) == 3:
                    milliseconds = int(ms_str)
                elif len(ms_str) == 2:
                    milliseconds = int(ms_str) * 10
                elif len(ms_str) == 1:
                    milliseconds = int(ms_str) * 100
            
            return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
        
        elif len(parts) == 2:
            # 格式：MM:SS 或 MM:SS.mmm
            minutes = int(parts[0])
            sec_parts = parts[1].split(".")
            seconds = int(sec_parts[0])
            milliseconds = int(sec_parts[1]) if len(sec_parts) > 1 else 0
            
            return minutes * 60 + seconds + milliseconds / 1000
        
        else:
            raise ValueError(f"無法解析時間格式: {time_str}")
    
    @staticmethod
    def detect_speakers(text: str) -> List[Tuple[str, str]]:
        """
        從文字中偵測語者及其對話內容
        
        Args:
            text: 字幕文字（可能包含多個語者）
            
        Returns:
            List[Tuple[str, str]]: [(語者, 內容), ...]
        """
        speakers = []
        
        # 定義可能的語者標籤模式
        # 支援格式：「測試者:」、「測試者：」、「小孩:」等
        patterns = [
            r'(測試者|小孩|兒童|大人|成人)[：:]\s*(.+?)(?=(?:測試者|小孩|兒童|大人|成人)[：:]|$)',
            r'(child|adult|examiner|tester)[：:]\s*(.+?)(?=(?:child|adult|examiner|tester)[：:]|$)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                for speaker, content in matches:
                    speaker = speaker.strip().lower()
                    content = content.strip()
                    if content:
                        speakers.append((speaker, content))
        
        # 如果沒有偵測到語者，嘗試簡單分割
        if not speakers:
            # 嘗試偵測是否有語者前綴
            simple_pattern = r'^(測試者|小孩|兒童|大人|成人|child|adult)[：:]\s*(.+)$'
            lines = text.strip().split('\n')
            for line in lines:
                match = re.match(simple_pattern, line.strip(), re.IGNORECASE)
                if match:
                    speaker = match.group(1).strip().lower()
                    content = match.group(2).strip()
                    if content:
                        speakers.append((speaker, content))
        
        return speakers
    
    def parse_file(self, file_path: str) -> List[SRTSegment]:
        """
        解析 SRT 檔案
        
        Args:
            file_path: SRT 檔案路徑
            
        Returns:
            List[SRTSegment]: 字幕片段列表
        """
        self.segments = []
        
        # 讀取檔案，嘗試多種編碼
        content = None
        for encoding in ['utf-8', 'utf-8-sig', 'cp950', 'big5', 'gbk']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        if content is None:
            raise ValueError(f"無法讀取檔案（編碼錯誤）: {file_path}")
        
        # 移除 BOM (Byte Order Mark)
        if content.startswith('\ufeff'):
            content = content[1:]
        
        # 分割字幕區塊
        blocks = re.split(r'\n\s*\n', content.strip())
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) < 2:
                continue
            
            try:
                # 第一行：序號
                index = int(lines[0].strip())
                
                # 第二行：時間碼
                time_line = lines[1].strip()
                time_match = re.match(
                    r'([\d:,\.]+)\s*-->\s*([\d:,\.]+)',
                    time_line
                )
                
                if not time_match:
                    continue
                
                start_time = self.parse_time(time_match.group(1))
                end_time = self.parse_time(time_match.group(2))
                
                # 第三行及之後：字幕文字
                text = '\n'.join(lines[2:]).strip()
                
                # 偵測語者
                speakers = self.detect_speakers(text)
                
                if speakers:
                    # 根據偵測到的語者數量判斷是否為 overlap
                    if len(speakers) > 1:
                        # 多個語者 = Overlap
                        self.segments.append(SRTSegment(
                            index=index,
                            start_time=start_time,
                            end_time=end_time,
                            speaker="overlap",
                            text=text
                        ))
                    else:
                        # 單一語者
                        speaker, _ = speakers[0]
                        # 標準化語者名稱
                        normalized_speaker = self._normalize_speaker(speaker)
                        self.segments.append(SRTSegment(
                            index=index,
                            start_time=start_time,
                            end_time=end_time,
                            speaker=normalized_speaker,
                            text=text
                        ))
                else:
                    # 無法偵測語者，嘗試根據文字內容推斷
                    # 預設為 silence 或跳過
                    pass
                    
            except (ValueError, IndexError) as e:
                print(f"警告：解析字幕區塊時發生錯誤: {e}")
                continue
        
        return self.segments
    
    def _normalize_speaker(self, speaker: str) -> str:
        """標準化語者名稱"""
        speaker = speaker.lower().strip()
        if speaker in ["小孩", "兒童", "child"]:
            return "小孩"
        elif speaker in ["測試者", "大人", "成人", "adult", "examiner", "tester"]:
            return "測試者"
        else:
            return speaker
    
    def get_segments(self) -> List[SRTSegment]:
        """取得已解析的字幕片段"""
        return self.segments


# ==================== RTTM 解析器 (NIST / Playlogue) ====================


class RTTMParser:
    """
    RTTM 語者分段解析，輸出 SRTSegment 列表供 FrameLabelGenerator 使用。
    Playlogue 使用 CHI（兒童）、ADT（成人）；其餘常見標籤映射為成人或略過。
    """

    def __init__(self):
        self.segments: List[SRTSegment] = []

    @staticmethod
    def _map_rttm_speaker(tag: str) -> Optional[str]:
        t = tag.strip().upper()
        if t == "CHI" or t.startswith("CHI"):
            return "child"
        if t in (
            "ADT",
            "ADU",
            "ADULT",
            "INV",
            "MOT",
            "FAT",
            "MOTHER",
            "FATHER",
            "PARENT",
            "EXAMINER",
        ):
            return "adult"
        return None

    def parse_file(self, file_path: str) -> List[SRTSegment]:
        self.segments = []
        lines: Optional[List[str]] = None
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    lines = f.readlines()
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        if lines is None:
            raise ValueError(f"無法讀取 RTTM（編碼錯誤）: {file_path}")

        idx = 0
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            try:
                start = float(parts[3])
                duration = float(parts[4])
            except (ValueError, IndexError):
                continue
            spk_tag = parts[7]
            canon = self._map_rttm_speaker(spk_tag)
            if canon is None:
                continue
            idx += 1
            end_time = start + duration
            self.segments.append(
                SRTSegment(
                    index=idx,
                    start_time=start,
                    end_time=end_time,
                    speaker=canon,
                    text=f"[{spk_tag}]",
                )
            )
        return self.segments

    def get_segments(self) -> List[SRTSegment]:
        return self.segments


def load_annotation_segments(
    sample: "AudioSample",
    srt_parser: SRTParser,
    rttm_parser: RTTMParser,
) -> List[SRTSegment]:
    """若存在有效 RTTM 路徑則讀 RTTM，否則讀 SRT。"""
    if sample.rttm_path and os.path.isfile(sample.rttm_path):
        return rttm_parser.parse_file(sample.rttm_path)
    if sample.srt_path and os.path.isfile(sample.srt_path):
        return srt_parser.parse_file(sample.srt_path)
    raise FileNotFoundError(
        f"找不到標註檔（RTTM 或 SRT）: subject={sample.subject_id}"
    )


# ==================== Frame-level 標籤生成器 ====================

class FrameLabelGenerator:
    """
    Frame-level 標籤生成器
    ======================
    
    將 SRT 時間標註轉換為 frame-level 的標籤
    
    3 分類標籤定義（無 Overlap）：
    - 0: Child (小孩)
    - 1: Adult (測試者/大人)
    - 2: Silence (靜音/背景)
    
    使用 Last-Write-Wins 策略：如果時間重疊，新的語者標籤直接覆蓋舊的
    """
    
    LABEL_CHILD = 0
    LABEL_ADULT = 1
    LABEL_SILENCE = 2  # 從 3 改為 2
    
    def __init__(self, step_size_ms: float = 20.0):
        """
        初始化標籤生成器
        
        Args:
            step_size_ms: frame 的步長（毫秒），與 Whisper 一致為 20ms
        """
        self.step_size_ms = step_size_ms
        self.step_size_sec = step_size_ms / 1000.0
    
    def generate_labels(
        self, 
        segments: List[SRTSegment], 
        total_duration: float
    ) -> np.ndarray:
        """
        根據 SRT 片段生成 frame-level 標籤
        
        使用 Last-Write-Wins 策略：後面的標註會覆蓋前面的
        
        Args:
            segments: SRT 字幕片段列表
            total_duration: 音訊總時長（秒）
            
        Returns:
            np.ndarray: shape (num_frames,), 每個 frame 的標籤
        """
        num_frames = int(np.ceil(total_duration / self.step_size_sec))
        labels = np.full(num_frames, self.LABEL_SILENCE, dtype=np.int64)
        
        for seg in segments:
            start_frame = int(seg.start_time / self.step_size_sec)
            end_frame = int(seg.end_time / self.step_size_sec)
            
            # 確保不超出範圍
            start_frame = max(0, start_frame)
            end_frame = min(num_frames, end_frame)
            
            # 根據語者設定標籤
            speaker = seg.speaker.lower()
            
            if speaker in ["小孩", "兒童", "child"]:
                label = self.LABEL_CHILD
            elif speaker in ["測試者", "大人", "成人", "adult", "examiner", "tester"]:
                label = self.LABEL_ADULT
            else:
                # 未知語者，預設為靜音
                label = self.LABEL_SILENCE
            
            # Last-Write-Wins 策略：直接覆蓋，不生成 Overlap
            labels[start_frame:end_frame] = label
        
        return labels
    
    def generate_labels_for_window(
        self,
        labels: np.ndarray,
        start_time: float,
        window_size: float
    ) -> np.ndarray:
        """
        擷取特定時間窗口的標籤
        
        Args:
            labels: 完整的 frame-level 標籤
            start_time: 窗口開始時間（秒）
            window_size: 窗口大小（秒）
            
        Returns:
            np.ndarray: 窗口內的標籤
        """
        start_frame = int(start_time / self.step_size_sec)
        num_frames = int(window_size / self.step_size_sec)
        end_frame = start_frame + num_frames
        
        # 處理超出範圍的情況
        if end_frame > len(labels):
            # 如果超出範圍，用 silence 填充
            window_labels = np.full(num_frames, self.LABEL_SILENCE, dtype=np.int64)
            valid_frames = len(labels) - start_frame
            if valid_frames > 0:
                window_labels[:valid_frames] = labels[start_frame:len(labels)]
            return window_labels
        
        return labels[start_frame:end_frame].copy()


# ==================== 音訊處理器 ====================

class AudioProcessor:
    """
    音訊處理器
    ==========
    
    負責音訊載入、重採樣與切片
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        初始化音訊處理器
        
        Args:
            sample_rate: 目標取樣率（Whisper 使用 16kHz）
        """
        self.sample_rate = sample_rate
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, float]:
        """
        載入音訊檔案並重採樣
        
        Args:
            audio_path: 音訊檔案路徑
            
        Returns:
            Tuple[np.ndarray, float]: (音訊波形, 時長秒數)
        """
        try:
            # 使用 librosa 載入並重採樣
            waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            duration = len(waveform) / self.sample_rate
            return waveform, duration
        except Exception as e:
            print(f"載入音訊失敗 {audio_path}: {e}")
            raise
    
    def slice_audio(
        self,
        waveform: np.ndarray,
        start_time: float,
        window_size: float
    ) -> np.ndarray:
        """
        切取音訊片段
        
        Args:
            waveform: 完整音訊波形
            start_time: 開始時間（秒）
            window_size: 窗口大小（秒）
            
        Returns:
            np.ndarray: 切取的音訊片段
        """
        start_sample = int(start_time * self.sample_rate)
        num_samples = int(window_size * self.sample_rate)
        end_sample = start_sample + num_samples
        
        # 處理超出範圍的情況
        if end_sample > len(waveform):
            # 用零填充
            audio_slice = np.zeros(num_samples, dtype=np.float32)
            valid_samples = len(waveform) - start_sample
            if valid_samples > 0:
                audio_slice[:valid_samples] = waveform[start_sample:len(waveform)]
            return audio_slice
        
        return waveform[start_sample:end_sample].astype(np.float32)


# ==================== 資料集類別 ====================

@dataclass
class AudioSample:
    """音訊樣本資料結構"""
    subject_id: str           # 受試者 ID
    audio_path: str           # 音訊檔案路徑
    srt_path: str             # SRT 標註檔案路徑（可為空字串，若僅使用 RTTM）
    waveform: Optional[np.ndarray] = None  # 預載入的波形
    labels: Optional[np.ndarray] = None    # 預載入的標籤
    duration: float = 0.0     # 音訊時長
    rttm_path: Optional[str] = None  # RTTM；若設定且檔案存在則優先於 SRT


class SpeakerDiarizationDataset(Dataset):
    """
    語者分離資料集
    ==============
    
    負責管理音訊與標籤資料，提供訓練用的切片樣本
    """
    
    def __init__(
        self,
        samples: List[AudioSample],
        config: Config,
        is_training: bool = True,
        preload: bool = True
    ):
        """
        初始化資料集
        
        Args:
            samples: 音訊樣本列表
            config: 配置物件
            is_training: 是否為訓練模式（影響資料增強與切片方式）
            preload: 是否預先載入所有音訊到記憶體
        """
        self.samples = samples
        self.config = config
        self.is_training = is_training
        self.preload = preload
        
        self.audio_processor = AudioProcessor(config.sample_rate)
        self.srt_parser = SRTParser()
        self.rttm_parser = RTTMParser()
        self.label_generator = FrameLabelGenerator(config.step_size_ms)
        
        # 預先載入音訊與標籤
        if preload:
            self._preload_data()
        
        # 建立切片索引
        self._build_slice_indices()
    
    def _preload_data(self):
        """預先載入所有音訊與標籤"""
        print("預先載入音訊資料...")
        for sample in tqdm(self.samples, desc="載入中"):
            try:
                # 載入音訊
                waveform, duration = self.audio_processor.load_audio(sample.audio_path)
                sample.waveform = waveform
                sample.duration = duration
                
                segments = load_annotation_segments(
                    sample, self.srt_parser, self.rttm_parser
                )
                sample.labels = self.label_generator.generate_labels(segments, duration)
                
            except Exception as e:
                print(f"載入樣本 {sample.subject_id} 時發生錯誤: {e}")
    
    def _build_slice_indices(self):
        """
        建立切片索引
        
        根據訓練/測試模式決定切片方式：
        - 訓練模式：使用重疊切片（50% overlap）
        - 測試模式：不重疊切片
        """
        self.slices = []
        
        for sample_idx, sample in enumerate(self.samples):
            if sample.duration <= 0:
                continue
            
            # 計算切片步長
            if self.is_training:
                step = self.config.window_size * (1 - self.config.train_overlap_ratio)
            else:
                step = self.config.window_size
            
            # 生成切片起始時間
            start_time = 0.0
            while start_time + self.config.window_size <= sample.duration:
                self.slices.append((sample_idx, start_time))
                start_time += step
            
            # 處理最後一個不完整的切片（如果有足夠長度）
            remaining = sample.duration - start_time
            if remaining > self.config.window_size * 0.5:  # 至少有一半長度
                # 調整起始時間以獲得完整窗口
                start_time = max(0, sample.duration - self.config.window_size)
                if len(self.slices) == 0 or self.slices[-1] != (sample_idx, start_time):
                    self.slices.append((sample_idx, start_time))
            
            # 評估模式：短音訊仍產生至少一個窗口（尾部由 slice_audio 零填充）
            if not self.is_training:
                has_any = any(si == sample_idx for si, _ in self.slices)
                if not has_any and sample.duration > 0:
                    self.slices.append((sample_idx, 0.0))
        
        print(f"資料集切片數量: {len(self.slices)}")
    
    def __len__(self) -> int:
        return len(self.slices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        取得單一訓練樣本
        
        Args:
            idx: 樣本索引
            
        Returns:
            Dict: 包含音訊與標籤的字典
        """
        sample_idx, start_time = self.slices[idx]
        sample = self.samples[sample_idx]
        
        # 取得音訊切片
        if self.preload and sample.waveform is not None:
            audio_slice = self.audio_processor.slice_audio(
                sample.waveform, start_time, self.config.window_size
            )
        else:
            # 動態載入
            waveform, _ = self.audio_processor.load_audio(sample.audio_path)
            audio_slice = self.audio_processor.slice_audio(
                waveform, start_time, self.config.window_size
            )
        
        # 取得標籤切片
        if self.preload and sample.labels is not None:
            label_slice = self.label_generator.generate_labels_for_window(
                sample.labels, start_time, self.config.window_size
            )
        else:
            segments = load_annotation_segments(
                sample, self.srt_parser, self.rttm_parser
            )
            full_labels = self.label_generator.generate_labels(segments, sample.duration)
            label_slice = self.label_generator.generate_labels_for_window(
                full_labels, start_time, self.config.window_size
            )
        
        return {
            "audio": torch.from_numpy(audio_slice).float(),
            "labels": torch.from_numpy(label_slice).long(),
            "sample_idx": sample_idx,
            "start_time": start_time,
        }


# ==================== 資料集建構函式 ====================

def discover_training_samples_from_dir(train_dir: Path) -> List[AudioSample]:
    """
    掃描單一根目錄下每個子資料夾：音訊 +（RTTM 優先，否則 SRT）。
    """
    samples: List[AudioSample] = []
    if not train_dir.exists():
        raise FileNotFoundError(f"資料目錄不存在: {train_dir}")

    print(f"探索資料目錄: {train_dir}")

    for subject_folder in sorted(train_dir.iterdir()):
        if not subject_folder.is_dir():
            continue

        subject_id = subject_folder.name
        print(f"  處理受試者: {subject_id}")

        audio_path = None
        for ext in [".mp3", ".wav", ".flac", ".m4a", ".MP3", ".WAV"]:
            audio_files = list(subject_folder.glob(f"*{ext}"))
            if audio_files:
                audio_path = str(audio_files[0])
                break

        if audio_path is None:
            print("    警告：找不到音訊檔案")
            continue

        rttm_path: Optional[str] = None
        for rttm_name in [
            "diarization.rttm",
            "GTruth.rttm",
            "ground_truth.rttm",
            "annotation.rttm",
        ]:
            p = subject_folder / rttm_name
            if p.is_file():
                rttm_path = str(p)
                break
        if rttm_path is None:
            rttm_files = sorted(subject_folder.glob("*.rttm"))
            if rttm_files:
                rttm_path = str(rttm_files[0])

        srt_path: Optional[str] = None
        for srt_name in [
            "Aegisub.srt",
            "GTruth.srt",
            "GroundTruth.srt",
            "annotation.srt",
        ]:
            srt_file = subject_folder / srt_name
            if srt_file.exists():
                srt_path = str(srt_file)
                break
        if srt_path is None:
            srt_files = list(subject_folder.glob("*.srt"))
            if srt_files:
                srt_path = str(srt_files[0])

        if rttm_path is None and srt_path is None:
            print("    警告：找不到 RTTM 或 SRT 標註檔")
            continue

        print(f"    音訊: {os.path.basename(audio_path)}")
        if rttm_path:
            print(f"    RTTM: {os.path.basename(rttm_path)}")
        if srt_path:
            print(f"    SRT: {os.path.basename(srt_path)}")

        samples.append(
            AudioSample(
                subject_id=subject_id,
                audio_path=audio_path,
                srt_path=srt_path or "",
                rttm_path=rttm_path,
            )
        )

    print(f"共發現 {len(samples)} 個樣本")
    return samples


def discover_training_samples(config: Config) -> List[AudioSample]:
    """自動探索 config.train_data_dir 下的受試者資料。"""
    return discover_training_samples_from_dir(Path(config.train_data_dir))


def _normalize_ratio(config: Config) -> Tuple[int, int, int]:
    """取得並驗證固定切分比例（train, val, test）。"""
    ratio = getattr(config, "train_val_test_ratio", (2, 1, 1))
    if not isinstance(ratio, (tuple, list)) or len(ratio) != 3:
        raise ValueError("train_val_test_ratio 必須為長度 3 的 tuple/list，例如 (2, 1, 1)")
    train_r, val_r, test_r = int(ratio[0]), int(ratio[1]), int(ratio[2])
    if train_r < 0 or val_r < 0 or test_r < 0 or (train_r + val_r + test_r) <= 0:
        raise ValueError("train_val_test_ratio 各值需為非負整數，且總和必須 > 0")
    return train_r, val_r, test_r


def _split_samples_by_fixed_ratio(
    samples: List[AudioSample],
    ratio: Tuple[int, int, int],
    seed: int,
) -> Tuple[List[AudioSample], List[AudioSample], List[AudioSample]]:
    """以固定比例隨機切分樣本（可重現）。"""
    train_r, val_r, test_r = ratio
    total_r = train_r + val_r + test_r

    shuffled = sorted(samples, key=lambda s: s.subject_id)
    rnd = random.Random(seed)
    rnd.shuffle(shuffled)

    train_samples: List[AudioSample] = []
    val_samples: List[AudioSample] = []
    test_samples: List[AudioSample] = []

    train_cut = train_r
    val_cut = train_r + val_r

    for idx, sample in enumerate(shuffled):
        bucket_pos = idx % total_r
        if bucket_pos < train_cut:
            train_samples.append(sample)
        elif bucket_pos < val_cut:
            val_samples.append(sample)
        else:
            test_samples.append(sample)

    if not train_samples and shuffled:
        train_samples.append(shuffled[0])
        if shuffled[0] in val_samples:
            val_samples.remove(shuffled[0])
        if shuffled[0] in test_samples:
            test_samples.remove(shuffled[0])
    if not val_samples and len(shuffled) >= 2:
        candidate = next((s for s in shuffled if s not in train_samples), None)
        if candidate is not None:
            val_samples.append(candidate)
            if candidate in test_samples:
                test_samples.remove(candidate)
    if not test_samples and len(shuffled) >= 3:
        candidate = next((s for s in shuffled if s not in train_samples and s not in val_samples), None)
        if candidate is not None:
            test_samples.append(candidate)

    return train_samples, val_samples, test_samples


def split_train_val_test_from_config(
    config: Config,
) -> Tuple[List[AudioSample], List[AudioSample], List[AudioSample]]:
    """從 train_data_dir 依 config.train_val_test_ratio 隨機固定比例切分。"""
    samples = discover_training_samples(config)
    if len(samples) == 0:
        raise ValueError("沒有找到任何可用樣本，無法切分 train/val/test")
    ratio = _normalize_ratio(config)
    return _split_samples_by_fixed_ratio(samples, ratio=ratio, seed=config.seed)


def create_dataloaders(
    config: Config,
    val_split_ratio: Optional[float] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    建立訓練與驗證資料載入器
    
    Args:
        config: 配置物件
        val_split_ratio: 驗證集比例（預設使用 config 中的設定）
        
    Returns:
        Tuple[DataLoader, DataLoader]: (訓練用 DataLoader, 驗證用 DataLoader)
    """
    train_samples = discover_training_samples(config)
    if len(train_samples) == 0:
        raise ValueError("沒有找到任何訓練樣本！")

    val_dir_raw = getattr(config, "val_data_dir", None) or ""
    val_dir = val_dir_raw.strip() if isinstance(val_dir_raw, str) else ""
    enable_fixed_split = bool(getattr(config, "enable_fixed_split_from_train_dir", False))

    if val_dir and os.path.isdir(val_dir):
        val_samples = discover_training_samples_from_dir(Path(val_dir))
        if len(val_samples) == 0:
            raise ValueError(f"val_data_dir 下沒有有效樣本: {val_dir}")
        print("\n資料集分割（使用獨立驗證目錄 val_data_dir）：")
        print(f"  訓練集: {len(train_samples)} 個受試者")
        print(f"  驗證集: {len(val_samples)} 個受試者")
    elif enable_fixed_split:
        ratio = _normalize_ratio(config)
        train_samples, val_samples, test_samples = split_train_val_test_from_config(config)
        print("\n資料集分割（固定比例隨機切分）：")
        print(f"  比例 train/val/test = {ratio[0]}/{ratio[1]}/{ratio[2]}")
        print(f"  訓練集: {len(train_samples)} 個受試者")
        print(f"  驗證集: {len(val_samples)} 個受試者")
        print(f"  測試集: {len(test_samples)} 個受試者（保留供 Step 2 評估）")
    else:
        random.seed(config.seed)
        shuffled_samples = train_samples.copy()
        random.shuffle(shuffled_samples)
        val_ratio = val_split_ratio if val_split_ratio is not None else config.val_split_ratio
        val_size = max(1, int(len(shuffled_samples) * val_ratio))
        val_samples = shuffled_samples[:val_size]
        train_samples = shuffled_samples[val_size:]
        if len(train_samples) == 0:
            train_samples = val_samples
            print("警告：訓練集為空，將使用驗證集作為訓練集")
        print("\n資料集分割（隨機比例）：")
        print(f"  訓練集: {len(train_samples)} 個受試者")
        print(f"  驗證集: {len(val_samples)} 個受試者")
    
    # 建立資料集
    train_dataset = SpeakerDiarizationDataset(
        samples=train_samples,
        config=config,
        is_training=True,
        preload=True
    )
    
    val_dataset = SpeakerDiarizationDataset(
        samples=val_samples,
        config=config,
        is_training=False,
        preload=True
    )
    
    # 建立 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
        persistent_workers=config.num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
        persistent_workers=config.num_workers > 0
    )
    
    return train_loader, val_loader


def create_test_dataloader(config: Config) -> DataLoader:
    """
    建立測試資料載入器
    
    Args:
        config: 配置物件
        
    Returns:
        DataLoader: 測試用 DataLoader
    """
    enable_fixed_split = bool(getattr(config, "enable_fixed_split_from_train_dir", False))
    test_dir_raw = getattr(config, "test_data_dir", None)
    test_dir = test_dir_raw.strip() if isinstance(test_dir_raw, str) else ""

    ap = getattr(config, "test_audio_path", "") or ""
    ap = ap.strip() if isinstance(ap, str) else ""
    if (not test_dir or not os.path.isdir(test_dir)) and ap and os.path.isdir(ap):
        test_dir = ap
        print(
            "\n提示: test_audio_path 指向目錄，已改為多 clip 測試模式掃描該路徑。"
            "建議在 config.py 使用 test_data_dir 指向測試集資料夾、test_audio_path 留空。"
        )

    if test_dir and os.path.isdir(test_dir):
        print(f"\n建立測試集（掃描目錄，與訓練相同結構）: {test_dir}")
        samples = discover_training_samples_from_dir(Path(test_dir))
        if len(samples) == 0:
            raise RuntimeError(
                "test_data_dir 下沒有可用測試樣本（長度為 0）。每個子資料夾須含音訊檔（wav/mp3/…）"
                "以及 RTTM 或 SRT。\n"
                f"目錄: {test_dir}\n"
                "若目前只有 diarization.rttm 而無音訊，請將對應音檔放入各 clip 子資料夾。"
            )
    elif enable_fixed_split:
        ratio = _normalize_ratio(config)
        _, _, samples = split_train_val_test_from_config(config)
        if len(samples) == 0:
            raise RuntimeError(
                "固定比例切分後測試集為空，請確認 train_data_dir 樣本數是否足夠，"
                "或調整 train_val_test_ratio。"
            )
        print("\n建立測試集（由 train_data_dir 固定比例隨機切分）：")
        print(f"  比例 train/val/test = {ratio[0]}/{ratio[1]}/{ratio[2]}")
        print(f"  測試集: {len(samples)} 個受試者")
    else:
        tr = getattr(config, "test_rttm_path", None)
        tr = tr.strip() if isinstance(tr, str) and tr.strip() else None
        test_sample = AudioSample(
            subject_id="test",
            audio_path=config.test_audio_path,
            srt_path=config.test_srt_path or "",
            rttm_path=tr,
        )
        samples = [test_sample]

    test_dataset = SpeakerDiarizationDataset(
        samples=samples,
        config=config,
        is_training=False,
        preload=True
    )

    if len(test_dataset) == 0:
        hint = (
            f"  1) 音檔是否存在且可讀: {getattr(config, 'test_audio_path', '')}\n"
            if not test_dir
            else f"  1) 測試目錄: {test_dir}\n"
        )
        raise RuntimeError(
            "測試資料集沒有任何可推論的切片（長度為 0）。請確認：\n"
            f"{hint}"
            "  2) 若終端機出現載入樣本錯誤，請檢查路徑、格式與標註檔\n"
            "  3) 音訊是否成功載入（duration > 0）"
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False
    )

    return test_loader


# ==================== 資料集統計分析 ====================

def analyze_dataset_statistics(config: Config):
    """
    分析資料集統計資訊
    
    Args:
        config: 配置物件
    """
    samples = discover_training_samples(config)
    
    audio_processor = AudioProcessor(config.sample_rate)
    srt_parser = SRTParser()
    rttm_parser = RTTMParser()
    label_generator = FrameLabelGenerator(config.step_size_ms)
    
    total_duration = 0.0
    class_counts = {0: 0, 1: 0, 2: 0}  # 3 分類: Child, Adult, Silence
    
    print("\n資料集統計分析")
    print("=" * 60)
    
    for sample in samples:
        try:
            waveform, duration = audio_processor.load_audio(sample.audio_path)
            segments = load_annotation_segments(sample, srt_parser, rttm_parser)
            labels = label_generator.generate_labels(segments, duration)
            
            total_duration += duration
            
            for label_idx in range(3):  # 3 分類
                class_counts[label_idx] += np.sum(labels == label_idx)
            
            print(f"\n受試者: {sample.subject_id}")
            print(f"  音訊時長: {duration:.2f} 秒 ({duration/60:.2f} 分鐘)")
            print(f"  字幕片段數: {len(segments)}")
            
            # 各類別時長統計
            frame_duration = config.step_size_ms / 1000.0
            for label_idx, label_name in enumerate(config.class_names):
                count = np.sum(labels == label_idx)
                label_duration = count * frame_duration
                percentage = count / len(labels) * 100
                print(f"  {label_name}: {label_duration:.2f} 秒 ({percentage:.1f}%)")
            
        except Exception as e:
            print(f"分析 {sample.subject_id} 時發生錯誤: {e}")
    
    print("\n" + "=" * 60)
    print("整體統計")
    print(f"  總時長: {total_duration:.2f} 秒 ({total_duration/60:.2f} 分鐘)")
    
    frame_duration = config.step_size_ms / 1000.0
    total_frames = sum(class_counts.values())
    for label_idx, label_name in enumerate(config.class_names):
        count = class_counts[label_idx]
        label_duration = count * frame_duration
        percentage = count / total_frames * 100 if total_frames > 0 else 0
        print(f"  {label_name}: {label_duration:.2f} 秒 ({percentage:.1f}%)")


if __name__ == "__main__":
    # 測試資料集模組
    config = Config()
    config.print_config()
    
    # 分析資料集統計
    analyze_dataset_statistics(config)
    
    # 測試建立 DataLoader
    print("\n測試建立 DataLoader...")
    try:
        train_loader, val_loader = create_dataloaders(config)
        print(f"訓練批次數: {len(train_loader)}")
        print(f"驗證批次數: {len(val_loader)}")
        
        # 測試取得一個批次
        for batch in train_loader:
            print(f"\n批次資料形狀:")
            print(f"  音訊: {batch['audio'].shape}")
            print(f"  標籤: {batch['labels'].shape}")
            break
            
    except Exception as e:
        print(f"建立 DataLoader 時發生錯誤: {e}")
