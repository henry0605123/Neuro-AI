"""
音訊處理工具
============
處理音訊檔案，包括格式轉換、取得時長等
"""

import os
from pathlib import Path
from typing import Optional, Tuple
import subprocess


class AudioUtils:
    """音訊處理工具類"""
    
    @staticmethod
    def get_duration_ffprobe(audio_path: str | Path) -> Optional[float]:
        """
        使用 ffprobe 取得音訊時長
        
        Args:
            audio_path: 音訊檔案路徑
            
        Returns:
            音訊時長（秒），失敗則返回 None
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            return None
        
        try:
            result = subprocess.run(
                [
                    'ffprobe', '-v', 'error',
                    '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    str(audio_path)
                ],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return float(result.stdout.strip())
        except (subprocess.SubprocessError, ValueError, FileNotFoundError):
            pass
        
        return None
    
    @staticmethod
    def get_duration_pydub(audio_path: str | Path) -> Optional[float]:
        """
        使用 pydub 取得音訊時長
        
        Args:
            audio_path: 音訊檔案路徑
            
        Returns:
            音訊時長（秒），失敗則返回 None
        """
        try:
            from pydub import AudioSegment
            
            audio_path = Path(audio_path)
            
            if not audio_path.exists():
                return None
            
            # 根據副檔名載入
            ext = audio_path.suffix.lower()
            
            if ext == '.mp3':
                audio = AudioSegment.from_mp3(str(audio_path))
            elif ext == '.wav':
                audio = AudioSegment.from_wav(str(audio_path))
            elif ext == '.ogg':
                audio = AudioSegment.from_ogg(str(audio_path))
            elif ext == '.flac':
                audio = AudioSegment.from_file(str(audio_path), format='flac')
            else:
                audio = AudioSegment.from_file(str(audio_path))
            
            return len(audio) / 1000.0  # 毫秒轉秒
            
        except Exception:
            return None
    
    @staticmethod
    def get_duration_torchaudio(audio_path: str | Path) -> Optional[float]:
        """
        使用 torchaudio 取得音訊時長
        
        Args:
            audio_path: 音訊檔案路徑
            
        Returns:
            音訊時長（秒），失敗則返回 None
        """
        try:
            import torchaudio
            
            audio_path = Path(audio_path)
            
            if not audio_path.exists():
                return None
            
            info = torchaudio.info(str(audio_path))
            return info.num_frames / info.sample_rate
            
        except Exception:
            return None
    
    @classmethod
    def get_duration(cls, audio_path: str | Path) -> Optional[float]:
        """
        取得音訊時長（自動選擇可用方法）
        
        Args:
            audio_path: 音訊檔案路徑
            
        Returns:
            音訊時長（秒），失敗則返回 None
        """
        # 嘗試不同方法
        methods = [
            cls.get_duration_torchaudio,
            cls.get_duration_pydub,
            cls.get_duration_ffprobe,
        ]
        
        for method in methods:
            duration = method(audio_path)
            if duration is not None:
                return duration
        
        return None
    
    @staticmethod
    def convert_to_wav(
        input_path: str | Path,
        output_path: str | Path = None,
        sample_rate: int = 16000,
        mono: bool = True
    ) -> Optional[Path]:
        """
        將音訊轉換為 WAV 格式
        
        Args:
            input_path: 輸入檔案路徑
            output_path: 輸出檔案路徑（預設與輸入同目錄，副檔名改為 .wav）
            sample_rate: 取樣率
            mono: 是否轉為單聲道
            
        Returns:
            輸出檔案路徑，失敗則返回 None
        """
        try:
            from pydub import AudioSegment
            
            input_path = Path(input_path)
            
            if output_path is None:
                output_path = input_path.with_suffix('.wav')
            else:
                output_path = Path(output_path)
            
            # 載入音訊
            audio = AudioSegment.from_file(str(input_path))
            
            # 轉換
            if mono:
                audio = audio.set_channels(1)
            
            audio = audio.set_frame_rate(sample_rate)
            
            # 匯出
            output_path.parent.mkdir(parents=True, exist_ok=True)
            audio.export(str(output_path), format='wav')
            
            return output_path
            
        except Exception as e:
            print(f"轉換失敗: {e}")
            return None
    
    @staticmethod
    def get_audio_info(audio_path: str | Path) -> Optional[dict]:
        """
        取得音訊資訊
        
        Args:
            audio_path: 音訊檔案路徑
            
        Returns:
            包含音訊資訊的字典
        """
        try:
            import torchaudio
            
            audio_path = Path(audio_path)
            
            if not audio_path.exists():
                return None
            
            info = torchaudio.info(str(audio_path))
            
            return {
                'sample_rate': info.sample_rate,
                'num_channels': info.num_channels,
                'num_frames': info.num_frames,
                'duration': info.num_frames / info.sample_rate,
                'bits_per_sample': info.bits_per_sample,
                'encoding': info.encoding,
            }
            
        except Exception:
            return None


if __name__ == "__main__":
    # 測試
    test_audio = r"C:\自閉症\data\Fine_tune\20250210-19683409吳苡恩\20250210-19683409吳苡恩.mp3"
    
    print(f"測試音檔: {test_audio}")
    
    duration = AudioUtils.get_duration(test_audio)
    print(f"時長: {duration:.2f} 秒" if duration else "無法取得時長")
    
    info = AudioUtils.get_audio_info(test_audio)
    if info:
        print(f"音訊資訊: {info}")
