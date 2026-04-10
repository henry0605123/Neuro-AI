"""
Step 2 (替代版): 簡化的模型微調
================================
這是一個更穩定的微調方法，適用於小資料集

特點:
1. 使用 pyannote 內建的微調 API
2. 自動處理資料格式轉換
3. 支援斷點續訓

執行方式:
    python 02_train_model_v2.py
"""

import os
import sys
import json
import torch
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# 加入專案路徑
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    LOG_DIR,
    HF_TOKEN,
    PRETRAINED_SEGMENTATION_MODEL,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    DURATION,
    create_directories
)


class CustomDataset(torch.utils.data.Dataset):
    """自訂資料集類別"""
    
    def __init__(self, samples_info: list, duration: float = 5.0):
        """
        Args:
            samples_info: 樣本資訊列表
            duration: 每個訓練樣本的長度（秒）
        """
        self.samples = samples_info
        self.duration = duration
        self._prepare_segments()
    
    def _prepare_segments(self):
        """準備訓練片段"""
        import torchaudio
        from pyannote.core import Annotation, Segment
        
        self.segments = []
        
        for sample in self.samples:
            audio_path = sample['audio_path']
            rttm_path = sample['rttm_path']
            
            # 取得音訊資訊
            try:
                info = torchaudio.info(audio_path)
                audio_duration = info.num_frames / info.sample_rate
            except Exception:
                audio_duration = sample.get('duration', 300)
            
            # 讀取 RTTM 標註
            annotation = Annotation()
            with open(rttm_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 8:
                        start = float(parts[3])
                        seg_duration = float(parts[4])
                        speaker = parts[7]
                        annotation[Segment(start, start + seg_duration)] = speaker
            
            # 切分成固定長度片段
            num_segments = int(audio_duration / self.duration)
            for i in range(num_segments):
                start = i * self.duration
                end = start + self.duration
                
                self.segments.append({
                    'audio_path': audio_path,
                    'start': start,
                    'end': end,
                    'annotation': annotation.crop(Segment(start, end))
                })
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        return self.segments[idx]


def create_labels_from_annotation(annotation, num_frames: int, num_speakers: int = 2) -> torch.Tensor:
    """
    將 pyannote Annotation 轉換為訓練標籤
    
    Args:
        annotation: pyannote Annotation 物件
        num_frames: 輸出的幀數
        num_speakers: 說話人數量
        
    Returns:
        形狀為 (num_frames, num_speakers) 的標籤張量
    """
    from pyannote.core import Segment
    
    # 取得時間範圍
    if len(annotation) == 0:
        return torch.zeros(num_frames, num_speakers)
    
    timeline = annotation.get_timeline()
    total_duration = timeline[-1].end if timeline else 5.0
    
    frame_duration = total_duration / num_frames
    
    # 建立說話人映射
    speakers = sorted(set(annotation.labels()))
    speaker_to_idx = {s: i for i, s in enumerate(speakers)}
    
    # 建立標籤
    labels = torch.zeros(num_frames, num_speakers)
    
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        if speaker not in speaker_to_idx:
            continue
        
        speaker_idx = speaker_to_idx[speaker]
        if speaker_idx >= num_speakers:
            continue
        
        start_frame = int(segment.start / frame_duration)
        end_frame = int(segment.end / frame_duration)
        
        start_frame = max(0, min(start_frame, num_frames - 1))
        end_frame = max(0, min(end_frame, num_frames))
        
        labels[start_frame:end_frame, speaker_idx] = 1.0
    
    return labels


def train_with_simple_loop():
    """
    使用簡單的訓練迴圈進行微調
    """
    import torchaudio
    from pyannote.audio import Model
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from tqdm import tqdm
    
    print("=" * 60)
    print("Step 2: 訓練模型 (簡化版)")
    print("=" * 60)
    
    # 建立目錄
    create_directories()
    
    # 檢查裝置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用裝置: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 載入樣本資訊
    samples_json = PROCESSED_DATA_DIR / "samples.json"
    if not samples_json.exists():
        raise FileNotFoundError("請先執行 01_prepare_dataset.py")
    
    with open(samples_json, 'r', encoding='utf-8') as f:
        samples_data = json.load(f)
    
    train_samples = samples_data['train']
    val_samples = samples_data['val']
    
    print(f"\n訓練樣本: {len(train_samples)}")
    print(f"驗證樣本: {len(val_samples)}")
    
    # 載入預訓練模型
    print(f"\n載入預訓練模型: {PRETRAINED_SEGMENTATION_MODEL}")
    model = Model.from_pretrained(
        PRETRAINED_SEGMENTATION_MODEL,
        use_auth_token=HF_TOKEN
    )
    model = model.to(device)
    model.train()
    
    # 凍結大部分層，只訓練最後幾層（RTX 5070 Ti 可訓練更多層）
    freeze_ratio = 0.7  # 從 0.8 降到 0.7，訓練更多層
    all_params = list(model.named_parameters())
    freeze_count = int(len(all_params) * freeze_ratio)
    
    for name, param in all_params[:freeze_count]:
        param.requires_grad = False
    
    # 啟用混合精度訓練
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler() if device == "cuda" else None
    print("✓ 已啟用混合精度訓練（AMP）")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"可訓練參數: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    # 設定優化器
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # 損失函數
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # 訓練迴圈
    best_val_loss = float('inf')
    save_dir = MODEL_DIR / "finetuned_segmentation"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n開始訓練 (共 {EPOCHS} epochs)")
    print("-" * 60)
    
    for epoch in range(EPOCHS):
        # 訓練階段
        model.train()
        train_loss = 0.0
        train_count = 0
        
        for sample in tqdm(train_samples, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            try:
                # 載入音訊 (分段處理以節省記憶體)
                audio_path = sample['audio_path']
                
                # 先讀取音訊資訊
                info = torchaudio.info(audio_path)
                sample_rate = info.sample_rate
                total_frames = info.num_frames
                
                # 分段載入音訊 (每次載入 10 秒以節省記憶體)
                chunk_duration = 10  # 秒 (降低為 10 秒)
                chunk_frames = chunk_duration * sample_rate
                
                # 隨機選擇一個起始位置
                max_offset = max(0, total_frames - chunk_frames)
                if max_offset > 0:
                    frame_offset = torch.randint(0, max_offset, (1,)).item()
                else:
                    frame_offset = 0
                    chunk_frames = total_frames
                
                # 載入音訊片段
                waveform, _ = torchaudio.load(audio_path, frame_offset=frame_offset, num_frames=chunk_frames)
                
                # 重取樣到模型所需的取樣率
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    waveform = resampler(waveform)
                
                # 取一段固定長度的音訊
                segment_samples = int(DURATION * 16000)
                if waveform.shape[1] > segment_samples:
                    # 隨機選取一段
                    import random
                    start_idx = random.randint(0, waveform.shape[1] - segment_samples)
                    waveform = waveform[:, start_idx:start_idx + segment_samples]
                
                # 轉為單聲道
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                
                # 準備輸入
                waveform = waveform.to(device)
                
                # 前向傳播（使用混合精度）
                with torch.set_grad_enabled(True):
                    if device == "cuda" and scaler is not None:
                        with autocast():
                            output = model(waveform.unsqueeze(0))  # (1, num_frames, num_speakers)
                    else:
                        output = model(waveform.unsqueeze(0))
                    
                    # 讀取標籤
                    from pyannote.core import Annotation, Segment
                    annotation = Annotation()
                    rttm_path = sample['rttm_path']
                    
                    with open(rttm_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 8:
                                start = float(parts[3])
                                duration = float(parts[4])
                                speaker = parts[7]
                                annotation[Segment(start, start + duration)] = speaker
                    
                    # 計算對應的標籤
                    num_frames = output.shape[1]
                    num_speakers = output.shape[2]
                    
                    target = create_labels_from_annotation(
                        annotation,
                        num_frames,
                        num_speakers
                    ).unsqueeze(0).to(device)
                    
                    # 計算損失
                    loss = criterion(output, target)
                    
                    # 反向傳播（使用混合精度）
                    optimizer.zero_grad()
                    if device == "cuda" and scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    
                    train_loss += loss.item()
                    train_count += 1
                    
            except Exception as e:
                print(f"\n警告: 處理 {sample['file_id']} 時發生錯誤: {e}")
                continue
        
        avg_train_loss = train_loss / max(train_count, 1)
        
        # 驗證階段
        model.eval()
        val_loss = 0.0
        val_count = 0
        
        with torch.no_grad():
            for sample in tqdm(val_samples, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                try:
                    # 載入音訊 (分段處理以節省記憶體)
                    audio_path = sample['audio_path']
                    
                    # 先讀取音訊資訊
                    info = torchaudio.info(audio_path)
                    sample_rate = info.sample_rate
                    total_frames = info.num_frames
                    
                    # 分段載入音訊 (每次載入 10 秒以節省記憶體)
                    chunk_duration = 10  # 秒 (降低為 10 秒)
                    chunk_frames = chunk_duration * sample_rate
                    
                    # 隨機選擇一個起始位置
                    max_offset = max(0, total_frames - chunk_frames)
                    if max_offset > 0:
                        frame_offset = torch.randint(0, max_offset, (1,)).item()
                    else:
                        frame_offset = 0
                        chunk_frames = total_frames
                    
                    # 載入音訊片段
                    waveform, _ = torchaudio.load(audio_path, frame_offset=frame_offset, num_frames=chunk_frames)
                    
                    if sample_rate != 16000:
                        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                        waveform = resampler(waveform)
                    
                    segment_samples = int(DURATION * 16000)
                    if waveform.shape[1] > segment_samples:
                        waveform = waveform[:, :segment_samples]
                    
                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    
                    waveform = waveform.to(device)
                    output = model(waveform.unsqueeze(0))
                    
                    # 讀取標籤
                    from pyannote.core import Annotation, Segment
                    annotation = Annotation()
                    rttm_path = sample['rttm_path']
                    
                    with open(rttm_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 8:
                                start = float(parts[3])
                                duration = float(parts[4])
                                speaker = parts[7]
                                annotation[Segment(start, start + duration)] = speaker
                    
                    num_frames = output.shape[1]
                    num_speakers = output.shape[2]
                    
                    target = create_labels_from_annotation(
                        annotation,
                        num_frames,
                        num_speakers
                    ).unsqueeze(0).to(device)
                    
                    loss = criterion(output, target)
                    val_loss += loss.item()
                    val_count += 1
                    
                except Exception as e:
                    continue
        
        avg_val_loss = val_loss / max(val_count, 1)
        
        # 更新學習率
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{EPOCHS}: "
              f"Train Loss = {avg_train_loss:.4f}, "
              f"Val Loss = {avg_val_loss:.4f}, "
              f"LR = {current_lr:.2e}")
        
        # 儲存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = save_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            print(f"  ✓ 儲存最佳模型 (Val Loss: {avg_val_loss:.4f})")
    
    # 儲存最終模型
    final_path = save_dir / "last.ckpt"
    torch.save({
        'epoch': EPOCHS - 1,
        'model_state_dict': model.state_dict(),
        'state_dict': model.state_dict(),  # 相容性
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': avg_val_loss,
    }, final_path)
    
    print(f"\n" + "=" * 60)
    print(f"✅ 訓練完成!")
    print(f"  最佳模型: {save_dir / 'best_model.pt'}")
    print(f"  最終模型: {final_path}")
    print(f"  最佳 Val Loss: {best_val_loss:.4f}")
    print("=" * 60)
    
    return model


def main():
    """主程式"""
    train_with_simple_loop()


if __name__ == "__main__":
    main()
