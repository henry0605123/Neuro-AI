# 📊 Pyannote Fine-tune V2 優化報告

## 目錄

1. [問題診斷](#問題診斷)
2. [原因分析](#原因分析)
3. [優化策略](#優化策略)
4. [實作細節](#實作細節)
5. [預期效果](#預期效果)
6. [程式碼對比](#程式碼對比)

---

## 問題診斷

### 原始版本 (V1) 的效能問題

根據評估結果，V1 模型表現如下：

| 指標 | 數值 | 問題嚴重度 |
|------|------|-----------|
| DER | 56.37% | 🔴 嚴重 |
| 小孩 Precision | 0.00% | 🔴 嚴重 |
| 小孩 Recall | 0.00% | 🔴 嚴重 |
| 小孩 F1 | 0.00% | 🔴 嚴重 |
| 測試者 Precision | 80.10% | 🟢 良好 |
| 測試者 Recall | 60.48% | 🟡 普通 |
| 測試者 F1 | 68.92% | 🟡 普通 |

### 核心問題

**小孩的所有指標為 0%**，意味著模型完全無法識別小孩的語音。

---

## 原因分析

### 問題 1：資料利用率極低 (最關鍵)

#### 原始程式碼邏輯

```python
# V1 訓練迴圈 (02_train_model_v2.py)
for sample in tqdm(train_samples, ...):  # train_samples = 3 個檔案
    # 隨機選擇一個起始位置
    frame_offset = torch.randint(0, max_offset, (1,)).item()
    
    # 只載入 10 秒音訊
    waveform, _ = torchaudio.load(audio_path, frame_offset=frame_offset, num_frames=chunk_frames)
    
    # 再從中取 5 秒
    waveform = waveform[:, start_idx:start_idx + segment_samples]
```

#### 問題分析

```
【資料浪費計算】

原始資料：
- 3 個訓練檔案
- 每個檔案約 30 分鐘 = 1800 秒
- 每個檔案約 500 個標註段落
- 總計: ~1500 個有標註的語音段落

每個 Epoch 實際使用：
- 每個檔案只隨機取 1 個 5 秒片段
- 每個 Epoch 實際看到: 3 個檔案 × 1 個片段 = 3 個片段 = 15 秒

資料利用率：
- 總資料: 90 分鐘 = 5400 秒
- 每 Epoch 使用: 15 秒
- 利用率: 15 / 5400 = 0.28%

結論: 99.7% 的資料被浪費！
```

### 問題 2：標籤嚴重不平衡

#### 統計分析

從 RTTM 檔案分析（以吳忻芸為例）：

```
說話人時長統計：
- 測試者 (SPEAKER_00): ~25 分鐘 (83%)
- 小孩 (SPEAKER_01): ~5 分鐘 (17%)

在任意 5 秒窗口內：
- 平均測試者說話: ~4.15 秒
- 平均小孩說話: ~0.85 秒
```

#### 模型行為

```
【損失函數計算】

如果模型全部猜「測試者」：
- 測試者部分: 正確率 83%
- 小孩部分: 錯誤，但只佔 17%
- 平均損失: 相對較低

如果模型嘗試猜「小孩」：
- 小孩部分: 可能正確，但只佔 17%
- 測試者部分: 可能錯誤，佔 83%
- 平均損失: 可能更高

結論: 標準 BCE 損失會讓模型傾向「全猜測試者」
```

### 問題 3：小孩語音特徵學習不足

#### 小孩語音特點

1. **段落短**：小孩回答通常 0.5-2 秒（「會啊」「我啊」）
2. **出現稀疏**：在 5 秒窗口中可能只有幾百毫秒
3. **聲學特徵差異大**：兒童音調較高、語速不穩定

#### 訓練問題

```
【5 秒窗口分析】

典型的 5 秒訓練片段：
[0-1秒]: 測試者說話
[1-2秒]: 靜音
[2-2.5秒]: 小孩回答「好」
[2.5-5秒]: 測試者繼續說

標籤分佈：
- 測試者: 2.5 秒 (50%)
- 小孩: 0.5 秒 (10%)
- 靜音: 2 秒 (40%)

如果每個 Epoch 只看 3 個這樣的片段：
- 小孩語音總共只有: 3 × 0.5 = 1.5 秒
- 完全不足以學習小孩的聲學特徵
```

### 問題 4：訓練標籤對齊問題

#### 原始程式碼問題

```python
# V1 中標籤建立邏輯
def create_labels_from_annotation(annotation, num_frames, num_speakers):
    # 使用完整音檔的標註
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        # 但實際訓練的是音檔中隨機的 5 秒
        # 標籤和音訊沒有正確對齊！
```

#### 具體問題

```
【標籤錯位示意】

假設音檔總長 30 分鐘，訓練時：
1. 隨機取 offset = 300 秒（第 5 分鐘開始）
2. 載入 300-305 秒的音訊
3. 但標籤建立使用的是完整的 annotation

問題：
- 音訊是 300-305 秒
- 但標籤可能包含 0-10 秒的資訊
- 標籤與音訊完全不對應！

這會導致模型學到錯誤的對應關係
```

---

## 優化策略

### 策略 1：資料切片預處理 (01_prepare_dataset_v2.py)

**核心改進**：在資料準備階段就將音檔切成小片段

```python
【新策略】

原始：每個 Epoch，每個檔案取 1 個隨機片段
     → 每 Epoch 只有 3 個片段（15 秒）

優化：預先將每個檔案切成重疊的 5 秒片段
     → 1 個 30 分鐘檔案 ≈ 700 個片段
     → 3 個檔案 ≈ 2000 個片段
     → 每 Epoch 使用全部 2000 個片段

資料利用率提升：2000 / 3 = 666 倍！
```

#### 切片參數設計

```python
# config_v2.py
SEGMENT_DURATION = 5.0   # 每個片段 5 秒
SEGMENT_OVERLAP = 0.5    # 50% 重疊

# 計算
步長 = 5.0 × (1 - 0.5) = 2.5 秒
30分鐘 = 1800 秒
片段數 = 1800 / 2.5 = 720 個
```

#### 過濾條件

```python
# 只保留有意義的片段
MIN_SPEECH_RATIO = 0.1        # 至少 10% 是語音（過濾純靜音）
REQUIRE_CHILD_SPEECH = True   # 優先保留有小孩語音的片段
MIN_CHILD_RATIO = 0.05        # 至少 5% 是小孩語音
```

### 策略 2：類別加權損失 (02_train_model_v3.py)

**核心改進**：給予少數類別（小孩）更高的損失權重

```python
【加權策略】

標準 BCE：
Loss = BCE(pred_測試者, label_測試者) + BCE(pred_小孩, label_小孩)
     = 大部分來自測試者的損失

加權 BCE：
Loss = 1.0 × BCE(pred_測試者, label_測試者) + 3.0 × BCE(pred_小孩, label_小孩)
     = 小孩的損失被放大 3 倍

效果：
- 猜錯小孩的懲罰是猜錯測試者的 3 倍
- 模型被迫學習識別小孩
```

#### 自動權重計算

```python
def compute_class_weights(all_segments):
    """根據實際分佈計算權重"""
    
    # 統計各類別時長
    total_by_speaker = defaultdict(float)
    for seg in all_segments:
        for speaker, ratio in seg.speaker_ratios.items():
            total_by_speaker[speaker] += ratio * duration
    
    # 逆頻率加權
    # 出現越少 → 權重越高
    total_all = sum(total_by_speaker.values())
    for speaker, duration in total_by_speaker.items():
        freq = duration / total_all
        weights[speaker] = 1.0 / (freq * num_classes)
    
    return weights
```

### 策略 3：Focal Loss (02_train_model_v3.py)

**核心改進**：自動調整每個樣本的權重，聚焦於困難樣本

```python
【Focal Loss 原理】

標準 BCE：
Loss = -y×log(p) - (1-y)×log(1-p)

Focal Loss：
Loss = -α × (1-p)^γ × y×log(p) - (1-α) × p^γ × (1-y)×log(1-p)

其中：
- γ (gamma) = 2.0：聚焦參數
- (1-p)^γ：當預測很確定時，此項很小；當預測不確定時，此項較大

效果：
- 對於「容易」的樣本（模型很確定）：降低權重
- 對於「困難」的樣本（模型不確定）：增加權重
- 模型會自動聚焦於語音邊界、小孩語音等困難情況
```

#### 實作

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None):
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
    
    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 計算焦點權重
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma
        
        # 應用類別權重
        if self.class_weights is not None:
            weight = self.class_weights.view(1, 1, -1)
            focal_loss = weight * focal_weight * bce_loss
        else:
            focal_loss = focal_weight * bce_loss
        
        return focal_loss.mean()
```

### 策略 4：資料增強 (02_train_model_v3.py)

**核心改進**：增加訓練資料的多樣性

```python
【增強方法】

1. 時間遮罩 (Time Masking)
   - 隨機將一小段時間設為 0
   - 模擬說話人被短暫打斷
   - 讓模型學習處理不完整的語音

2. 音量變化 (Volume Change)
   - 隨機調整音量 (0.8x ~ 1.2x)
   - 模擬不同錄音條件
   - 讓模型對音量變化更穩健

3. 加入噪音 (Add Noise)
   - 加入高斯噪音 (SNR 15-30dB)
   - 模擬真實環境噪音
   - 提高模型在噪音環境下的表現
```

#### 實作

```python
class AudioAugmentation:
    def time_mask(self, waveform, max_length=0.5):
        if random.random() > TIME_MASK_PROB:
            return waveform
        
        mask_length = int(random.uniform(0.1, max_length) * self.sample_rate)
        mask_start = random.randint(0, length - mask_length)
        
        waveform = waveform.clone()
        waveform[..., mask_start:mask_start + mask_length] = 0
        return waveform
    
    def volume_change(self, waveform):
        if random.random() > VOLUME_CHANGE_PROB:
            return waveform
        
        factor = random.uniform(*VOLUME_CHANGE_RANGE)
        return waveform * factor
    
    def add_noise(self, waveform):
        if random.random() > ADD_NOISE_PROB:
            return waveform
        
        snr_db = random.uniform(*NOISE_SNR_RANGE)
        signal_power = waveform.pow(2).mean()
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(waveform) * noise_power.sqrt()
        
        return waveform + noise
```

### 策略 5：學習率調整 (02_train_model_v3.py)

**核心改進**：使用更智慧的學習率調度

```python
【學習率策略】

V1：ReduceLROnPlateau
- 當驗證損失停滯時降低學習率
- 被動調整，可能錯過最佳時機

V2：OneCycleLR
- 先升後降的循環學習率
- 主動探索最佳學習率
- 更快收斂

                    最大學習率
                    /\
                   /  \
                  /    \
                 /      \
初始學習率 ----/        \---- 最終學習率
               
    |----預熱----|----衰減----|
```

#### 實作

```python
self.scheduler = OneCycleLR(
    self.optimizer,
    max_lr=LEARNING_RATE,           # 最大學習率
    total_steps=total_steps,         # 總步數
    pct_start=0.1,                   # 預熱佔 10%
    anneal_strategy='cos'            # 餘弦衰減
)
```

---

## 實作細節

### 資料準備流程 (01_prepare_dataset_v2.py)

```
【完整流程】

1. 讀取原始資料夾
   ├── 20250210-19683409吳苡恩/
   │   ├── audio.mp3
   │   └── Aegisub.srt
   └── ...

2. 轉換 SRT → RTTM
   SPEAKER file_id 1 3.45 1.5 <NA> <NA> SPEAKER_01 <NA>

3. 切片
   ├── 片段 0: [0.0, 5.0] 秒
   ├── 片段 1: [2.5, 7.5] 秒 (50% 重疊)
   ├── 片段 2: [5.0, 10.0] 秒
   └── ...

4. 計算每個片段的統計
   ├── speech_ratio: 0.75 (75% 是語音)
   ├── child_ratio: 0.15 (15% 是小孩)
   └── has_child_speech: True

5. 過濾
   ├── 移除 speech_ratio < 0.1 的片段
   └── 優先保留有小孩語音的片段

6. 計算類別權重
   ├── SPEAKER_00: 1.0
   ├── SPEAKER_01: 2.5 (根據逆頻率計算)
   └── SPEAKER_02: 0.5

7. 輸出
   ├── segments_v2.json (片段資訊)
   └── class_weights.json (類別權重)
```

### 訓練流程 (02_train_model_v3.py)

```
【完整流程】

1. 載入資料
   ├── 從 segments_v2.json 讀取片段
   └── 從 class_weights.json 讀取權重

2. 建立 Dataset
   ├── 預載入所有 RTTM 標註
   └── 啟用資料增強 (訓練集)

3. 載入預訓練模型
   ├── pyannote/segmentation-3.0
   └── 凍結前 50% 的層

4. 設定損失函數
   └── FocalLoss(alpha=0.25, gamma=2.0, weights)

5. 訓練迴圈
   For epoch in range(30):
       ├── 訓練
       │   ├── 載入批次 (32 個片段)
       │   ├── 前向傳播 (混合精度)
       │   ├── 計算損失 (Focal Loss + 類別加權)
       │   ├── 反向傳播
       │   └── 更新學習率
       │
       ├── 驗證
       │   ├── 計算驗證損失
       │   └── 計算 Precision/Recall/F1
       │
       └── 儲存
           ├── 如果是最佳 → 儲存 best_model.pt
           └── 儲存 checkpoint.pt (可續訓)

6. 早停
   └── 連續 10 個 epoch 無改善則停止
```

### 標籤建立邏輯

```python
【正確的標籤建立】

def _create_labels(self, start_time, end_time, annotations, num_frames):
    """
    關鍵：只使用片段時間範圍內的標註
    """
    duration = end_time - start_time
    frame_duration = duration / num_frames
    
    labels = torch.zeros(num_frames, self.num_speakers)
    
    for ann_start, ann_end, speaker in annotations:
        # 計算標註與片段的重疊區域
        overlap_start = max(start_time, ann_start)
        overlap_end = min(end_time, ann_end)
        
        if overlap_end > overlap_start:
            # 轉換為相對於片段開始的幀索引
            start_frame = int((overlap_start - start_time) / frame_duration)
            end_frame = int((overlap_end - start_time) / frame_duration)
            
            labels[start_frame:end_frame, speaker_idx] = 1.0
    
    return labels
```

---

## 預期效果

### 資料量對比

| 指標 | V1 | V2 | 提升 |
|------|-----|-----|------|
| 每 Epoch 片段數 | 3 | ~2000 | 666x |
| 每 Epoch 音訊量 | 15 秒 | ~166 分鐘 | 666x |
| 小孩語音曝光量 | ~1.5 秒 | ~25 分鐘 | 1000x |

### 預期指標改善

| 指標 | V1 | V2 預期 | 改善 |
|------|-----|---------|------|
| DER | 56% | <30% | -26% |
| 小孩 Recall | 0% | >50% | +50% |
| 小孩 F1 | 0% | >45% | +45% |
| 測試者 Recall | 60% | >70% | +10% |

### 改善原因

```
【小孩 Recall 提升原因】

1. 資料曝光量增加
   - 每 Epoch 看到的小孩語音: 1.5秒 → 25分鐘
   - 模型有足夠機會學習小孩的聲學特徵

2. 類別加權
   - 小孩權重 3.0，猜錯小孩的懲罰更大
   - 模型被迫學習識別小孩

3. Focal Loss
   - 自動聚焦於小孩語音（困難樣本）
   - 減少對「全猜測試者」的獎勵

4. 資料增強
   - 模擬不同情況下的小孩語音
   - 提高泛化能力
```

---

## 程式碼對比

### 資料載入對比

```python
# ===== V1 (錯誤) =====
for sample in train_samples:  # 只有 3 個檔案
    # 隨機選一個位置
    frame_offset = torch.randint(0, max_offset, (1,)).item()
    # 載入 10 秒
    waveform, _ = torchaudio.load(audio_path, frame_offset=frame_offset, num_frames=chunk_frames)
    # 再取 5 秒
    waveform = waveform[:, start_idx:start_idx + segment_samples]
    # 問題：每個 Epoch 每個檔案只用 1 個片段

# ===== V2 (正確) =====
class SegmentDataset(Dataset):
    def __init__(self, segments):  # 預先切好的 2000 個片段
        self.segments = segments
    
    def __getitem__(self, idx):
        segment = self.segments[idx]
        
        # 直接載入對應的片段
        frame_offset = int(segment['start_time'] * sample_rate)
        num_frames = int((segment['end_time'] - segment['start_time']) * sample_rate)
        
        waveform, _ = torchaudio.load(audio_path, frame_offset=frame_offset, num_frames=num_frames)
        # 每個 Epoch 使用全部 2000 個片段
```

### 損失函數對比

```python
# ===== V1 (標準 BCE) =====
criterion = torch.nn.BCEWithLogitsLoss()
loss = criterion(output, target)
# 問題：不考慮類別不平衡

# ===== V2 (Focal Loss + 類別加權) =====
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma, class_weights):
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights  # [1.0, 3.0, 0.5]
    
    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Focal 權重：聚焦於困難樣本
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma
        
        # 類別權重：平衡不同說話人
        weight = self.class_weights.view(1, 1, -1)
        
        return (weight * focal_weight * bce_loss).mean()
```

### 標籤建立對比

```python
# ===== V1 (可能錯位) =====
def create_labels_from_annotation(annotation, num_frames, num_speakers):
    # 使用完整音檔的標註
    total_duration = timeline[-1].end  # 完整音檔長度
    frame_duration = total_duration / num_frames
    
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        start_frame = int(segment.start / frame_duration)  # 相對於完整音檔
        end_frame = int(segment.end / frame_duration)
        labels[start_frame:end_frame, speaker_idx] = 1.0
    # 問題：如果訓練的是音檔中間的 5 秒，標籤會錯位

# ===== V2 (正確對齊) =====
def _create_labels(self, start_time, end_time, annotations, num_frames):
    duration = end_time - start_time  # 只考慮片段長度
    frame_duration = duration / num_frames
    
    for ann_start, ann_end, speaker in annotations:
        # 計算與當前片段的重疊
        overlap_start = max(start_time, ann_start)
        overlap_end = min(end_time, ann_end)
        
        if overlap_end > overlap_start:
            # 轉換為相對於片段開始的索引
            start_frame = int((overlap_start - start_time) / frame_duration)
            end_frame = int((overlap_end - start_time) / frame_duration)
            labels[start_frame:end_frame, speaker_idx] = 1.0
    # 正確：標籤與音訊片段精確對齊
```

---

## 總結

### 核心問題
1. 資料利用率只有 0.28%
2. 小孩語音樣本嚴重不足
3. 標準損失函數導致類別不平衡

### 解決方案
1. 預切片增加資料利用率 666 倍
2. 類別加權和 Focal Loss 平衡訓練
3. 資料增強提高泛化能力
4. 正確的標籤對齊邏輯

### 預期改善
- DER: 56% → <30%
- 小孩 Recall: 0% → >50%
- 小孩 F1: 0% → >45%

---

*文檔版本: 2.0*
*最後更新: 2026-01-12*
