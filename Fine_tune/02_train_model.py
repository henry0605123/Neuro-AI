"""
Step 2: 訓練模型
================
使用準備好的資料對 pyannote segmentation 模型進行微調

執行方式:
    python 02_train_model.py

注意事項:
    - 需要 GPU 支援
    - 建議 GPU 記憶體 >= 8GB
    - 訓練時間取決於資料量和硬體配置
"""

import os
import sys
import json
import torch
from pathlib import Path
from datetime import datetime

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


def setup_pyannote_database():
    """
    設定 pyannote 資料庫環境變數
    """
    database_yml = PROCESSED_DATA_DIR / "database.yml"
    
    if not database_yml.exists():
        raise FileNotFoundError(
            f"找不到 database.yml，請先執行 01_prepare_dataset.py\n"
            f"預期路徑: {database_yml}"
        )
    
    # 設定環境變數
    os.environ["PYANNOTE_DATABASE_CONFIG"] = str(database_yml)
    print(f"✓ 已設定資料庫配置: {database_yml}")


def load_pretrained_model(device: str):
    """
    載入預訓練的 segmentation 模型
    
    Args:
        device: 運算裝置
        
    Returns:
        預訓練模型
    """
    from pyannote.audio import Model
    
    print(f"正在載入預訓練模型: {PRETRAINED_SEGMENTATION_MODEL}")
    
    model = Model.from_pretrained(
        PRETRAINED_SEGMENTATION_MODEL,
        use_auth_token=HF_TOKEN
    )
    
    print(f"✓ 模型載入完成")
    print(f"  模型架構: {model.__class__.__name__}")
    
    return model


def create_training_task(model):
    """
    建立訓練任務
    
    Args:
        model: 預訓練模型
        
    Returns:
        訓練任務配置
    """
    from pyannote.audio.tasks import Segmentation
    from pyannote.database import get_protocol, FileFinder
    
    # 取得訓練協定
    protocol = get_protocol(
        "ASD_Diarization.SpeakerDiarization.finetune",
        preprocessors={"audio": FileFinder()}
    )
    
    # 建立 Segmentation 任務
    task = Segmentation(
        protocol,
        duration=DURATION,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"✓ 訓練任務建立完成")
    print(f"  Duration: {DURATION} 秒")
    print(f"  Batch size: {BATCH_SIZE}")
    
    return task


def train_model_simple(device: str = "cuda"):
    """
    簡化版訓練流程（直接使用 pyannote 的 fine-tuning API）
    """
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        LearningRateMonitor
    )
    from pytorch_lightning.loggers import TensorBoardLogger
    from pyannote.audio import Model
    from pyannote.audio.tasks import Segmentation
    from pyannote.database import get_protocol, FileFinder
    
    print("=" * 60)
    print("Step 2: 訓練模型 (簡化版)")
    print("=" * 60)
    
    # 設定資料庫
    setup_pyannote_database()
    
    # 檢查 GPU
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA 不可用，改用 CPU")
        device = "cpu"
    
    print(f"使用裝置: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 載入預訓練模型
    print(f"\n載入預訓練模型...")
    model = Model.from_pretrained(
        PRETRAINED_SEGMENTATION_MODEL,
        use_auth_token=HF_TOKEN
    )
    
    # 取得訓練協定
    print(f"\n設定訓練任務...")
    
    # 建立 FileFinder 來定位音訊檔案
    preprocessors = {"audio": FileFinder()}
    
    try:
        protocol = get_protocol(
            "ASD_Diarization.SpeakerDiarization.finetune",
            preprocessors=preprocessors
        )
    except Exception as e:
        print(f"無法載入協定，嘗試手動建立...")
        protocol = create_manual_protocol()
    
    # 建立任務（優化為 RTX 5070 Ti + Ryzen 7 7700）
    task = Segmentation(
        protocol,
        duration=DURATION,
        batch_size=BATCH_SIZE,
        num_workers=4 if device == "cuda" else 2,  # Windows 相容性
        pin_memory=True  # 加速資料傳輸到 GPU
    )
    
    # 將任務綁定到模型
    model.task = task
    model.setup(stage="fit")
    
    # 設定訓練回調
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    callbacks = [
        ModelCheckpoint(
            dirpath=MODEL_DIR / "finetuned_segmentation",
            filename="checkpoint-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            mode="min"
        ),
        LearningRateMonitor(logging_interval="epoch")
    ]
    
    # 設定 Logger
    logger = TensorBoardLogger(
        save_dir=LOG_DIR,
        name="finetune",
        version=timestamp
    )
    
    # 設定 Trainer（針對 RTX 5070 Ti 優化）
    trainer = pl.Trainer(
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        max_epochs=EPOCHS,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        log_every_n_steps=10,
        enable_progress_bar=True,
        precision="16-mixed",  # 混合精度訓練（RTX 5070 Ti Tensor Cores）
        benchmark=True,  # 自動尋找最佳演算法
        deterministic=False,  # 提升效能（犧牲可重現性）
        enable_model_summary=True
    )
    
    # 開始訓練
    print(f"\n開始訓練...")
    print(f"  最大 Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  模型儲存路徑: {MODEL_DIR / 'finetuned_segmentation'}")
    print(f"  TensorBoard 日誌: {LOG_DIR}")
    print("-" * 60)
    
    trainer.fit(model)
    
    # 儲存最終模型
    final_model_path = MODEL_DIR / "finetuned_segmentation" / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"\n✓ 最終模型已儲存: {final_model_path}")
    
    return model


def create_manual_protocol():
    """
    手動建立訓練協定（當自動載入失敗時使用）
    """
    from pyannote.database import Protocol, ProtocolFile
    from pyannote.core import Annotation, Segment
    
    # 讀取樣本資訊
    samples_json = PROCESSED_DATA_DIR / "samples.json"
    with open(samples_json, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    class ManualProtocol(Protocol):
        def __init__(self):
            super().__init__()
            self.train_samples = samples.get('train', [])
            self.val_samples = samples.get('val', [])
        
        def _load_annotation(self, rttm_path):
            """從 RTTM 檔案載入標註"""
            annotation = Annotation()
            
            with open(rttm_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 8:
                        start = float(parts[3])
                        duration = float(parts[4])
                        speaker = parts[7]
                        annotation[Segment(start, start + duration)] = speaker
            
            return annotation
        
        def train_iter(self):
            for sample in self.train_samples:
                yield {
                    "uri": sample['file_id'],
                    "audio": sample['audio_path'],
                    "annotation": self._load_annotation(sample['rttm_path'])
                }
        
        def development_iter(self):
            for sample in self.val_samples:
                yield {
                    "uri": sample['file_id'],
                    "audio": sample['audio_path'],
                    "annotation": self._load_annotation(sample['rttm_path'])
                }
    
    return ManualProtocol()


def train_with_custom_loop(device: str = "cuda"):
    """
    自訂訓練迴圈（更靈活的控制）
    """
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from pyannote.audio import Model
    from pyannote.audio.tasks import Segmentation
    from tqdm import tqdm
    
    print("=" * 60)
    print("Step 2: 訓練模型 (自訂迴圈)")
    print("=" * 60)
    
    # 設定資料庫
    setup_pyannote_database()
    
    # 檢查裝置
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    
    print(f"使用裝置: {device}")
    
    # 載入模型
    model = Model.from_pretrained(
        PRETRAINED_SEGMENTATION_MODEL,
        use_auth_token=HF_TOKEN
    )
    model = model.to(device)
    
    # 凍結部分層（只訓練最後幾層）
    # 這可以加速訓練並防止過擬合
    freeze_layers = True
    if freeze_layers:
        for name, param in model.named_parameters():
            if "classifier" not in name and "lstm" not in name:
                param.requires_grad = False
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"可訓練參數: {trainable_params:,} / {total_params:,}")
    
    # 設定優化器
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    # 損失函數
    criterion = nn.BCEWithLogitsLoss()
    
    # 載入資料（這裡使用簡化版）
    print("\n注意：此訓練方式需要自行實作 DataLoader")
    print("建議使用 train_model_simple() 函數進行訓練")
    
    return model


def main():
    """主程式"""
    # 建立目錄
    create_directories()
    
    # 檢查資料是否已準備
    samples_json = PROCESSED_DATA_DIR / "samples.json"
    if not samples_json.exists():
        print("❌ 資料尚未準備，請先執行 01_prepare_dataset.py")
        return
    
    # 開始訓練
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        model = train_model_simple(device)
        
        print("\n" + "=" * 60)
        print("✅ 訓練完成!")
        print(f"  模型儲存位置: {MODEL_DIR / 'finetuned_segmentation'}")
        print(f"  TensorBoard 日誌: {LOG_DIR}")
        print("\n查看訓練曲線:")
        print(f"  tensorboard --logdir={LOG_DIR}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 訓練失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
