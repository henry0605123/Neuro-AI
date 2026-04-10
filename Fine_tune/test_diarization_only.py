"""
簡化測試：使用 fine-tuned 模型進行語者分離
====================================
僅測試語者分離功能，不包含 WhisperX 轉錄
"""

import os
import torch
import warnings
from pathlib import Path
from pyannote.audio import Pipeline, Model

# 過濾警告
warnings.filterwarnings('ignore')

# 配置
HF_TOKEN = os.environ.get("HF_TOKEN")
AUDIO_FILE = r"C:\自閉症\data\ASD\20250106\20250106-蔡oo\20250106-蔡oo.mp3"
MODEL_PATH = Path(__file__).parent / "models" / "finetuned_segmentation" / "best_model.pt"

print("=" * 60)
print("簡化測試：語者分離")
print("=" * 60)

# 檢查裝置
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"裝置: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 載入原始 diarization pipeline
print("\n載入原始語者分離模型...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN
)

# 載入 fine-tuned 模型並替換
if MODEL_PATH.exists():
    print(f"\n載入 fine-tuned 模型: {MODEL_PATH}")
    
    try:
        # 載入預訓練模型架構
        segmentation_model = Model.from_pretrained(
            "pyannote/segmentation-3.0",
            use_auth_token=HF_TOKEN
        )
        
        # 載入 fine-tuned 權重
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            epoch = checkpoint.get('epoch', '?')
            val_loss = checkpoint.get('val_loss', '?')
            print(f"✓ 載入 Epoch {epoch}, Val Loss: {val_loss:.4f}")
        else:
            state_dict = checkpoint
            print("✓ 載入模型權重")
        
        # 載入權重
        segmentation_model.load_state_dict(state_dict)
        segmentation_model = segmentation_model.to(torch.device(device))
        segmentation_model.eval()
        
        # 替換 pipeline 中的模型
        pipeline._segmentation.model = segmentation_model
        
        print("✓ 已將 fine-tuned 模型整合到 pipeline")
        
    except Exception as e:
        print(f"❌ 載入模型失敗: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
else:
    print(f"❌ 找不到模型: {MODEL_PATH}")
    exit(1)

# 移動 pipeline 到裝置
pipeline = pipeline.to(torch.device(device))

# 執行語者分離
print(f"\n執行語者分離: {AUDIO_FILE}")
print("請稍候，這可能需要幾分鐘...")

try:
    diarization = pipeline(AUDIO_FILE, min_speakers=2, max_speakers=2)
    
    print("\n✅ 語者分離完成！")
    print("\n" + "=" * 60)
    print("結果摘要:")
    print("=" * 60)
    
    # 統計說話人資訊
    speakers = set()
    segments_count = 0
    
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speakers.add(speaker)
        segments_count += 1
    
    print(f"偵測到說話人數量: {len(speakers)}")
    print(f"語音段數: {segments_count}")
    
    print("\n前 10 個語音段:")
    print("-" * 60)
    for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
        if i >= 10:
            break
        print(f"{i+1}. [{turn.start:.2f}s - {turn.end:.2f}s] {speaker}")
    
    print("\n" + "=" * 60)
    print("測試完成！Fine-tuned 模型運作正常！")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ 語者分離失敗: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
