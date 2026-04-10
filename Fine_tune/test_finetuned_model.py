"""
測試 Fine-tuned 模型
==================
快速測試 fine-tuned pyannote 模型是否正常工作
"""

import torch
from pyannote.audio import Model
from pathlib import Path

# 設定路徑
MODEL_PATH = Path(__file__).parent / "models" / "finetuned_segmentation" / "best_model.pt"

print("=" * 60)
print("測試 Fine-tuned 模型")
print("=" * 60)

# 檢查模型文件是否存在
if not MODEL_PATH.exists():
    print(f"❌ 模型文件不存在: {MODEL_PATH}")
    exit(1)

print(f"✓ 找到模型文件: {MODEL_PATH}")

# 載入模型
try:
    model = Model.from_pretrained("pyannote/segmentation-3.0")
    print("✓ 成功載入預訓練模型架構")
    
    # 載入 fine-tuned 權重
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    
    # 從 checkpoint 中取出實際的模型權重
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"✓ 從 checkpoint 載入模型權重 (Epoch {checkpoint.get('epoch', '?')}, Val Loss: {checkpoint.get('val_loss', '?'):.4f})")
    else:
        state_dict = checkpoint
        print("✓ 直接載入模型權重")
    
    model.load_state_dict(state_dict)
    print("✓ 成功載入 fine-tuned 權重")
    
    # 測試模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    # 創建測試輸入
    test_input = torch.randn(1, 1, 16000).to(device)  # 1 秒音訊
    
    with torch.no_grad():
        output = model(test_input)
    
    print(f"✓ 模型輸出形狀: {output.shape}")
    print("\n✅ Fine-tuned 模型運行正常！")
    print("\n模型資訊:")
    print(f"  - 裝置: {device}")
    print(f"  - 參數數量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - 可訓練參數: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
except Exception as e:
    print(f"❌ 載入模型失敗: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("測試完成！")
print("=" * 60)
