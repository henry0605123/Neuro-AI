# -*- coding: utf-8 -*-
"""
快速測試腳本：驗證所有模組是否正常運作

使用方式：
    python test_pipeline.py
"""

import os
import sys
from pathlib import Path

def test_imports():
    """測試所有模組是否可以正常匯入"""
    print("=" * 60)
    print("測試模組匯入...")
    print("=" * 60)
    
    try:
        print("  ✓ 匯入 config...")
        from config import Config, get_default_config
        
        print("  ✓ 匯入 dataset...")
        from dataset import (
            SRTParser, FrameLabelGenerator, AudioProcessor,
            SpeakerDiarizationDataset, discover_training_samples
        )
        
        print("  ✓ 匯入 model...")
        from model import (
            WeightedAverageLayer, ConvHead, ClassifierHead,
            WhisperDiarizationModelV2, create_model
        )
        
        print("  ✓ 匯入 train...")
        from train import EarlyStopping, TrainingLogger, Trainer
        
        print("  ✓ 匯入 evaluate...")
        from evaluate import (
            DERCalculator, ConfusionMatrixVisualizer,
            GanttChartVisualizer, Evaluator
        )
        
        print("  ✓ 匯入 main...")
        from main import setup_environment
        
        print("\n✅ 所有模組匯入成功！\n")
        return True
        
    except ImportError as e:
        print(f"\n❌ 匯入失敗: {e}")
        return False


def test_config():
    """測試配置模組"""
    print("=" * 60)
    print("測試配置模組...")
    print("=" * 60)
    
    from config import Config
    
    config = Config()
    config.print_config()
    
    print(f"\n  Whisper 模型名稱: {config.whisper_model_name}")
    print(f"  窗口取樣點數: {config.window_samples}")
    print(f"  每窗口 Frame 數: {config.frames_per_window}")
    
    print("\n✅ 配置模組正常！\n")
    return True


def test_srt_parser():
    """測試 SRT 解析器"""
    print("=" * 60)
    print("測試 SRT 解析器...")
    print("=" * 60)
    
    from config import Config
    from dataset import SRTParser, FrameLabelGenerator
    
    config = Config()
    parser = SRTParser()
    
    # 尋找一個 SRT 檔案進行測試
    train_dir = Path(config.train_data_dir)
    srt_files = list(train_dir.rglob("*.srt"))
    
    if srt_files:
        test_file = srt_files[0]
        print(f"\n  測試檔案: {test_file}")
        
        segments = parser.parse_file(str(test_file))
        print(f"  解析到 {len(segments)} 個片段")
        
        if segments:
            print(f"\n  前 3 個片段:")
            for seg in segments[:3]:
                print(f"    [{seg.start_time:.2f}s - {seg.end_time:.2f}s] {seg.speaker}: {seg.text[:30]}...")
        
        print("\n✅ SRT 解析器正常！\n")
        return True
    else:
        print("  ⚠️ 找不到 SRT 檔案進行測試")
        return False


def test_audio_processor():
    """測試音訊處理器"""
    print("=" * 60)
    print("測試音訊處理器...")
    print("=" * 60)
    
    from config import Config
    from dataset import AudioProcessor
    
    config = Config()
    processor = AudioProcessor(config.sample_rate)
    
    # 尋找一個音訊檔案進行測試
    train_dir = Path(config.train_data_dir)
    audio_files = list(train_dir.rglob("*.mp3"))
    
    if audio_files:
        test_file = audio_files[0]
        print(f"\n  測試檔案: {test_file}")
        
        try:
            waveform, duration = processor.load_audio(str(test_file))
            print(f"  音訊時長: {duration:.2f} 秒")
            print(f"  波形形狀: {waveform.shape}")
            print(f"  取樣率: {config.sample_rate} Hz")
            
            # 測試切片
            slice_audio = processor.slice_audio(waveform, 0, 20.0)
            print(f"  切片形狀 (20s): {slice_audio.shape}")
            
            print("\n✅ 音訊處理器正常！\n")
            return True
            
        except Exception as e:
            print(f"  ❌ 音訊處理失敗: {e}")
            return False
    else:
        print("  ⚠️ 找不到音訊檔案進行測試")
        return False


def test_model_creation():
    """測試模型建立（使用 tiny 模型快速測試）"""
    print("=" * 60)
    print("測試模型建立 (Whisper-tiny)...")
    print("=" * 60)
    
    import torch
    from config import Config
    from model import create_model
    
    # 使用 tiny 模型進行測試
    config = Config()
    config.model_size = "tiny"
    
    print(f"\n  建立 Whisper-{config.model_size} 模型...")
    
    try:
        model = create_model(config, model_version="v2")
        
        print(f"\n  模型建立成功！")
        print(f"  裝置: {config.device}")
        
        # 測試前向傳播
        print("\n  測試前向傳播...")
        batch_size = 1
        num_samples = int(config.window_size * config.sample_rate)
        
        dummy_audio = torch.randn(batch_size, num_samples).to(config.device)
        
        with torch.no_grad():
            logits = model(dummy_audio)
        
        print(f"  輸入形狀: {dummy_audio.shape}")
        print(f"  輸出形狀: {logits.shape}")
        
        # 檢查層權重
        weights = model.get_layer_weights()
        print(f"\n  層權重數量: {len(weights)}")
        
        print("\n✅ 模型建立正常！\n")
        return True
        
    except Exception as e:
        print(f"  ❌ 模型建立失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader():
    """測試資料載入器"""
    print("=" * 60)
    print("測試資料載入器...")
    print("=" * 60)
    
    from config import Config
    from dataset import discover_training_samples, SpeakerDiarizationDataset
    from torch.utils.data import DataLoader
    
    config = Config()
    
    try:
        # 探索訓練樣本
        samples = discover_training_samples(config)
        print(f"\n  發現 {len(samples)} 個訓練樣本")
        
        if len(samples) == 0:
            print("  ⚠️ 沒有找到訓練樣本")
            return False
        
        # 建立資料集（只使用第一個樣本進行快速測試）
        print("\n  建立資料集...")
        dataset = SpeakerDiarizationDataset(
            samples=samples[:1],  # 只用一個樣本測試
            config=config,
            is_training=True,
            preload=True
        )
        
        print(f"  資料集大小: {len(dataset)}")
        
        # 建立 DataLoader
        print("\n  建立 DataLoader...")
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0  # 測試時使用單進程
        )
        
        # 取得一個批次
        print("\n  取得第一個批次...")
        batch = next(iter(loader))
        
        print(f"  音訊形狀: {batch['audio'].shape}")
        print(f"  標籤形狀: {batch['labels'].shape}")
        
        # 檢查標籤分布
        labels = batch['labels'].numpy().flatten()
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\n  標籤分布:")
        for u, c in zip(unique, counts):
            class_name = config.class_names[u]
            print(f"    {class_name}: {c}")
        
        print("\n✅ 資料載入器正常！\n")
        return True
        
    except Exception as e:
        print(f"  ❌ 資料載入失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation_tools():
    """測試評估工具"""
    print("=" * 60)
    print("測試評估工具...")
    print("=" * 60)
    
    import numpy as np
    from evaluate import DERCalculator, ConfusionMatrixVisualizer, GanttChartVisualizer
    
    # 創建假資料進行測試
    np.random.seed(42)
    num_frames = 1000
    
    # 生成隨機預測和參考標籤
    predictions = np.random.randint(0, 4, num_frames)
    references = np.random.randint(0, 4, num_frames)
    
    # 測試 DER 計算
    print("\n  測試 DER 計算...")
    der_calc = DERCalculator()
    result = der_calc.calculate(predictions, references)
    
    print(f"    DER: {result.der:.2f}%")
    print(f"    False Alarm: {result.false_alarm:.2f}%")
    print(f"    Missed Detection: {result.missed_detection:.2f}%")
    print(f"    Speaker Confusion: {result.speaker_confusion:.2f}%")
    
    # 測試混淆矩陣
    print("\n  測試混淆矩陣...")
    cm_vis = ConfusionMatrixVisualizer()
    cm = cm_vis.compute(predictions, references)
    print(f"    混淆矩陣形狀: {cm.shape}")
    
    # 測試甘特圖（不儲存圖片）
    print("\n  測試甘特圖轉換...")
    gantt_vis = GanttChartVisualizer()
    segments = gantt_vis._labels_to_segments(predictions[:100])
    print(f"    轉換為 {len(segments)} 個片段")
    
    print("\n✅ 評估工具正常！\n")
    return True


def main():
    """主測試函式"""
    import numpy as np
    
    print("\n" + "=" * 60)
    print("Whisper 語者分離 Pipeline 測試")
    print("=" * 60 + "\n")
    
    results = {}
    
    # 測試模組匯入
    results["imports"] = test_imports()
    
    if not results["imports"]:
        print("\n❌ 模組匯入失敗，請檢查依賴安裝")
        return
    
    # 測試配置
    results["config"] = test_config()
    
    # 測試 SRT 解析器
    results["srt_parser"] = test_srt_parser()
    
    # 測試音訊處理器
    results["audio_processor"] = test_audio_processor()
    
    # 測試評估工具
    results["evaluation"] = test_evaluation_tools()
    
    # 測試資料載入器（需要實際資料）
    if results["srt_parser"] and results["audio_processor"]:
        results["dataloader"] = test_dataloader()
    
    # 測試模型建立（需要 GPU 和網路下載模型）
    try:
        import torch
        if torch.cuda.is_available():
            results["model"] = test_model_creation()
        else:
            print("\n⚠️ CUDA 不可用，跳過模型測試")
            results["model"] = None
    except Exception as e:
        print(f"\n⚠️ 模型測試跳過: {e}")
        results["model"] = None
    
    # 輸出總結
    print("\n" + "=" * 60)
    print("測試總結")
    print("=" * 60)
    
    for test_name, passed in results.items():
        if passed is None:
            status = "⏭️ 跳過"
        elif passed:
            status = "✅ 通過"
        else:
            status = "❌ 失敗"
        print(f"  {test_name}: {status}")
    
    # 計算通過率
    total = len([v for v in results.values() if v is not None])
    passed = len([v for v in results.values() if v is True])
    
    print(f"\n  通過率: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 所有測試通過！可以開始訓練了。")
        print("\n執行訓練命令：")
        print("  python main.py train --model-size medium --epochs 15")
    else:
        print("\n⚠️ 部分測試失敗，請檢查錯誤訊息")


if __name__ == "__main__":
    main()
