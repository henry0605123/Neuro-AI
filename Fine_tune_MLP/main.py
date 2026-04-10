# -*- coding: utf-8 -*-
"""
Whisper + MLP + Median Filter 語者分離模型
主程式：整合訓練、評估、推論的統一入口

架構：Whisper Encoder + Weighted Average + MLP + Median Filter + Classifier
日期：2026-01-26

使用方式：
=========
1. 訓練模型：
   python main.py train --model-size medium --epochs 15

2. 評估模型：
   python main.py evaluate --model-path models/whisper_medium_mlp_best.pt

3. 推論單一音檔：
   python main.py infer --model-path models/whisper_medium_mlp_best.pt --audio-path path/to/audio.mp3

4. 完整 Pipeline（訓練 + 評估）：
   python main.py pipeline --model-size medium --epochs 15
"""

import os
import sys
import argparse
import numpy as np
import torch
from datetime import datetime
from pathlib import Path


def setup_environment():
    """設定運行環境"""
    torch.manual_seed(42)
    np.random.seed(42)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.benchmark = True
        print(f"🚀 GPU 可用: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA 版本: {torch.version.cuda}")
    else:
        print("⚠️ GPU 不可用，將使用 CPU 運算")
    
    if sys.platform == 'win32':
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def train_command(args):
    """執行訓練命令"""
    from config import Config
    from train import train_model
    
    print("\n" + "=" * 60)
    print("🚀 開始訓練 Whisper + MLP + Median Filter 語者分離模型")
    print("=" * 60)
    
    # 解析 MLP 隱藏層維度
    mlp_hidden_dims = [int(x) for x in args.mlp_hidden.split(",")]
    
    config = Config(
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        window_size=args.window_size,
        mlp_hidden_dims=mlp_hidden_dims,
        median_filter_size=args.median_size,
    )
    
    model, summary = train_model(config)
    
    print("\n✅ 訓練完成！")
    
    return model, summary


def evaluate_command(args):
    """執行評估命令"""
    from config import Config
    from model import load_model
    from dataset import create_test_dataloader
    from evaluate import Evaluator
    
    print("\n" + "=" * 60)
    print("📊 開始評估 MLP + Median Filter 模型")
    print("=" * 60)
    
    # 解析 MLP 隱藏層維度
    mlp_hidden_dims = [int(x) for x in args.mlp_hidden.split(",")]
    
    config = Config(
        model_size=args.model_size,
        mlp_hidden_dims=mlp_hidden_dims,
        median_filter_size=args.median_size,
    )
    
    if not os.path.exists(args.model_path):
        print(f"❌ 模型檔案不存在: {args.model_path}")
        return None
    
    model = load_model(config, args.model_path)
    
    test_loader = create_test_dataloader(config)
    
    evaluator = Evaluator(config)
    results = evaluator.evaluate(
        model=model,
        data_loader=test_loader,
        save_dir=config.output_dir,
        apply_median_filter=args.use_median  # 評估時可選擇是否使用中值濾波
    )
    
    print("\n✅ 評估完成！")
    
    return results


def infer_command(args):
    """執行推論命令"""
    from config import Config
    from model import load_model
    from evaluate import inference_on_audio, predictions_to_srt
    
    print("\n" + "=" * 60)
    print("🔍 開始推論 (MLP + Median Filter)")
    print("=" * 60)
    
    # 解析 MLP 隱藏層維度
    mlp_hidden_dims = [int(x) for x in args.mlp_hidden.split(",")]
    
    config = Config(
        model_size=args.model_size,
        mlp_hidden_dims=mlp_hidden_dims,
        median_filter_size=args.median_size,
    )
    
    if not os.path.exists(args.model_path):
        print(f"❌ 模型檔案不存在: {args.model_path}")
        return None
    
    model = load_model(config, args.model_path)
    
    if not os.path.exists(args.audio_path):
        print(f"❌ 音訊檔案不存在: {args.audio_path}")
        return None
    
    results = inference_on_audio(
        model=model,
        audio_path=args.audio_path,
        config=config,
        srt_path=args.srt_path,
        save_dir=args.output_dir or config.output_dir,
        apply_median_filter=args.use_median  # 推論時可選擇是否使用中值濾波
    )
    
    if args.output_srt:
        output_path = Path(args.output_srt)
    else:
        output_path = Path(args.audio_path).with_suffix(".srt")
    
    predictions_to_srt(
        results["predictions"],
        config.step_size_ms,
        str(output_path)
    )
    
    print("\n✅ 推論完成！")
    
    return results


def pipeline_command(args):
    """執行完整 Pipeline（訓練 + 評估）"""
    from config import Config
    from train import train_model
    from dataset import create_test_dataloader
    from evaluate import Evaluator
    
    print("\n" + "=" * 60)
    print("🔄 執行完整 Pipeline (Whisper + MLP + Median Filter)")
    print("=" * 60)
    
    # 解析 MLP 隱藏層維度
    mlp_hidden_dims = [int(x) for x in args.mlp_hidden.split(",")]
    
    config = Config(
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        window_size=args.window_size,
        mlp_hidden_dims=mlp_hidden_dims,
        median_filter_size=args.median_size,
    )
    
    # Step 1: 訓練
    print("\n📌 Step 1: 訓練模型")
    print("-" * 40)
    model, train_summary = train_model(config)
    
    # Step 2: 評估
    use_median = getattr(args, 'use_median', True)
    median_status = "使用" if use_median else "不使用"
    print(f"\n📌 Step 2: 評估模型（{median_status} Median Filter）")
    print("-" * 40)
    
    test_loader = create_test_dataloader(config)
    
    evaluator = Evaluator(config)
    eval_results = evaluator.evaluate(
        model=model,
        data_loader=test_loader,
        save_dir=config.output_dir,
        apply_median_filter=use_median
    )
    
    # Step 3: 生成報告
    print("\n📌 Step 3: 生成報告")
    print("-" * 40)
    
    generate_report(config, train_summary, eval_results)
    
    print("\n✅ Pipeline 完成！")
    
    return {
        "training": train_summary,
        "evaluation": eval_results,
    }


def generate_report(config, train_summary, eval_results):
    """生成結果分析報告 (MLP + Median Filter 版本)"""
    from datetime import datetime
    
    report_path = Path(config.output_dir) / "ANALYSIS_REPORT.md"
    
    der = eval_results.get("der")
    
    report = f"""# 語者分離模型實驗報告 (Whisper + MLP + Median Filter)

**生成時間**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 1. 實驗設置摘要

### 1.1 模型配置
| 參數 | 值 |
|------|------|
| Whisper 模型大小 | {config.model_size} |
| 音訊窗口大小 | {config.window_size} 秒 |
| Step Size | {config.step_size_ms} ms |
| 批次大小 | {config.batch_size} |
| 學習率 | {config.learning_rate} |
| 權重衰減 | {config.weight_decay} |
| 最大訓練 Epochs | {config.epochs} |
| **分類類別數** | **3 (Child, Adult, Silence)** |

### 1.2 MLP + Median Filter 架構設定
| 參數 | 值 |
|------|------|
| MLP 隱藏層維度 | {config.mlp_hidden_dims} |
| MLP 層數 | {config.mlp_num_layers} |
| Dropout | {config.dropout} |
| Median Filter 核心大小 | {config.median_filter_size} |
| **中間層架構** | **MLP + Median Filter（替換 Bi-LSTM）** |

### 1.3 資料集
- **訓練資料**: {config.train_data_dir}
- **測試資料**: {config.test_audio_path}
- **驗證集比例**: {config.val_split_ratio * 100:.0f}%

### 1.4 硬體配置
- **GPU**: NVIDIA RTX 5070 Ti
- **混合精度訓練 (AMP)**: {'啟用' if config.use_amp else '停用'}
- **cuDNN Benchmark**: {'啟用' if config.cudnn_benchmark else '停用'}

---

## 2. 訓練結果分析

### 2.1 訓練摘要
| 指標 | 值 |
|------|------|
| 總訓練 Epochs | {train_summary.get('total_epochs', 'N/A')} |
| 最佳 Epoch | {train_summary.get('best_epoch', 'N/A')} |
| 最佳驗證損失 | {train_summary.get('best_val_loss', 0):.4f} |
| 最佳驗證準確率 | {train_summary.get('best_val_acc', 0):.2f}% |
| 總訓練時間 | {train_summary.get('total_time', 0)/60:.1f} 分鐘 |
| 平均每 Epoch 時間 | {train_summary.get('avg_epoch_time', 0):.1f} 秒 |

### 2.2 Loss 收斂情形
訓練曲線圖已儲存至: `{config.log_dir}/training_curves.png`

---

## 3. 評估結果

### 3.1 Diarization Error Rate (DER)

| 指標 | 值 |
|------|------|
| **DER** | **{der.der:.2f}%** |
| False Alarm Rate | {der.false_alarm:.2f}% |
| Missed Detection Rate | {der.missed_detection:.2f}% |
| Speaker Confusion Rate | {der.speaker_confusion:.2f}% |

### 3.2 架構比較

| 架構 | 描述 |
|------|------|
| 原始 (Conv1D) | 3 層 1D 卷積 + ReLU + Dropout |
| Bi-LSTM | 2 層雙向 LSTM + LayerNorm + Dropout |
| **MLP + Median Filter** | **多層 MLP + GELU + Median Filter 後處理** |

---

## 4. MLP + Median Filter 分析

### 4.1 架構設計理念

**MLP 的優勢**:
1. **計算效率**: 完全平行化，訓練和推論速度快
2. **簡單架構**: 較少的超參數，容易調整
3. **記憶體友好**: 不需要儲存隱藏狀態

**Median Filter 的作用**:
1. **平滑預測**: 減少時間序列中的抖動
2. **去除噪音**: 過濾掉短暫的錯誤預測
3. **後處理增強**: 不影響訓練，只在推論時使用

### 4.2 與其他架構對比

| 架構 | 優點 | 缺點 |
|------|------|------|
| Conv1D | 局部特徵提取佳 | 感受野有限 |
| Bi-LSTM | 長距離依賴 | 訓練較慢 |
| **MLP + Median Filter** | **快速、平滑輸出** | **無時序建模能力** |

---

## 5. Median Filter 效果分析

Median Filter 核心大小 = {config.median_filter_size}

這意味著每個預測會考慮前後共 {config.median_filter_size} 個 frame，
相當於 {config.median_filter_size * config.step_size_ms} ms 的時間範圍。

---

## 6. 視覺化結果

以下視覺化圖表已生成：

1. **訓練曲線**: `{config.log_dir}/training_curves.png`
2. **混淆矩陣**: `{config.output_dir}/confusion_matrix.png`
3. **甘特圖**: `{config.output_dir}/gantt_chart.png`
4. **多窗口比較圖**: `{config.output_dir}/gantt_comparison.png`

---

## 7. 結論

本實驗使用 Whisper-{config.model_size} 作為特徵提取器，
將中間層替換為 MLP + Median Filter 架構。

### 架構摘要:
```
Whisper Encoder (凍結)
    ↓
Weighted Average Layer
    ↓
MLP (隱藏層: {config.mlp_hidden_dims}) + GELU + Dropout
    ↓
Conv1D Classifier (→ 3 類別)
    ↓
Median Filter (kernel={config.median_filter_size}, 僅推論時使用)
```

---

## 附錄

### A. 類別定義 (3 分類)
- **0 (Child)**: 小孩語音
- **1 (Adult)**: 測試者/成人語音
- **2 (Silence)**: 靜音/背景噪音

### B. 評估指標說明
- **DER**: 語者分離總錯誤率
- **False Alarm**: 將非語音誤判為語音的比例
- **Missed Detection**: 將語音誤判為非語音的比例
- **Speaker Confusion**: 語音區段語者判斷錯誤的比例

### C. Median Filter 說明
Median Filter 是一種非線性濾波器，在時間序列上取滑動窗口內的中值。
這可以有效去除短暫的抖動和噪音，使預測結果更加平滑連貫。

"""
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"報告已生成: {report_path}")


def compare_models_command(args):
    """比較不同模型的效能"""
    from config import Config
    from model import load_model
    from dataset import create_test_dataloader
    from evaluate import Evaluator
    
    print("\n" + "=" * 60)
    print("📊 比較多個 MLP 模型")
    print("=" * 60)
    
    # 解析 MLP 隱藏層維度
    mlp_hidden_dims = [int(x) for x in args.mlp_hidden.split(",")]
    
    results = []
    
    for model_size in args.model_sizes:
        print(f"\n評估 Whisper-{model_size} + MLP...")
        
        config = Config(
            model_size=model_size,
            mlp_hidden_dims=mlp_hidden_dims,
            median_filter_size=args.median_size,
        )
        
        model_path = config.model_weights_path
        
        if not os.path.exists(model_path):
            print(f"  ⚠️ 模型檔案不存在: {model_path}")
            continue
        
        model = load_model(config, model_path)
        test_loader = create_test_dataloader(config)
        
        evaluator = Evaluator(config)
        eval_results = evaluator.evaluate(
            model=model,
            data_loader=test_loader,
            save_dir=None,  # 不儲存視覺化
            apply_median_filter=True
        )
        
        results.append({
            "model_size": model_size,
            "der": eval_results["der"].der,
            "confusion": eval_results["der"].speaker_confusion,
        })
    
    print("\n" + "=" * 60)
    print("模型比較結果 (MLP + Median Filter)")
    print("=" * 60)
    print(f"{'模型':<15} {'DER':<10} {'Speaker Confusion':<20}")
    print("-" * 45)
    for r in results:
        print(f"{r['model_size']:<15} {r['der']:<10.2f}% {r['confusion']:<20.2f}%")
    
    return results


def main():
    """主程式入口"""
    parser = argparse.ArgumentParser(
        description="Whisper + MLP + Median Filter 語者分離模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例：
  # 訓練 medium 模型
  python main.py train --model-size medium --epochs 15

  # 評估模型
  python main.py evaluate --model-path models/whisper_medium_mlp_best.pt

  # 推論單一音檔
  python main.py infer --model-path models/whisper_medium_mlp_best.pt --audio-path audio.mp3

  # 執行完整 pipeline
  python main.py pipeline --model-size medium
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # ===== 通用 MLP 參數 =====
    def add_mlp_args(p):
        p.add_argument(
            "--mlp-hidden", type=str, default="512,256",
            help="MLP 隱藏層維度，用逗號分隔（如 512,256）"
        )
        p.add_argument(
            "--median-size", type=int, default=5,
            help="Median Filter 核心大小（必須為奇數）"
        )
    
    # ===== 訓練命令 =====
    train_parser = subparsers.add_parser("train", help="訓練模型")
    train_parser.add_argument(
        "--model-size", type=str, default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper 模型大小"
    )
    train_parser.add_argument("--epochs", type=int, default=15, help="訓練 epochs")
    train_parser.add_argument("--batch-size", type=int, default=8, help="批次大小")
    train_parser.add_argument("--lr", type=float, default=5e-4, help="學習率")
    train_parser.add_argument("--window-size", type=float, default=20.0, help="音訊窗口大小（秒）")
    add_mlp_args(train_parser)
    
    # ===== 評估命令 =====
    eval_parser = subparsers.add_parser("evaluate", help="評估模型")
    eval_parser.add_argument("--model-path", type=str, required=True, help="模型權重路徑")
    eval_parser.add_argument(
        "--model-size", type=str, default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper 模型大小"
    )
    eval_parser.add_argument(
        "--use-median", action="store_true", default=True,
        help="是否使用 Median Filter（預設啟用）"
    )
    eval_parser.add_argument(
        "--no-median", action="store_false", dest="use_median",
        help="不使用 Median Filter"
    )
    add_mlp_args(eval_parser)
    
    # ===== 推論命令 =====
    infer_parser = subparsers.add_parser("infer", help="推論單一音檔")
    infer_parser.add_argument("--model-path", type=str, required=True, help="模型權重路徑")
    infer_parser.add_argument("--audio-path", type=str, required=True, help="音訊檔案路徑")
    infer_parser.add_argument("--srt-path", type=str, default=None, help="標註檔案路徑（用於評估）")
    infer_parser.add_argument("--output-srt", type=str, default=None, help="輸出 SRT 檔案路徑")
    infer_parser.add_argument("--output-dir", type=str, default=None, help="輸出目錄")
    infer_parser.add_argument(
        "--model-size", type=str, default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper 模型大小"
    )
    infer_parser.add_argument(
        "--use-median", action="store_true", default=True,
        help="是否使用 Median Filter（預設啟用）"
    )
    infer_parser.add_argument(
        "--no-median", action="store_false", dest="use_median",
        help="不使用 Median Filter"
    )
    add_mlp_args(infer_parser)
    
    # ===== Pipeline 命令 =====
    pipeline_parser = subparsers.add_parser("pipeline", help="完整 Pipeline（訓練 + 評估）")
    pipeline_parser.add_argument(
        "--model-size", type=str, default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper 模型大小"
    )
    pipeline_parser.add_argument("--epochs", type=int, default=15, help="訓練 epochs")
    pipeline_parser.add_argument("--batch-size", type=int, default=8, help="批次大小")
    pipeline_parser.add_argument("--lr", type=float, default=5e-4, help="學習率")
    pipeline_parser.add_argument("--window-size", type=float, default=20.0, help="音訊窗口大小（秒）")
    pipeline_parser.add_argument(
        "--use-median", action="store_true", default=True,
        help="評估時是否使用 Median Filter（預設啟用）"
    )
    pipeline_parser.add_argument(
        "--no-median", action="store_false", dest="use_median",
        help="評估時不使用 Median Filter"
    )
    add_mlp_args(pipeline_parser)
    
    # ===== 比較命令 =====
    compare_parser = subparsers.add_parser("compare", help="比較不同模型的效能")
    compare_parser.add_argument(
        "--model-sizes", type=str, nargs="+",
        default=["small", "medium", "large"],
        help="要比較的模型大小列表"
    )
    add_mlp_args(compare_parser)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    setup_environment()
    
    if args.command == "train":
        train_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "infer":
        infer_command(args)
    elif args.command == "pipeline":
        pipeline_command(args)
    elif args.command == "compare":
        compare_models_command(args)


if __name__ == "__main__":
    main()
