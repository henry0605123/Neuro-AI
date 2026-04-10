# -*- coding: utf-8 -*-
"""
論文復現：Exploring Speech Foundation Models for Speaker Diarization in Child-Adult Dyadic Interactions
主程式：整合訓練、評估、推論的統一入口

架構：Whisper Encoder（凍結）+ Weighted Average + 3×Conv1D Head + Classifier（與 Fine_tune_LSTM 的 Bi-LSTM 版為平行實作）

作者：AI Engineering Team
日期：2026-01-19
修訂：2026-03-24 — pipeline 與 Fine_tune_LSTM/main.py 對齊（子命令、參數風格、epochs 未指定時沿用 config.py）

使用方式：
=========
1. 訓練模型：
   python main.py train --model-size medium --epochs 15

2. 評估模型：
   python main.py evaluate --model-path models/whisper_medium_diarization_best.pt

3. 推論單一音檔：
   python main.py infer --model-path models/whisper_medium_diarization_best.pt --audio-path path/to/audio.mp3

4. 完整 Pipeline（Step 1 訓練 → Step 2 評估 → Step 3 ANALYSIS_REPORT.md）：
   python main.py pipeline --model-size medium --epochs 15
   # 未加 --epochs 時，訓練週期以 config.py 的 epochs 為準（與 Fine_tune_LSTM 相同）
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np


def setup_environment():
    """設定運行環境"""
    torch.manual_seed(42)
    np.random.seed(42)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"✅ CUDA 可用: {torch.cuda.get_device_name(0)}")
        print(f"   GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️ CUDA 不可用，將使用 CPU 訓練（速度會很慢）")

    if sys.platform == "win32":
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def train_command(args):
    """執行訓練命令"""
    from config import Config
    from train import train_model

    print("\n" + "=" * 60)
    print("🚀 開始訓練 Whisper + Conv1D 語者分離模型（論文復現）")
    print("=" * 60)

    cfg_ref = Config()
    config = Config(
        model_size=args.model_size,
        epochs=args.epochs if args.epochs is not None else cfg_ref.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        window_size=args.window_size,
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
    print("📊 開始評估 Conv1D 模型（論文復現）")
    print("=" * 60)

    config = Config(
        model_size=args.model_size,
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
        apply_median_filter=args.use_median,
    )

    print("\n✅ 評估完成！")

    return results


def infer_command(args):
    """執行推論命令"""
    from config import Config
    from model import load_model
    from evaluate import inference_on_audio, predictions_to_srt

    print("\n" + "=" * 60)
    print("🔍 開始推論 (Conv1D)")
    print("=" * 60)

    config = Config(
        model_size=args.model_size,
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
        rttm_path=args.rttm_path,
        save_dir=args.output_dir or config.output_dir,
    )

    if args.output_srt:
        output_path = args.output_srt
    else:
        output_path = Path(args.audio_path).with_suffix(".predicted.srt")

    predictions_to_srt(
        results["predictions"],
        config.step_size_ms,
        str(output_path),
    )

    print("\n✅ 推論完成！")

    return results


def pipeline_command(args):
    """執行完整 Pipeline：訓練 → 評估 → 生成 ANALYSIS_REPORT.md"""
    from config import Config
    from train import train_model
    from dataset import create_test_dataloader
    from evaluate import Evaluator

    print("\n" + "=" * 60)
    print("🔄 執行完整 Pipeline (Whisper + Conv1D · 論文復現)")
    print("=" * 60)

    cfg_ref = Config()
    config = Config(
        model_size=args.model_size,
        epochs=args.epochs if args.epochs is not None else cfg_ref.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        window_size=args.window_size,
        median_filter_size=args.median_size,
    )

    # Step 1: 訓練
    print("\n📌 Step 1: 訓練模型")
    print("-" * 40)
    model, train_summary = train_model(config)

    # Step 2: 評估
    print("\n📌 Step 2: 評估模型")
    print("-" * 40)

    test_loader = create_test_dataloader(config)

    evaluator = Evaluator(config)
    eval_results = evaluator.evaluate(
        model=model,
        data_loader=test_loader,
        save_dir=config.output_dir,
        apply_median_filter=args.use_median,
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
    """生成結果分析報告（論文 Conv1D 版本）"""
    from datetime import datetime

    report_path = Path(config.output_dir) / "ANALYSIS_REPORT.md"

    der = eval_results.get("der")

    test_data_line = (
        f"- **測試資料目錄**: `{config.test_data_dir}`（多 clip，RTTM/SRT 與訓練同結構）"
        if getattr(config, "test_data_dir", None)
        else f"- **測試音檔**: {config.test_audio_path}"
    )

    report = f"""# 語者分離模型實驗報告 (Whisper + Conv1D · 論文復現)

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

### 1.2 Conv Head 架構設定
| 參數 | 值 |
|------|------|
| Conv 通道數 | {config.conv_channels} |
| Dropout | {config.dropout} |
| **中間層** | **3 層 Conv1D + ReLU + Dropout（論文設定）** |

### 1.3 資料集
- **訓練資料**: {config.train_data_dir}
{test_data_line}
- **驗證集比例**: {config.val_split_ratio * 100:.0f}%

### 1.4 硬體配置
- **GPU**: NVIDIA RTX 5070 Ti（或本機實際 GPU）
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

### 3.2 與論文 Baseline 的比較

| 方法 | DER |
|------|-----|
| VBx (論文 Baseline) | ~25.8% |
| PyAnnote (論文 Baseline) | ~26.0% |
| WSP-Medium (論文) | 15.6% |
| **本實驗結果 (3 分類)** | **{der.der:.1f}%** |

---

## 4. 混淆矩陣與常見錯誤

混淆矩陣圖已儲存至: `{config.output_dir}/confusion_matrix.png`

- **Child / Adult 混淆**: 語調或發聲特徵重疊時較易發生  
- **邊界誤差**: 語音起迄處 frame 標記易有偏差  
- **短語音**: 可能被誤判為 Silence  

---

## 5. 視覺化結果

1. **訓練曲線**: `{config.log_dir}/training_curves.png`
2. **混淆矩陣**: `{config.output_dir}/confusion_matrix.png`
3. **甘特圖**: `{config.output_dir}/gantt_chart.png`
4. **多窗口比較圖**: `{config.output_dir}/gantt_comparison.png`

---

## 6. 結論與架構摘要

本實驗復現論文方法：Whisper-{config.model_size} 作為特徵提取器，下游為 **Conv1D Head**（與 Fine_tune_LSTM 專案中的 **Bi-LSTM** 替換版為不同架構選項）。

```
Whisper Encoder (凍結)
    ↓
Weighted Average Layer
    ↓
3 × Conv1D + ReLU + Dropout
    ↓
Conv1D Classifier (→ 3 類別)
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

"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"報告已生成: {report_path}")


def compare_models_command(args):
    """比較不同 Whisper 尺寸的 Conv1D 模型效能"""
    from config import Config
    from model import load_model
    from dataset import create_test_dataloader
    from evaluate import Evaluator

    print("\n" + "=" * 60)
    print("📊 比較多個 Whisper + Conv1D 模型")
    print("=" * 60)

    results = []

    for model_size in args.model_sizes:
        print(f"\n評估 Whisper-{model_size} + Conv1D...")

        config = Config(model_size=model_size)
        model_path = Path(config.model_save_dir) / f"whisper_{model_size}_diarization_best.pt"

        if not model_path.exists():
            print(f"  ⚠️ 模型不存在: {model_path}")
            continue

        model = load_model(config, str(model_path))
        test_loader = create_test_dataloader(config)

        evaluator = Evaluator(config)
        eval_results = evaluator.evaluate(
            model=model,
            data_loader=test_loader,
            save_dir=None,
        )

        results.append(
            {
                "model_size": model_size,
                "der": eval_results["der"].der,
                "speaker_confusion": eval_results["der"].speaker_confusion,
            }
        )

    print("\n" + "=" * 60)
    print("模型比較結果 (Conv1D)")
    print("=" * 60)
    print(f"{'模型':<15} {'DER':<10} {'Speaker Confusion':<20}")
    print("-" * 45)
    for r in results:
        print(
            f"Whisper-{r['model_size']:<7} {r['der']:.2f}%{'':<5} {r['speaker_confusion']:.2f}%"
        )

    return results


def main():
    """主程式入口"""
    parser = argparse.ArgumentParser(
        description="Whisper + Conv1D 語者分離 — 論文復現（Interspeech 2024）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例：
  # 訓練 medium 模型
  python main.py train --model-size medium --epochs 15

  # 評估模型
  python main.py evaluate --model-path models/whisper_medium_diarization_best.pt

  # 推論單一音檔
  python main.py infer --model-path models/whisper_medium_diarization_best.pt --audio-path audio.mp3

  # 完整 pipeline（與 Fine_tune_LSTM 相同：train → evaluate → ANALYSIS_REPORT.md）
  python main.py pipeline --model-size medium --epochs 15
  python main.py pipeline --model-size medium
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # ===== 訓練命令 =====
    train_parser = subparsers.add_parser("train", help="訓練模型")
    train_parser.add_argument(
        "--model-size",
        type=str,
        default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper 模型大小",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="訓練 epochs（未指定時沿用 config.py 的 epochs）",
    )
    train_parser.add_argument("--batch-size", type=int, default=8, help="批次大小")
    train_parser.add_argument("--lr", type=float, default=5e-4, help="學習率")
    train_parser.add_argument(
        "--window-size", type=float, default=20.0, help="音訊窗口大小（秒）"
    )

    # ===== 評估命令 =====
    eval_parser = subparsers.add_parser("evaluate", help="評估模型")
    eval_parser.add_argument(
        "--model-path", type=str, required=True, help="模型權重路徑"
    )
    eval_parser.add_argument(
        "--model-size",
        type=str,
        default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper 模型大小",
    )
    median_group = eval_parser.add_mutually_exclusive_group()
    median_group.add_argument(
        "--use-median",
        dest="use_median",
        action="store_true",
        help="啟用中值濾波後處理",
    )
    median_group.add_argument(
        "--no-median",
        dest="use_median",
        action="store_false",
        help="停用中值濾波後處理",
    )
    eval_parser.set_defaults(use_median=False)
    eval_parser.add_argument(
        "--median-size",
        type=int,
        default=5,
        help="中值濾波器核大小（必須是奇數，預設=5）",
    )

    # ===== 推論命令 =====
    infer_parser = subparsers.add_parser("infer", help="推論單一音檔")
    infer_parser.add_argument(
        "--model-path", type=str, required=True, help="模型權重路徑"
    )
    infer_parser.add_argument(
        "--audio-path", type=str, required=True, help="音訊檔案路徑"
    )
    infer_parser.add_argument(
        "--srt-path", type=str, default=None, help="SRT 標註檔案路徑（可選，用於對照評估）"
    )
    infer_parser.add_argument(
        "--rttm-path",
        type=str,
        default=None,
        help="RTTM 標註檔路徑（可選；與 SRT 皆提供時優先 RTTM）",
    )
    infer_parser.add_argument(
        "--output-srt", type=str, default=None, help="輸出 SRT 路徑"
    )
    infer_parser.add_argument("--output-dir", type=str, default=None, help="輸出目錄")
    infer_parser.add_argument(
        "--model-size",
        type=str,
        default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper 模型大小",
    )
    median_group_infer = infer_parser.add_mutually_exclusive_group()
    median_group_infer.add_argument(
        "--use-median",
        dest="use_median",
        action="store_true",
        help="啟用中值濾波後處理",
    )
    median_group_infer.add_argument(
        "--no-median",
        dest="use_median",
        action="store_false",
        help="停用中值濾波後處理",
    )
    infer_parser.set_defaults(use_median=False)
    infer_parser.add_argument(
        "--median-size",
        type=int,
        default=5,
        help="中值濾波器核大小（必須是奇數，預設=5）",
    )

    # ===== Pipeline 命令（與 Fine_tune_LSTM/main.py 同層級選項）=====
    pipeline_parser = subparsers.add_parser(
        "pipeline", help="執行完整 pipeline（訓練+評估+報告，與 Fine_tune_LSTM 相同步驟）"
    )
    pipeline_parser.add_argument(
        "--model-size",
        type=str,
        default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper 模型大小",
    )
    pipeline_parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="訓練 epochs（未指定時沿用 config.py 的 epochs）",
    )
    pipeline_parser.add_argument("--batch-size", type=int, default=8, help="批次大小")
    pipeline_parser.add_argument("--lr", type=float, default=5e-4, help="學習率")
    pipeline_parser.add_argument(
        "--window-size", type=float, default=20.0, help="音訊窗口大小（秒）"
    )
    median_group_pipe = pipeline_parser.add_mutually_exclusive_group()
    median_group_pipe.add_argument(
        "--use-median",
        dest="use_median",
        action="store_true",
        help="啟用中值濾波後處理",
    )
    median_group_pipe.add_argument(
        "--no-median",
        dest="use_median",
        action="store_false",
        help="停用中值濾波後處理",
    )
    pipeline_parser.set_defaults(use_median=False)
    pipeline_parser.add_argument(
        "--median-size",
        type=int,
        default=5,
        help="中值濾波器核大小（必須是奇數，預設=5）",
    )

    # ===== 比較命令 =====
    compare_parser = subparsers.add_parser("compare", help="比較多個模型")
    compare_parser.add_argument(
        "--model-sizes",
        type=str,
        nargs="+",
        default=["tiny", "small", "medium"],
        help="要比較的模型大小列表",
    )

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
