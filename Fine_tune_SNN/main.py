# -*- coding: utf-8 -*-
"""
Whisper + ReLU Student（LSTM Teacher 知識蒸餾）— 統一入口

架構：Whisper Encoder + Weighted Average（自 Teacher 複製，凍結）+ ReLU 前饋頭（Conv 或 MLP）
日期：2026-03-24

使用方式：
=========
1. 知識蒸餾訓練 Student：
   python main.py train --model-size medium --student-head conv --teacher-path path/to/whisper_medium_bilstm_best.pt

2. 評估 Student：
   python main.py evaluate --model-path models/whisper_medium_student_relu_conv_best.pt --student-head conv

3. 推論單一音檔：
   python main.py infer --model-path models/whisper_medium_student_relu_conv_best.pt --student-head conv --audio-path audio.mp3

4. 完整 Pipeline（訓練 + ANN 評估 + Conv→SNN + 報告；可 --skip-snn）：
   python main.py pipeline --model-size medium --student-head conv --teacher-path path/to/teacher.pt

5. 比較多個 Whisper 大小的 Student：
   python main.py compare --model-sizes tiny small medium --student-head conv
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def setup_environment():
    """設定運行環境（與 Fine_tune_LSTM 對齊）"""
    torch.manual_seed(42)
    np.random.seed(42)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"✅ CUDA 可用: {torch.cuda.get_device_name(0)}")
        print(f"   GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️ CUDA 不可用，將使用 CPU（速度會很慢）")

    if sys.platform == "win32":
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def _mlp_dims_from_args(args) -> list:
    return [int(x) for x in args.mlp_hidden.split(",")]


def _train_config_from_args(args) -> "Config":
    from config import Config

    return Config(
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        window_size=args.window_size,
        student_head_type=args.student_head,
        conv_channels=args.conv_channels,
        mlp_hidden_dims=_mlp_dims_from_args(args),
        teacher_checkpoint=args.teacher_path,
        lstm_hidden_size=args.lstm_hidden,
        lstm_num_layers=args.lstm_layers,
        kd_temperature=args.kd_temp,
        kd_alpha=args.kd_alpha,
    )


def _eval_config_from_args(args) -> "Config":
    from config import Config

    return Config(
        model_size=args.model_size,
        student_head_type=args.student_head,
        conv_channels=args.conv_channels,
        mlp_hidden_dims=_mlp_dims_from_args(args),
        median_filter_size=args.median_size,
    )


def train_command(args):
    """知識蒸餾訓練 ReLU Student"""
    from train import train_model

    print("\n" + "=" * 60)
    print("🚀 開始訓練 Whisper + ReLU Student（KD 自 Bi-LSTM Teacher）")
    print("=" * 60)

    config = _train_config_from_args(args)
    model, summary = train_model(config)

    print("\n✅ 訓練完成！")
    return model, summary


def evaluate_command(args):
    """評估 Student"""
    from config import Config
    from model import load_model
    from dataset import create_test_dataloader
    from evaluate import Evaluator

    print("\n" + "=" * 60)
    print("📊 開始評估 ReLU Student 模型")
    print("=" * 60)

    config = _eval_config_from_args(args)

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
    """推論單一音檔"""
    from model import load_model
    from evaluate import inference_on_audio, predictions_to_srt

    print("\n" + "=" * 60)
    print("🔍 開始推論 (ReLU Student)")
    print("=" * 60)

    config = _eval_config_from_args(args)

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
    )

    if args.output_srt:
        output_path = Path(args.output_srt)
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
    """訓練 + 評估 + 產生 ANALYSIS_REPORT（與 LSTM 相同流程）"""
    from train import train_model
    from dataset import create_test_dataloader
    from evaluate import Evaluator

    print("\n" + "=" * 60)
    print("🔄 執行完整 Pipeline (KD → Student → ANN 評估 → SNN Conv 頭 → 報告)")
    print("=" * 60)

    config = _train_config_from_args(args)

    print("\n📌 Step 1: 知識蒸餾訓練 Student")
    print("-" * 40)
    model, train_summary = train_model(config)

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

    config.snn_finetune_epochs = getattr(args, "snn_finetune_epochs", 0)
    config.snn_finetune_lr = getattr(args, "snn_ft_lr", 1e-4)

    eval_snn = None
    if not getattr(args, "skip_snn", False):
        print("\n📌 Step 3: Conv 頭 ANN→SNN（Whisper 維持 ANN）與混合模型評估")
        print("-" * 40)
        from ann_to_snn import run_ann_to_snn_pipeline_step
        from dataset import create_dataloaders

        train_ld, val_ld = None, None
        if config.snn_finetune_epochs > 0:
            train_ld, val_ld = create_dataloaders(config)

        _, eval_snn = run_ann_to_snn_pipeline_step(
            student=model,
            config=config,
            test_loader=test_loader,
            timesteps=args.snn_timesteps,
            beta=args.snn_beta,
            apply_median_filter=args.use_median,
            save_bundle=True,
            finetune_epochs=config.snn_finetune_epochs,
            finetune_lr=config.snn_finetune_lr,
            train_loader=train_ld,
            val_loader=val_ld,
        )
    else:
        print("\n📌 Step 3: 已略過 SNN 轉換（--skip-snn）")

    print("\n📌 Step 4: 生成報告")
    print("-" * 40)
    generate_report(config, train_summary, eval_results, eval_snn_results=eval_snn)

    print("\n✅ Pipeline 完成！")
    return {
        "training": train_summary,
        "evaluation": eval_results,
        "evaluation_snn": eval_snn,
    }


def generate_report(config, train_summary, eval_results, eval_snn_results=None):
    """生成結果分析報告（ReLU Student / KD 版本）"""
    report_path = Path(config.output_dir) / "ANALYSIS_REPORT.md"
    der = eval_results.get("der")
    head_desc = (
        "3 層 Conv1d (k=1) + ReLU + Dropout（論文式）"
        if config.student_head_type == "conv"
        else f"MLP + ReLU：{config.mlp_hidden_dims}"
    )

    report = f"""# 語者分離模型實驗報告 (Whisper + ReLU Student，KD)

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

### 1.2 知識蒸餾與 Student 頭
| 參數 | 值 |
|------|------|
| Teacher 權重 | `{config.teacher_checkpoint}` |
| Teacher LSTM（載入時依 checkpoint 對齊） | hidden={config.lstm_hidden_size}（單向）, layers={config.lstm_num_layers} |
| Student 頭類型 | **{config.student_head_type}** |
| Student 頭結構 | {head_desc} |
| KD 溫度 T | {config.kd_temperature} |
| KD 混合係數 α（CE 權重） | {config.kd_alpha} |
| KD 損失權重 (1-α) | {1.0 - config.kd_alpha:.4f} |

### 1.3 資料集
- **訓練資料**: {config.train_data_dir}
- **測試資料**: {config.test_audio_path}
- **驗證集比例**: {config.val_split_ratio * 100:.0f}%

### 1.4 硬體配置
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

### 3.2 架構說明

| 項目 | 描述 |
|------|------|
| **Teacher** | Bi-LSTM（凍結，僅提供軟標籤） |
| **Student** | ReLU 前饋頭（蒸餾目標） |

"""

    if eval_snn_results is not None:
        der_s = eval_snn_results.get("der")
        snn_ft_note = ""
        if getattr(config, "snn_finetune_epochs", 0) > 0:
            snn_ft_note = (
                "\n\n**SNN 頭微調**（轉換與校準後）："
                f"{config.snn_finetune_epochs} epochs，lr={config.snn_finetune_lr}；"
                "僅更新 SNN 內 Conv/LIF 路徑，Whisper 與 WeightedAvg 凍結。"
            )
        report += f"""
### 3.3 混合 SNN 頭評估（Pipeline Step 3）

Whisper Encoder + Weighted Average 仍為 **ANN**；蒸餾後之 **Conv 頭與最後 Conv1d 分類層** 以權重複製建立獨立 **snntorch Leaky LIF** 前向鏈，經校準縮放對齊 ANN 特徵域（SNN 支路與 Student 的 ANN 頭 **不共用模組**，可僅微調 SNN 頭而不改壞已蒸餾之 ANN）。

| 指標 | 值 |
|------|------|
| **DER（SNN 頭）** | **{der_s.der:.2f}%** |
| False Alarm Rate | {der_s.false_alarm:.2f}% |
| Missed Detection Rate | {der_s.missed_detection:.2f}% |
| Speaker Confusion Rate | {der_s.speaker_confusion:.2f}% |

額外圖表（若已寫入）目錄：`{config.output_dir}/snn_eval/`
{snn_ft_note}

"""

    report += f"""
---

## 4. 視覺化結果

以下視覺化圖表已生成：

1. **訓練曲線**: `{config.log_dir}/training_curves.png`
2. **混淆矩陣**: `{config.output_dir}/confusion_matrix.png`
3. **甘特圖**: `{config.output_dir}/gantt_chart.png`
4. **多窗口比較圖**: `{config.output_dir}/gantt_comparison.png`

---

## 5. 結論

本實驗以 Whisper-{config.model_size} 為特徵提取器，使用 **Bi-LSTM Teacher** 對 **ReLU Student** 進行知識蒸餾。

### 架構摘要:
```
Whisper Encoder (凍結，與 Teacher 對齊)
    ↓
Weighted Average Layer (凍結，自 Teacher 複製)
    ↓
ReLU Student 頭 ({config.student_head_type})
    ↓
Conv1d Classifier (→ 3 類別)
```

Pipeline 另可將 **Conv 頭** 轉為 **SNN（LIF）** 並評估混合模型（見上節 3.3，若已執行）。

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
    """比較不同 Whisper 大小的 ReLU Student（同一 student-head 類型）"""
    from config import Config
    from model import load_model
    from dataset import create_test_dataloader
    from evaluate import Evaluator

    print("\n" + "=" * 60)
    print("📊 比較多個 ReLU Student 模型")
    print("=" * 60)

    results = []
    tag = args.student_head

    for model_size in args.model_sizes:
        print(f"\n評估 Whisper-{model_size} + Student ({tag})...")

        config = Config(
            model_size=model_size,
            student_head_type=args.student_head,
            conv_channels=args.conv_channels,
            mlp_hidden_dims=_mlp_dims_from_args(args),
        )
        model_path = Path(config.model_save_dir) / f"whisper_{model_size}_student_relu_{tag}_best.pt"

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
    print(f"模型比較結果 (ReLU Student, head={tag})")
    print("=" * 60)
    print(f"{'模型':<15} {'DER':<10} {'Speaker Confusion':<20}")
    print("-" * 45)
    for r in results:
        print(f"Whisper-{r['model_size']:<7} {r['der']:.2f}%{'':<5} {r['speaker_confusion']:.2f}%")

    return results


def _add_common_student_args(p):
    p.add_argument(
        "--model-size",
        type=str,
        default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper 模型大小",
    )
    p.add_argument(
        "--student-head",
        type=str,
        default="conv",
        choices=["conv", "mlp"],
        help="conv=論文 Conv1d+ReLU；mlp=全連接+ReLU",
    )
    p.add_argument("--conv-channels", type=int, default=256, help="conv 頭通道數")
    p.add_argument("--mlp-hidden", type=str, default="512,256", help="MLP 隱藏層，逗號分隔")


def main():
    parser = argparse.ArgumentParser(
        description="Whisper + ReLU Student（LSTM Teacher KD），供後續 SNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例：
  python main.py train --model-size medium --student-head conv --teacher-path models/teacher.pt
  python main.py evaluate --model-path models/whisper_medium_student_relu_conv_best.pt --student-head conv
  python main.py infer --model-path ... --student-head conv --audio-path audio.mp3
  python main.py pipeline --model-size medium --student-head conv --teacher-path models/teacher.pt
  python main.py pipeline ... --skip-snn
  python main.py compare --model-sizes tiny small medium --student-head conv
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    train_parser = subparsers.add_parser("train", help="知識蒸餾訓練 Student")
    _add_common_student_args(train_parser)
    train_parser.add_argument("--epochs", type=int, default=15, help="訓練 epochs")
    train_parser.add_argument("--batch-size", type=int, default=8, help="批次大小")
    train_parser.add_argument("--lr", type=float, default=5e-4, help="學習率")
    train_parser.add_argument("--window-size", type=float, default=20.0, help="音訊窗口大小（秒）")
    train_parser.add_argument(
        "--teacher-path",
        type=str,
        default=r"C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_LSTM\models\whisper_medium_bilstm_best.pt",
        help="Bi-LSTM Teacher 權重",
    )
    train_parser.add_argument("--lstm-hidden", type=int, default=256, help="Teacher LSTM 隱藏層（單向），載入時會依權重校正")
    train_parser.add_argument("--lstm-layers", type=int, default=2, help="Teacher LSTM 層數")
    train_parser.add_argument("--kd-temp", type=float, default=4.0, help="KD 溫度 T")
    train_parser.add_argument("--kd-alpha", type=float, default=0.5, help="CE 權重 α（KD 為 1-α）")

    eval_parser = subparsers.add_parser("evaluate", help="評估 Student")
    _add_common_student_args(eval_parser)
    eval_parser.add_argument("--model-path", type=str, required=True, help="Student 權重路徑")
    median_group = eval_parser.add_mutually_exclusive_group()
    median_group.add_argument("--use-median", dest="use_median", action="store_true", help="啟用中值濾波後處理")
    median_group.add_argument("--no-median", dest="use_median", action="store_false", help="停用中值濾波後處理")
    eval_parser.set_defaults(use_median=False)
    eval_parser.add_argument("--median-size", type=int, default=5, help="中值濾波器核大小（奇數）")

    infer_parser = subparsers.add_parser("infer", help="推論單一音檔")
    _add_common_student_args(infer_parser)
    infer_parser.add_argument("--model-path", type=str, required=True, help="Student 權重路徑")
    infer_parser.add_argument("--audio-path", type=str, required=True, help="音訊檔案路徑")
    infer_parser.add_argument("--srt-path", type=str, default=None, help="SRT 標註（可選）")
    infer_parser.add_argument("--output-srt", type=str, default=None, help="輸出 SRT 路徑")
    infer_parser.add_argument("--output-dir", type=str, default=None, help="輸出目錄")
    median_group_infer = infer_parser.add_mutually_exclusive_group()
    median_group_infer.add_argument("--use-median", dest="use_median", action="store_true")
    median_group_infer.add_argument("--no-median", dest="use_median", action="store_false")
    infer_parser.set_defaults(use_median=False)
    infer_parser.add_argument("--median-size", type=int, default=5)

    pipeline_parser = subparsers.add_parser("pipeline", help="完整 pipeline（訓練+評估+報告）")
    _add_common_student_args(pipeline_parser)
    pipeline_parser.add_argument("--epochs", type=int, default=15)
    pipeline_parser.add_argument("--batch-size", type=int, default=8)
    pipeline_parser.add_argument("--lr", type=float, default=5e-4)
    pipeline_parser.add_argument("--window-size", type=float, default=20.0)
    pipeline_parser.add_argument(
        "--teacher-path",
        type=str,
        default=r"C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_LSTM\models\whisper_medium_bilstm_best.pt",
    )
    pipeline_parser.add_argument("--lstm-hidden", type=int, default=256)
    pipeline_parser.add_argument("--lstm-layers", type=int, default=2)
    pipeline_parser.add_argument("--kd-temp", type=float, default=4.0)
    pipeline_parser.add_argument("--kd-alpha", type=float, default=0.5)
    pipeline_parser.add_argument(
        "--skip-snn",
        action="store_true",
        help="略過 Conv 頭轉 SNN 與混合模型評估",
    )
    pipeline_parser.add_argument(
        "--snn-timesteps",
        type=int,
        default=8,
        help="SNN 模擬時間步數 T（Leaky LIF）",
    )
    pipeline_parser.add_argument(
        "--snn-beta",
        type=float,
        default=0.9,
        help="Leaky LIF 漏電係數 β",
    )
    pipeline_parser.add_argument(
        "--snn-finetune-epochs",
        type=int,
        default=0,
        help="轉 SNN 並校準後，僅微調 SNN 頭（複製之 Conv+LIF）的 epoch 數；0 表示不微調",
    )
    pipeline_parser.add_argument(
        "--snn-ft-lr",
        type=float,
        default=1e-4,
        help="SNN 頭微調學習率（僅 --snn-finetune-epochs>0 時生效）",
    )
    median_group_pipe = pipeline_parser.add_mutually_exclusive_group()
    median_group_pipe.add_argument("--use-median", dest="use_median", action="store_true")
    median_group_pipe.add_argument("--no-median", dest="use_median", action="store_false")
    pipeline_parser.set_defaults(use_median=False)
    pipeline_parser.add_argument("--median-size", type=int, default=5)

    compare_parser = subparsers.add_parser("compare", help="比較多個 Whisper 大小的 Student")
    _add_common_student_args(compare_parser)
    compare_parser.add_argument(
        "--model-sizes",
        type=str,
        nargs="+",
        default=["tiny", "small", "medium"],
        help="要比較的 Whisper 大小列表",
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
