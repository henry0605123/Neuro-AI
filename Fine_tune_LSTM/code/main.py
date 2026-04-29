# -*- coding: utf-8 -*-
"""
Whisper + Bi-LSTM 語者分離模型
主程式：整合訓練、評估、推論的統一入口

架構：Whisper Encoder + Weighted Average + 2 層 Bi-LSTM + Classifier
日期：2026-01-24

使用方式：
=========
1. 訓練模型：
   python main.py train --model-size medium --epochs 15

2. 評估模型：
   python main.py evaluate --model-path models/whisper_medium_bilstm_best.pt

3. 推論單一音檔：
   python main.py infer --model-path models/whisper_medium_bilstm_best.pt --audio-path path/to/audio.mp3

4. 完整 Pipeline（訓練 + 評估）：
   python main.py pipeline --model-size medium --epochs 15

5. K-fold Pipeline（config.enable_kfold_cv=True 或加上 --kfold）：
   python main.py pipeline --model-size medium --epochs 15 --kfold
"""

import os
import sys
import json
import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
import numpy as np

# 對應 Hugging Face：openai/whisper-{size}；large-turbo → large-v3-turbo（見 config.whisper_model_name）
WHISPER_MODEL_SIZE_CHOICES = (
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
    "large-turbo",
)


def add_lora_cli_group(parser: argparse.ArgumentParser) -> None:
    """Whisper Encoder LoRA：與 config.use_lora 對齊；未指定時沿用 config.py。"""
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--use-lora",
        dest="use_lora",
        action="store_const",
        const=True,
        default=None,
        help="啟用 Whisper Encoder LoRA（未指定時沿用 config.py）",
    )
    grp.add_argument(
        "--no-use-lora",
        dest="use_lora",
        action="store_const",
        const=False,
        default=None,
        help="停用 Whisper Encoder LoRA",
    )


def _config_lora_overrides(args) -> dict:
    if getattr(args, "use_lora", None) is None:
        return {}
    return {"use_lora": args.use_lora}


def _build_pipeline_config(args, cfg_ref):
    """合併 config.py 預設（cfg_ref）與 pipeline CLI 覆寫欄位。"""
    from config import Config

    extra: Dict[str, Any] = {}
    pkf = getattr(args, "pipeline_use_kfold", None)
    if pkf is not None:
        extra["enable_kfold_cv"] = bool(pkf)
    nf = getattr(args, "n_folds", None)
    if nf is not None:
        extra["n_folds"] = int(nf)
    return Config(
        model_size=args.model_size,
        epochs=args.epochs if args.epochs is not None else cfg_ref.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        window_size=args.window_size,
        lstm_hidden_size=args.lstm_hidden,
        lstm_num_layers=args.lstm_layers,
        median_filter_size=args.median_size,
        **extra,
        **_config_lora_overrides(args),
    )


def _pipeline_single_fold(config, args):
    """單次訓練／驗證切分 + 固定測試集評估 + 單一報告。"""
    from train import train_model
    from dataset import create_test_dataloader
    from evaluate import Evaluator

    print("\n📌 Step 1: 訓練模型")
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

    print("\n📌 Step 3: 生成報告")
    print("-" * 40)
    generate_report(config, train_summary, eval_results)
    print("\n✅ Pipeline 完成！")
    return {"training": train_summary, "evaluation": eval_results}


def pipeline_kfold_command(config, args):
    """
    K-fold：每折獨立訓練與驗證切分。
    - 若 train_val_test_ratio 第三項 > 0：各折在「同一固定測試集」上評估後平均。
    - 若第三項 = 0：不保留測試桶，改在各折的「該折驗證集」上評估後平均（無獨立測試集）。
    不另選整體最佳折作為唯一輸出。
    """
    from train import train_model
    from dataset import (
        _normalize_ratio,
        create_dataloaders_from_sample_lists,
        create_test_dataloader,
        get_kfold_train_val_samples,
    )
    from evaluate import Evaluator

    n_folds = int(getattr(config, "n_folds", 5))
    base_out = config.output_dir
    base_log = config.log_dir
    base_model = config.model_save_dir

    ratio = _normalize_ratio(config)
    eval_on_per_fold_val_only = ratio[2] == 0

    print("\n" + "=" * 60)
    print(f"🔄 K-fold Pipeline（{n_folds} 折交叉驗證）")
    print("=" * 60)
    print(
        "每折權重與圖表寫入各子目錄 kfold_fold{k}；"
        f"彙總報告與 JSON 寫入: {base_out}"
    )
    if eval_on_per_fold_val_only:
        print(
            "\nℹ️  train_val_test_ratio 測試份額為 0：不建立獨立測試集；"
            "彙總 DER 為各折在「該折驗證集」上之結果再平均（與單一固定測試集模式不同）。"
        )
        test_loader = None
    else:
        test_loader = create_test_dataloader(config)

    fold_rows: List[Dict[str, Any]] = []
    fold_summaries: List[Dict[str, Any]] = []

    for fold in range(n_folds):
        print("\n" + "=" * 60)
        print(f"📂 第 {fold + 1} / {n_folds} 折")
        print("=" * 60)

        fold_config = replace(
            config,
            model_save_dir=os.path.join(base_model, f"kfold_fold{fold}"),
            log_dir=os.path.join(base_log, f"kfold_fold{fold}"),
            output_dir=os.path.join(base_out, f"kfold_fold{fold}"),
            kfold_fold_index=fold,
        )
        for d in (fold_config.model_save_dir, fold_config.log_dir, fold_config.output_dir):
            os.makedirs(d, exist_ok=True)

        seed_f = int(config.seed) + fold
        torch.manual_seed(seed_f)
        np.random.seed(seed_f)

        train_samples, val_samples = get_kfold_train_val_samples(config, fold)
        train_loader, val_loader = create_dataloaders_from_sample_lists(
            fold_config,
            train_samples,
            val_samples,
            split_message=f"\nK-fold 折 {fold + 1}/{n_folds}：訓練 {len(train_samples)} 樣本 / 驗證 {len(val_samples)} 樣本",
        )

        model, summary = train_model(
            fold_config,
            train_loader,
            val_loader,
            print_config=(fold == 0),
        )
        fold_summaries.append(summary)

        evaluator = Evaluator(fold_config)
        eval_loader = val_loader if eval_on_per_fold_val_only else test_loader
        eval_results = evaluator.evaluate(
            model=model,
            data_loader=eval_loader,
            save_dir=fold_config.output_dir,
            apply_median_filter=args.use_median,
        )
        der = eval_results["der"]
        fold_rows.append(
            {
                "fold": fold,
                "der": float(der.der),
                "false_alarm": float(der.false_alarm),
                "missed_detection": float(der.missed_detection),
                "speaker_confusion": float(der.speaker_confusion),
            }
        )

        model.cpu()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    keys = ["der", "false_alarm", "missed_detection", "speaker_confusion"]
    aggregate = {}
    for k in keys:
        vals = np.array([r[k] for r in fold_rows], dtype=np.float64)
        aggregate[f"{k}_mean"] = float(vals.mean())
        aggregate[f"{k}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

    summary_path = Path(base_out) / "kfold_summary.json"
    Path(base_out).mkdir(parents=True, exist_ok=True)
    split_note = (
        "train_val_test_ratio 第三項為 0：各折 DER 在該折驗證集上計算後平均（無獨立測試集）。"
        if eval_on_per_fold_val_only
        else "各折使用相同固定測試集（與單次 pipeline 之 create_test_dataloader 一致）。"
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_folds": n_folds,
                "folds": fold_rows,
                "aggregate": aggregate,
                "eval_metric_source": (
                    "per_fold_validation" if eval_on_per_fold_val_only else "held_out_test"
                ),
                "test_split_note": split_note,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n📄 K-fold 彙總 JSON: {summary_path}")

    generate_report_kfold(
        config=config,
        fold_rows=fold_rows,
        fold_summaries=fold_summaries,
        aggregate=aggregate,
        use_median=bool(args.use_median),
        report_dir=base_out,
        eval_on_per_fold_val_only=eval_on_per_fold_val_only,
    )
    done_msg = (
        "報告為各折驗證集指標之平均（無獨立測試集），非單選最佳折。"
        if eval_on_per_fold_val_only
        else "報告為各折測試集指標之平均，非單選最佳折。"
    )
    print(f"\n✅ K-fold Pipeline 完成（{done_msg}）")
    return {"kfold_folds": fold_rows, "aggregate": aggregate, "summaries": fold_summaries}


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
    
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def train_command(args):
    """執行訓練命令"""
    from config import Config
    from train import train_model
    
    print("\n" + "=" * 60)
    print("🚀 開始訓練 Whisper + Bi-LSTM 語者分離模型")
    print("=" * 60)
    
    cfg_ref = Config()
    config = Config(
        model_size=args.model_size,
        epochs=args.epochs if args.epochs is not None else cfg_ref.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        window_size=args.window_size,
        lstm_hidden_size=args.lstm_hidden,
        lstm_num_layers=args.lstm_layers,
        **_config_lora_overrides(args),
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
    print("📊 開始評估 Bi-LSTM 模型")
    print("=" * 60)
    
    # 更新 median filter 配置
    config = Config(
        model_size=args.model_size,
        lstm_hidden_size=args.lstm_hidden,
        lstm_num_layers=args.lstm_layers,
        median_filter_size=args.median_size,
        **_config_lora_overrides(args),
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
        apply_median_filter=args.use_median
    )
    
    print("\n✅ 評估完成！")
    
    return results


def infer_command(args):
    """執行推論命令"""
    from config import Config
    from model import load_model
    from evaluate import inference_on_audio, predictions_to_srt
    
    print("\n" + "=" * 60)
    print("🔍 開始推論 (Bi-LSTM)")
    print("=" * 60)
    
    config = Config(
        model_size=args.model_size,
        lstm_hidden_size=args.lstm_hidden,
        lstm_num_layers=args.lstm_layers,
        **_config_lora_overrides(args),
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
        json_path=args.json_path,
        save_dir=args.output_dir or config.output_dir
    )
    
    if args.output_srt:
        output_path = args.output_srt
    else:
        output_path = Path(args.audio_path).with_suffix('.predicted.srt')
    
    predictions_to_srt(
        results["predictions"],
        config.step_size_ms,
        str(output_path)
    )
    
    print("\n✅ 推論完成！")
    
    return results


def pipeline_command(args):
    """執行完整 Pipeline（訓練 + 評估）；若 config 或 CLI 啟用 K-fold 則改跑五折流程。"""
    from config import Config

    print("\n" + "=" * 60)
    print("🔄 執行完整 Pipeline (Whisper + Bi-LSTM)")
    print("=" * 60)

    cfg_ref = Config()
    config = _build_pipeline_config(args, cfg_ref)

    if getattr(config, "enable_kfold_cv", False):
        return pipeline_kfold_command(config, args)
    return _pipeline_single_fold(config, args)


def generate_report_kfold(
    config,
    fold_rows: List[Dict[str, Any]],
    fold_summaries: List[Dict[str, Any]],
    aggregate: Dict[str, float],
    use_median: bool,
    report_dir: str,
    *,
    eval_on_per_fold_val_only: bool = False,
) -> None:
    """K-fold 彙總報告：各折 DER 等之平均 ± 標準差（不選整體最佳折）。"""
    report_path = Path(report_dir) / "ANALYSIS_REPORT.md"
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    if getattr(config, "test_data_dir", None):
        test_data_source = str(config.test_data_dir)
    elif getattr(config, "enable_fixed_split_from_train_dir", False) and not (
        getattr(config, "test_audio_path", "") or ""
    ).strip():
        ratio = getattr(config, "train_val_test_ratio", (2, 1, 1))
        test_data_source = (
            f"由 train_data_dir 固定比例隨機切分 (train/val/test = {ratio[0]}/{ratio[1]}/{ratio[2]})"
        )
    else:
        test_data_source = config.test_audio_path

    rows_md = "\n".join(
        f"| {r['fold'] + 1} | {r['der']:.2f}% | {r['false_alarm']:.2f}% | {r['missed_detection']:.2f}% | {r['speaker_confusion']:.2f}% |"
        for r in fold_rows
    )

    best_val_losses = [s.get("best_val_loss", 0.0) for s in fold_summaries if s]
    mean_bvl = float(np.mean(best_val_losses)) if best_val_losses else 0.0
    std_bvl = float(np.std(best_val_losses, ddof=1)) if len(best_val_losses) > 1 else 0.0

    eval_bullet = (
        "- **最終指標**: `train_val_test_ratio` 第三項為 **0**（無獨立測試桶）。各折在**該折驗證集**上計算 DER 等後，取**算術平均 ± 樣本標準差**（與「同一固定測試集」模式不同；驗證集亦用於該折內 early stopping，解讀時略偏樂觀屬常見限制）。\n"
        "- **不**以整體最佳折作為唯一代表模型。"
        if eval_on_per_fold_val_only
        else "- **最終指標**: 各折在**同一固定測試集**上計算 DER 等指標後，取**算術平均 ± 樣本標準差**；**不**以整體最佳折作為唯一代表模型。\n"
    )
    metric_target = "該折驗證集" if eval_on_per_fold_val_only else "固定測試集"
    section2_title = "各折於「該折驗證集」之 DER 分量" if eval_on_per_fold_val_only else "各折於固定測試集之 DER 分量"
    section3_caption = "五折平均 ± 標準差（主要輸出）"
    sec4_ref = "第 3 節「平均」" if eval_on_per_fold_val_only else "第 3 節「測試集平均」"
    conclusion = (
        f"本實驗以 {int(getattr(config, 'n_folds', 5))} 折交叉驗證訓練多個模型，並在各折驗證集上彙總平均表現（無另留測試集）。"
        if eval_on_per_fold_val_only
        else f"本實驗以 {int(getattr(config, 'n_folds', 5))} 折交叉驗證訓練多個模型，並在固定測試集上彙總平均表現，避免單次隨機切分或單折最佳化造成的樂觀偏差。"
    )

    report = f"""# 語者分離模型實驗報告 — K-fold 交叉驗證 (Whisper + Bi-LSTM)

**生成時間**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 1. 實驗設置摘要

- **驗證方式**: {int(getattr(config, "n_folds", 5))}-fold 交叉驗證（樣本層級 `sklearn.model_selection.KFold`，與獨立 `val_data_dir` 互斥）。
{eval_bullet}
- **Median filter（評估）**: {'啟用' if use_median else '停用'}
- **訓練資料根目錄**: {config.train_data_dir}
- **評估資料來源說明**: {test_data_source}（指標實際計算於：**{metric_target}**）
- **每折子目錄**: `{config.model_save_dir}/kfold_fold0` … `kfold_fold{int(getattr(config, "n_folds", 5)) - 1}`（同樣結構於 `log_dir`、`output_dir`）
- **彙總 JSON**: `{report_dir}/kfold_summary.json`

### 1.1 模型與訓練（與單次 pipeline 相同之 CLI／config）
| 參數 | 值 |
|------|------|
| Whisper 模型大小 | {config.model_size} |
| 批次大小 | {config.batch_size} |
| 學習率 | {config.learning_rate} |
| Epochs | {config.epochs} |
| LSTM hidden | {config.lstm_hidden_size} |
| LSTM 層數 | {config.lstm_num_layers} |
| Encoder LoRA | {'啟用' if getattr(config, 'use_lora', False) else '停用'} |

---

## 2. {section2_title}

| 折 | DER | False Alarm | Missed Detection | Speaker Confusion |
|----|-----|-------------|------------------|-------------------|
{rows_md}

---

## 3. {section3_caption}

| 指標 | 平均 ± 標準差 |
|------|----------------|
| **DER** | **{aggregate['der_mean']:.2f}% ± {aggregate['der_std']:.2f}%** |
| False Alarm | {aggregate['false_alarm_mean']:.2f}% ± {aggregate['false_alarm_std']:.2f}% |
| Missed Detection | {aggregate['missed_detection_mean']:.2f}% ± {aggregate['missed_detection_std']:.2f}% |
| Speaker Confusion | {aggregate['speaker_confusion_mean']:.2f}% ± {aggregate['speaker_confusion_std']:.2f}% |

---

## 4. 各折訓練摘要（驗證損失最佳值，僅供參考）

各折內部仍以驗證損失選該折最佳 epoch 權重；**報告主結論請以上方第 3 節{sec4_ref}為準。**

- 各折最佳驗證損失之平均 ± 標準差: **{mean_bvl:.4f} ± {std_bvl:.4f}**

---

## 5. 結論

{conclusion}

"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"報告已生成: {report_path}")


def generate_report(config, train_summary, eval_results):
    """生成結果分析報告 (Bi-LSTM 版本)"""
    from datetime import datetime
    
    report_path = Path(config.output_dir) / "ANALYSIS_REPORT.md"
    
    der = eval_results.get("der")
    
    if getattr(config, "test_data_dir", None):
        test_data_source = config.test_data_dir
    elif getattr(config, "enable_fixed_split_from_train_dir", False) and not (getattr(config, "test_audio_path", "") or "").strip():
        ratio = getattr(config, "train_val_test_ratio", (2, 1, 1))
        test_data_source = (
            f"由 train_data_dir 固定比例隨機切分 (train/val/test = {ratio[0]}/{ratio[1]}/{ratio[2]})"
        )
    else:
        test_data_source = config.test_audio_path

    report = f"""# 語者分離模型實驗報告 (Whisper + Bi-LSTM)

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
| Encoder LoRA | {'啟用 (rank=' + str(config.lora_rank) + ')' if getattr(config, 'use_lora', False) else '停用'} |
| 凍結 Whisper Decoder | {'是（建議）' if getattr(config, 'freeze_whisper_decoder', True) else '否'} |

### 1.2 Bi-LSTM 架構設定
| 參數 | 值 |
|------|------|
| LSTM 隱藏層維度（單向） | {config.lstm_hidden_size} |
| LSTM 隱藏層維度（雙向拼接） | {config.lstm_hidden_size * 2} |
| LSTM 層數 | {config.lstm_num_layers} |
| Dropout | {config.dropout} |
| **中間層架構** | **2 層 Bi-LSTM（替換原本的 3 層 Conv1D）** |

### 1.3 資料集
- **訓練資料**: {config.train_data_dir}
- **測試資料**: {test_data_source}
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

### 2.3 若驗證準確率突然掉到約 37% 且不再回升
多為**訓練不穩定**（與「三類別亂猜約 33%」接近）：常見原因為 **學習率過高**、**Whisper Decoder 一併更新**、或 **AMP 數值問題**。請確認已使用較新版本程式之預設：**凍結 Decoder**、**較保守學習率（如 1e-4）**；必要時關閉 AMP（`use_amp=False`）或再降 LR。

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
| **Bi-LSTM** | **2 層雙向 LSTM + LayerNorm + Dropout** |

---

## 4. Bi-LSTM vs Conv1D 分析

### 4.1 理論優勢

**Bi-LSTM 的優勢**:
1. **長距離依賴**: LSTM 可以捕捉更長的時間依賴關係
2. **雙向上下文**: 同時利用過去和未來的資訊
3. **時序建模**: 更適合語音這種時序資料

**Conv1D 的優勢**:
1. **平行計算**: 可以完全平行化，訓練更快
2. **局部特徵**: 擅長捕捉局部模式
3. **參數效率**: 通常參數量較少

### 4.2 實驗結果對比

請比較此實驗的 DER 與之前 Conv1D 版本的結果，
以判斷哪種架構在您的資料集上表現更好。

---

## 5. 視覺化結果

以下視覺化圖表已生成：

1. **訓練曲線**: `{config.log_dir}/training_curves.png`
2. **混淆矩陣**: `{config.output_dir}/confusion_matrix.png`
3. **甘特圖**: `{config.output_dir}/gantt_chart.png`
4. **多窗口比較圖**: `{config.output_dir}/gantt_comparison.png`

---

## 6. 結論

本實驗使用 Whisper-{config.model_size} 作為特徵提取器，
將中間層的 3 層 Conv1D 替換為 2 層 Bi-LSTM。

### 架構摘要:
```
Whisper Encoder (凍結)
    ↓
Weighted Average Layer
    ↓
2 層 Bi-LSTM (hidden_size={config.lstm_hidden_size}, bidirectional)
    ↓
Layer Normalization + Dropout
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
    """比較不同模型的效能"""
    from config import Config
    from model import load_model
    from dataset import create_test_dataloader
    from evaluate import Evaluator
    
    print("\n" + "=" * 60)
    print("📊 比較多個 Bi-LSTM 模型")
    print("=" * 60)
    
    results = []
    
    for model_size in args.model_sizes:
        print(f"\n評估 Whisper-{model_size} + Bi-LSTM...")
        
        config = Config(model_size=model_size)
        model_path = Path(config.model_save_dir) / f"whisper_{model_size}_bilstm_best.pt"
        
        if not model_path.exists():
            print(f"  ⚠️ 模型不存在: {model_path}")
            continue
        
        model = load_model(config, str(model_path))
        test_loader = create_test_dataloader(config)
        
        evaluator = Evaluator(config)
        eval_results = evaluator.evaluate(
            model=model,
            data_loader=test_loader,
            save_dir=None
        )
        
        results.append({
            "model_size": model_size,
            "der": eval_results["der"].der,
            "speaker_confusion": eval_results["der"].speaker_confusion,
        })
    
    print("\n" + "=" * 60)
    print("模型比較結果 (Bi-LSTM)")
    print("=" * 60)
    print(f"{'模型':<15} {'DER':<10} {'Speaker Confusion':<20}")
    print("-" * 45)
    for r in results:
        print(f"Whisper-{r['model_size']:<7} {r['der']:.2f}%{'':<5} {r['speaker_confusion']:.2f}%")
    
    return results


def main():
    """主程式入口"""
    parser = argparse.ArgumentParser(
        description="Whisper + Bi-LSTM 語者分離模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例：
  # 訓練 medium 模型
  python main.py train --model-size medium --epochs 15

  # 評估模型
  python main.py evaluate --model-path models/whisper_medium_bilstm_best.pt

  # 推論單一音檔
  python main.py infer --model-path models/whisper_medium_bilstm_best.pt --audio-path audio.mp3

  # 執行完整 pipeline
  python main.py pipeline --model-size medium
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # ===== 通用 LSTM 參數 =====
    lstm_args = {
        "--lstm-hidden": {"type": int, "default": 256, "help": "LSTM 隱藏層維度（單向）"},
        "--lstm-layers": {"type": int, "default": 2, "help": "LSTM 層數"},
    }
    
    # ===== 訓練命令 =====
    train_parser = subparsers.add_parser("train", help="訓練模型")
    train_parser.add_argument(
        "--model-size", type=str, default="medium",
        choices=WHISPER_MODEL_SIZE_CHOICES,
        help="Whisper 模型大小"
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="訓練 epochs（未指定時沿用 config.py 的 epochs）",
    )
    train_parser.add_argument("--batch-size", type=int, default=8, help="批次大小")
    train_parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="學習率（Whisper-large + LoRA 建議 1e-4～5e-5；過高易在第 3～4 epoch 崩潰）",
    )
    train_parser.add_argument("--window-size", type=float, default=20.0, help="音訊窗口大小（秒）")
    train_parser.add_argument("--lstm-hidden", type=int, default=256, help="LSTM 隱藏層維度")
    train_parser.add_argument(
        "--lstm-layers",
        "--lstm-num-layers",
        type=int,
        default=2,
        dest="lstm_layers",
        help="LSTM 層數（--lstm-num-layers 為同義別名）",
    )
    add_lora_cli_group(train_parser)
    
    # ===== 評估命令 =====
    eval_parser = subparsers.add_parser("evaluate", help="評估模型")
    eval_parser.add_argument("--model-path", type=str, required=True, help="模型權重路徑")
    eval_parser.add_argument(
        "--model-size", type=str, default="medium",
        choices=WHISPER_MODEL_SIZE_CHOICES,
        help="Whisper 模型大小"
    )
    eval_parser.add_argument("--lstm-hidden", type=int, default=256, help="LSTM 隱藏層維度")
    eval_parser.add_argument(
        "--lstm-layers",
        "--lstm-num-layers",
        type=int,
        default=2,
        dest="lstm_layers",
        help="LSTM 層數（--lstm-num-layers 為同義別名）",
    )
    add_lora_cli_group(eval_parser)
    # Median Filter 參數
    median_group = eval_parser.add_mutually_exclusive_group()
    median_group.add_argument("--use-median", dest="use_median", action="store_true", help="啟用中值濾波後處理")
    median_group.add_argument("--no-median", dest="use_median", action="store_false", help="停用中值濾波後處理")
    eval_parser.set_defaults(use_median=False)
    eval_parser.add_argument(
        "--median-size",
        "--median-filter-size",
        type=int,
        default=5,
        dest="median_size",
        help="中值濾波器核大小（奇數；預設=5）。須併用 --use-median 才會在評估時套用",
    )
    
    # ===== 推論命令 =====
    infer_parser = subparsers.add_parser("infer", help="推論單一音檔")
    infer_parser.add_argument("--model-path", type=str, required=True, help="模型權重路徑")
    infer_parser.add_argument("--audio-path", type=str, required=True, help="音訊檔案路徑")
    infer_parser.add_argument("--srt-path", type=str, default=None, help="SRT 標註檔案路徑（可選）")
    infer_parser.add_argument("--rttm-path", type=str, default=None, help="RTTM 標註檔路徑（可選；與 SRT 皆提供時優先 RTTM）")
    infer_parser.add_argument(
        "--json-path",
        type=str,
        default=None,
        help="編輯標註 JSON 路徑（可選；優先順序：RTTM → JSON → SRT）",
    )
    infer_parser.add_argument("--output-srt", type=str, default=None, help="輸出 SRT 路徑")
    infer_parser.add_argument("--output-dir", type=str, default=None, help="輸出目錄")
    infer_parser.add_argument(
        "--model-size", type=str, default="medium",
        choices=WHISPER_MODEL_SIZE_CHOICES,
        help="Whisper 模型大小"
    )
    infer_parser.add_argument("--lstm-hidden", type=int, default=256, help="LSTM 隱藏層維度")
    infer_parser.add_argument(
        "--lstm-layers",
        "--lstm-num-layers",
        type=int,
        default=2,
        dest="lstm_layers",
        help="LSTM 層數（--lstm-num-layers 為同義別名）",
    )
    add_lora_cli_group(infer_parser)
    # Median Filter 參數
    median_group_infer = infer_parser.add_mutually_exclusive_group()
    median_group_infer.add_argument("--use-median", dest="use_median", action="store_true", help="啟用中值濾波後處理")
    median_group_infer.add_argument("--no-median", dest="use_median", action="store_false", help="停用中值濾波後處理")
    infer_parser.set_defaults(use_median=False)
    infer_parser.add_argument(
        "--median-size",
        "--median-filter-size",
        type=int,
        default=5,
        dest="median_size",
        help="中值濾波器核大小（奇數；預設=5）。須併用 --use-median 才會套用",
    )
    
    # ===== Pipeline 命令 =====
    pipeline_parser = subparsers.add_parser("pipeline", help="執行完整 pipeline（訓練+評估）")
    pipeline_parser.add_argument(
        "--model-size", type=str, default="medium",
        choices=WHISPER_MODEL_SIZE_CHOICES,
        help="Whisper 模型大小"
    )
    pipeline_parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="訓練 epochs（未指定時沿用 config.py 的 epochs）",
    )
    pipeline_parser.add_argument("--batch-size", type=int, default=8, help="批次大小")
    pipeline_parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="學習率（與 train 子命令相同預設）",
    )
    pipeline_parser.add_argument("--window-size", type=float, default=20.0, help="音訊窗口大小（秒）")
    pipeline_parser.add_argument("--lstm-hidden", type=int, default=256, help="LSTM 隱藏層維度")
    pipeline_parser.add_argument(
        "--lstm-layers",
        "--lstm-num-layers",
        type=int,
        default=2,
        dest="lstm_layers",
        help="LSTM 層數（--lstm-num-layers 為同義別名）",
    )
    add_lora_cli_group(pipeline_parser)
    # Median Filter 參數
    median_group_pipe = pipeline_parser.add_mutually_exclusive_group()
    median_group_pipe.add_argument("--use-median", dest="use_median", action="store_true", help="啟用中值濾波後處理")
    median_group_pipe.add_argument("--no-median", dest="use_median", action="store_false", help="停用中值濾波後處理")
    pipeline_parser.set_defaults(use_median=False)
    pipeline_parser.add_argument(
        "--median-size",
        "--median-filter-size",
        type=int,
        default=5,
        dest="median_size",
        help="中值濾波器核大小（奇數；預設=5）。評估階段須併用 --use-median 才會套用",
    )
    kfold_grp = pipeline_parser.add_mutually_exclusive_group()
    kfold_grp.add_argument(
        "--kfold",
        dest="pipeline_use_kfold",
        action="store_true",
        default=None,
        help="啟用 K-fold 交叉驗證 pipeline（折數見 config.n_folds，預設 5）",
    )
    kfold_grp.add_argument(
        "--no-kfold",
        dest="pipeline_use_kfold",
        action="store_false",
        default=None,
        help="明確停用 K-fold（沿用 config.enable_kfold_cv）",
    )
    pipeline_parser.set_defaults(pipeline_use_kfold=None)
    pipeline_parser.add_argument(
        "--n-folds",
        type=int,
        default=None,
        dest="n_folds",
        help="K-fold 折數（未指定時沿用 config.n_folds）",
    )

    # ===== 比較命令 =====
    compare_parser = subparsers.add_parser("compare", help="比較多個模型")
    compare_parser.add_argument(
        "--model-sizes", type=str, nargs="+",
        default=["tiny", "small", "medium"],
        help="要比較的模型大小列表"
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
