# DER_finetune — 兒童臨床語音語者分離(Child/Adult Diarization)骨幹比較

Breeze-ASR-25 / Whisper 系列骨幹在兒童臨床語音 child/adult 語者分離任務上的
Leave-One-Out(LOO)交叉驗證程式碼：凍結 encoder、LoRA、全參數微調(含抗 OOM 配方：
gradient checkpointing + bf16-true)、以及 ChangPN 中文語音兩階段預訓練(pretrain → finetune)。

> ⚠️ **本 repo 只含程式碼。** 所有臨床資料(音檔、逐字稿、標註)、模型權重、訓練 log
> 與特徵快取，因隱私與檔案大小考量**皆未上傳**，已於 `.gitignore` 排除。
> Session 排除清單一律透過環境變數(`DIAR_EXCLUDE` 等)指定，程式碼本身**不內建任何
> 真實個案姓名、病歷號或本機路徑**。要實際執行需自備資料與模型。

## 目錄

| 路徑 | 內容 |
|---|---|
| `models/whisper.py` | `WhisperWrapper`：載入 whisper-base/small/medium 或 Breeze-ASR-25 當 encoder，支援 LoRA(loralib)+ BiLSTM/Conv 分類頭。 |
| `dataset_classes/` | 10 秒滑動視窗切割、frame-level 標籤(靜音/兒童/成人)、`DatasetGenerator`。 |
| `lightning_modules/classifier.py` | PyTorch Lightning 訓練模組（Adam / bitsandbytes 8-bit Adam、linear warmup）。 |
| `scripts/loo_cv_breeze_diar_val.py` | **主力腳本**：Breeze/whisper 系列 LOO，支援凍結 / LoRA / 全參數微調(`FULL_FT`+`GRAD_CKPT`+`BF16`)、single-split 模式。 |
| `scripts/loo_cv_changpn.py` / `loo_cv_changpn_scratch.py` | ChangPN(TalkBank)中文語音兩階段(pretrain+finetune)LOO 變體。 |
| `scripts/loo_cv_changpn_sim*.py` | ChangPN segment-level 模擬對話變體。 |
| `scripts/loo_cv_audioset_bilstm.py` / `loo_cv_audioset_fixed.py` | 以 AudioSet 起點的 ChangPN 兩階段變體。 |
| `scripts/loo_cv.py` / `loo_cv_fast.py` / `loo_cv_aug.py` | 基礎 LOO / 加速版 / pitch augmentation 變體。 |
| `scripts/bootstrap_ci.py` | 對 LOO fold 結果做 bootstrap 重抽，回報 25/50/75% 信賴區間。 |
| `scripts/paths_config.py` | 集中路徑設定(`ASD_DATA_ROOT` 環境變數)。 |
| `scripts/convert_output.py` / `evaluate.py` / `eval_example.py` / `infer_wav_file.py` / `main.py` / `read_phase1_acc.py` | 推論後處理、評估、單檔推論等輔助腳本。 |
| `configs/*.yaml` | 訓練超參數範例設定檔(路徑需自行編輯)。 |

## 執行前需自備

- **資料**：用 `ASD_DATA_ROOT` 環境變數指到你的資料根目錄，底下需有
  `data/Fine_tune/<session>/*.mp3` + 對應標註 json（新舊兩種 GT 格式皆支援，見各腳本的
  `parse_json`）。
- **Session 排除清單**（若有標註品質問題的個案）：用 `DIAR_EXCLUDE` 環境變數（逗號分隔關鍵字）。
- **ChangPN 語料**（選用，僅 `loo_cv_changpn*.py` 需要）：TalkBank 格式音檔 + `.cha` 逐字稿。
- **AudioSet / whisper_base_5k 預訓練 checkpoint**（選用，僅部分 `loo_cv_audioset_*.py` 需要）。

## 主要相依套件

`torch 2.8 (cu129)`、`transformers 4.46.3`（**勿升級**，`WhisperAttention` 有版本相容 shim）、
`pytorch-lightning`、`torchaudio`、`loralib`（不是 peft）、`bitsandbytes`（全參數微調抗 OOM 用）、
`scikit-learn`、`pandas`、`tensorboard`。

## 全參數微調抗 OOM 配方

```bash
FULL_FT=1 GRAD_CKPT=1 BF16=1 DIAR_LR=1e-5 DIAR_WARMUP_STEPS=200 \
  BREEZE_DIAR_BATCH=3 python scripts/loo_cv_breeze_diar_val.py
```

實測發現：對大 encoder（例如 Breeze 32 層）做全參數微調時，**gradient checkpointing 只能解決
一部分**（`output_hidden_states=True` 保留全部層輸出仍佔大量顯存），8-bit optimizer 幾乎沒有
額外幫助（瓶頸不在 optimizer states）；真正關鍵是 **`bf16-true`**（權重/梯度/optimizer 全 bf16，
省一半固定記憶體）。純 bf16 有 swamping 風險，務必搭配小 learning rate + warmup，並盯 `val_loss`
是否正常下降。細節見 `loo_cv_breeze_diar_val.py` 的 `GRAD_CKPT` / `BF16` 開關與行內註解。

## 主要環境變數一覽

| 變數 | 作用 |
|---|---|
| `ASD_DATA_ROOT` | 資料根目錄 |
| `DIAR_EXCLUDE` | 要排除的 session 關鍵字（逗號分隔） |
| `DIAR_EMBEDDING` | backbone：`breeze` / `whisper_base` / `whisper_small` / `whisper_medium` |
| `FREEZE_ENCODER` | 凍結整個 encoder，只訓練分類頭 + 層加權 |
| `FULL_FT` | 全參數微調（解凍整個 encoder） |
| `GRAD_CKPT` | gradient checkpointing（`FULL_FT` 時預設開啟） |
| `BF16` | bf16-true 精度（大模型全微調抗 OOM） |
| `ADAM8BIT` | bitsandbytes 8-bit Adam |
| `DIAR_LR` / `DIAR_WARMUP_STEPS` | 學習率 / linear warmup 步數 |
| `BREEZE_DIAR_BATCH` / `BREEZE_DIAR_FOLDS` / `BREEZE_DIAR_PATIENCE` | batch size / 折數 / early-stopping patience |
| `DIAR_TAG` | 輸出目錄後綴，避免不同設定撞名 |
