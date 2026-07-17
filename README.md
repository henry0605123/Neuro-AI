# Neuro-AI — 兒童臨床語音 ASR / 語者分離

兒童臨床訪談語音的處理程式碼:**語者分離(child/adult diarization)** 與 **ASR 微調**,
以及把兩者串成「一段完整音檔 → 小孩語音片段 + 逐字稿」的整合 pipeline。

> ⚠️ **本 repo 只含程式碼。** 所有臨床資料(音檔、逐字稿、標註)、模型權重與特徵快取
> 因隱私與檔案大小考量**皆未上傳**,已於 `.gitignore` 排除。要實際執行需自備資料與模型。

## 目錄

| 路徑 | 內容 |
|---|---|
| `ASR_finetune/` | ASR 微調(Breeze-ASR-25 / Whisper),19 折 LOO、bootstrap 信賴區間、全參數微調的抗 OOM 配方等。核心:`run_loo_breeze_full.py`(全參數微調)、`lora_train.py`、`evaluate.py`、`bootstrap_ci.py`、`config.py`。 |
| `child_split/` | 整合 pipeline:`diar_infer.py`(語者分離推論)+ `pipeline_app.py`(Gradio 前端:上傳音檔 → 輸出小孩語音片段與逐字稿)。 |

## 執行前需自備

- **ASR 模型**:HuggingFace 格式的微調權重資料夾,用 `ASR_MODEL_DIR` 環境變數指定
  (`pipeline_app.py`)。
- **語者分離模型**:handoff 資料夾(含 `model_code/` 與權重),用 `DIAR_HANDOFF_DIR` 環境變數指定
  (`diar_infer.py`)。
- **原始資料**:用 `ASR_DATA_ROOT` 環境變數指定(`config.py`)。

## 主要相依套件

`torch 2.8 (cu129)`、`transformers 4.46.3`(diarization 有版本相容 shim,勿升級)、
`torchaudio`、`gradio`、`loralib`、`bitsandbytes`、`jiwer`、`pytorch-lightning`。

## 推論注意事項

微調後的 Whisper/Breeze ASR 推論**務必**加 `repetition_penalty=1.3, no_repeat_ngram_size=5`,
否則短句會陷入重複解碼。詳見各模型資料夾的 `README_HANDOFF.md`。
