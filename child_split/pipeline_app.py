# -*- coding: utf-8 -*-
"""Child-speech pipeline:整段音檔 → diarization 抓小孩片段 → 微調 ASR 轉錄 → Gradio UI。

用法:D:/Anaconda/envs/paper/python.exe child_split/pipeline_app.py
(首次點「開始分析」才載入模型;diar encoder ~2.5GB + ASR fp16 ~3GB,16GB 卡可同時常駐)
"""
import os
import sys
import json
import csv
import base64
import io
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import torch
import torchaudio
import gradio as gr

import diar_infer  # 語者分離(載入時會套 transformers shim)
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# ASR 模型資料夾(HuggingFace 格式)。用環境變數指定,預設本機的 breeze_full_final。
ASR_DIR = os.environ.get("ASR_MODEL_DIR", os.path.join(
    _HERE, "..", "ASR_finetune", "models", "breeze_full_final"))
SR = 16000
PAD_SEC = 0.25    # 切片前後 padding,避免切到字中間
MIN_DUR = 0.3     # 過短片段(多為雜訊)不轉錄,但仍列出時間戳

_models = {}      # lazy singletons: {"diar": ..., "asr": (model, processor)}


def _get_models():
    if "diar" not in _models:
        _models["diar"] = diar_infer.load_diar_model()
    if "asr" not in _models:
        proc = WhisperProcessor.from_pretrained(ASR_DIR)
        m = WhisperForConditionalGeneration.from_pretrained(
            ASR_DIR, torch_dtype=torch.float16).cuda().eval()
        _models["asr"] = (m, proc)
    return _models["diar"], _models["asr"]


def transcribe(wav_1d, asr_model, processor):
    """16kHz mono float tensor → 簡體中文文字。必加 anti-repetition 參數(handoff README)。"""
    feats = processor(wav_1d.numpy(), sampling_rate=SR,
                      return_tensors="pt").input_features.cuda().to(torch.float16)
    with torch.no_grad():
        ids = asr_model.generate(
            input_features=feats,
            language="zh", task="transcribe",
            num_beams=1, max_new_tokens=150,
            repetition_penalty=1.3,        # 必要,防重複鬼打牆
            no_repeat_ngram_size=5,        # 必要
        )
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def _fmt_time(t):
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:05.2f}"


def _audio_data_uri(wav_1d):
    buf = io.BytesIO()
    torchaudio.save(buf, wav_1d.unsqueeze(0), SR, format="wav")
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


def analyze(audio_path, progress=gr.Progress()):
    if not audio_path:
        return "<p>請先上傳音檔。</p>", None, None

    def prog(frac, desc):  # CLI 測試時沒有 gradio event,Progress 呼叫失敗就印出來
        try:
            progress(frac, desc=desc)
        except Exception:
            print(f"[{frac:.0%}] {desc}")

    prog(0.0, "載入模型(首次約 1-2 分鐘)...")
    diar_model, (asr_model, processor) = _get_models()

    prog(0.02, "語者分離中...")
    child_segs, _adult_segs, duration = diar_infer.run_inference(
        audio_path, diar_model,
        progress_cb=lambda i, n: prog(0.02 + 0.48 * i / max(n, 1),
                                      f"語者分離 {i}/{n} 視窗"))

    # 讀原音檔一次,之後直接切 tensor
    audio, sr = torchaudio.load(audio_path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != SR:
        audio = torchaudio.functional.resample(audio, sr, SR)
    audio = audio.float().squeeze(0)

    out_dir = tempfile.mkdtemp(prefix="child_segs_")
    results, html_rows = [], []
    n = len(child_segs)
    for i, (s, e) in enumerate(child_segs):
        ps, pe = max(0.0, s - PAD_SEC), min(duration, e + PAD_SEC)
        clip = audio[int(ps * SR):int(pe * SR)]
        wav_path = os.path.join(out_dir, f"child_{i+1:03d}_{s:.2f}-{e:.2f}.wav")
        torchaudio.save(wav_path, clip.unsqueeze(0), SR)
        if e - s >= MIN_DUR:
            prog(0.5 + 0.5 * i / max(n, 1), f"ASR 轉錄 {i+1}/{n}")
            text = transcribe(clip, asr_model, processor)
        else:
            text = "(片段過短,未轉錄)"
        results.append({"index": i + 1, "start_sec": round(s, 2), "end_sec": round(e, 2),
                        "duration_sec": round(e - s, 2), "text": text, "wav": wav_path})
        html_rows.append(
            f"<tr><td>{i+1}</td><td>{_fmt_time(s)} – {_fmt_time(e)}</td>"
            f"<td><audio controls preload='none' src='{_audio_data_uri(clip)}'></audio></td>"
            f"<td>{text}</td></tr>")

    # 下載檔
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"audio": os.path.basename(audio_path), "duration_sec": round(duration, 2),
                   "child_segments": results}, f, ensure_ascii=False, indent=2)
    csv_path = os.path.join(out_dir, "results.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["index", "start_sec", "end_sec", "duration_sec", "text"])
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in w.fieldnames})

    html = (f"<p>音檔長度 {_fmt_time(duration)},共 {n} 段兒童語音。切片存於 {out_dir}</p>"
            "<table border='1' cellpadding='6' style='border-collapse:collapse'>"
            "<tr><th>#</th><th>時間</th><th>播放</th><th>轉錄</th></tr>"
            + "".join(html_rows) + "</table>")
    return html, json_path, csv_path


with gr.Blocks(title="兒童語音分析 pipeline") as demo:
    gr.Markdown("## 兒童語音切片 + 轉錄\n上傳訪談音檔(mp3/wav)→ 語者分離抓出兒童語音 → 逐段 ASR 轉錄(簡體輸出)")
    audio_in = gr.Audio(type="filepath", label="上傳音檔")
    btn = gr.Button("開始分析", variant="primary")
    out_html = gr.HTML()
    with gr.Row():
        json_file = gr.File(label="下載 JSON")
        csv_file = gr.File(label="下載 CSV")
    btn.click(analyze, inputs=audio_in, outputs=[out_html, json_file, csv_file])

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--test":   # CLI end-to-end 測試,不開 UI
        html, jp, cp = analyze(sys.argv[2])
        with open(jp, encoding="utf-8") as f:
            data = json.load(f)
        for seg in data["child_segments"]:
            print(f"  #{seg['index']} {seg['start_sec']:7.2f}-{seg['end_sec']:7.2f}  {seg['text']}")
        print("JSON:", jp, "\nCSV:", cp)
    else:
        demo.queue().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
