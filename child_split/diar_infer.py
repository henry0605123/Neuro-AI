# -*- coding: utf-8 -*-
"""語者分離推論模組(child/adult 時間戳)。

邏輯逐字複製自 diarization handoff 的 model_code/loo_cv_breeze_diar_val.py
(該腳本 module 層級會直接執行訓練,無法 import,故複製必要片段;勿改邏輯)。
唯一增修:run_inference 內音檔尾端 zero-pad 到 10s 倍數,避免最後不足一個視窗的音訊被丟棄。
"""
import os
import sys
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
# 語者分離 handoff 資料夾(含 model_code/ 與權重)。用環境變數指定,預設 ./diar_model
_HANDOFF = os.environ.get("DIAR_HANDOFF_DIR", os.path.join(_HERE, "diar_model"))
_MODEL_CODE = os.path.join(_HANDOFF, "model_code")
sys.path.insert(0, _MODEL_CODE)

import torch
import torchaudio
from whisper import WhisperWrapper  # handoff 的 model_code/whisper.py

# majority_filter / get_timestamps 直接 import 原始檔(只依賴 numpy)
_CONVERT_OUTPUT = os.path.join(
    _HERE, "..", "child-adult-diarization", "whisper-modeling", "scripts", "convert_output.py")
_spec = importlib.util.spec_from_file_location("convert_output", _CONVERT_OUTPUT)
_co = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_co)
majority_filter = _co.majority_filter
get_timestamps = _co.get_timestamps

# ── Transformers 4.46.3 compatibility shim(逐字複製) ────────────────────────
import transformers.models.whisper.modeling_whisper as _wm
_orig_wa_forward = _wm.WhisperAttention.forward

def _compat_wa_forward(self, *args, **kwargs):
    if getattr(self, 'config', None) is None:
        class _C:
            _attn_implementation = "eager"
        self.config = _C()
    result = _orig_wa_forward(self, *args, **kwargs)
    if isinstance(result, tuple) and len(result) == 2:
        return result[0], result[1], None
    return result

_wm.WhisperAttention.forward = _compat_wa_forward

ENC_LEN = 500      # first 500 of 1500 encoder frames = actual 10s
SAMPLE_RATE = 16000
WINDOW_SIZE = 10
DIAR_LORA = os.path.join(_HANDOFF, "lora_adapter_weights.pt")
DIAR_HEAD = os.path.join(_HANDOFF, "bilstm_head_weights.pt")


class WhisperWrapperFast(WhisperWrapper):
    """Pre-computed-mel / raw-audio wrapper with the mel->3000 padding shim.(逐字複製)"""

    def forward(self, x, length=None):
        _dt = next(self.backbone_model.parameters()).dtype
        if x.dim() == 3:
            features = torch.nn.functional.pad(x, (0, 3000 - x.shape[2])).cuda().to(_dt)
        else:
            new_x = [x[i].detach().cpu().numpy() for i in range(len(x))]
            features = self.feature_extractor(
                new_x, return_tensors="pt", sampling_rate=16000
            ).input_features.cuda().to(_dt)

        self.backbone_model.encoder.embed_positions = \
            self.backbone_model.encoder.embed_positions.from_pretrained(self.embed_positions.to(_dt))

        hidden_states = self.backbone_model.encoder(
            features, output_hidden_states=True
        ).hidden_states

        stacked = torch.stack(hidden_states, dim=0)[1:]
        stacked = stacked[:, :, :ENC_LEN, :].contiguous()
        _, *shape = stacked.shape
        stacked = stacked.view(self.backbone_model.config.num_hidden_layers, -1)
        norm_weights = torch.nn.functional.softmax(self.weights, dim=-1)
        weighted = (norm_weights.unsqueeze(-1) * stacked).sum(dim=0)
        return self.model_seq(weighted.view(*shape).transpose(1, 2))


def load_diar_model(lora_path=DIAR_LORA, head_path=DIAR_HEAD):
    """建骨架(HF 原始 Breeze)→ 套 LoRA adapter + BiLSTM head/層加權。

    注意:handoff 的完整 ckpt(epoch=5-val_loss=0.522.ckpt)只有 127MB,傳輸被截斷不可用;
    但 LoRA 訓練時 base 權重凍結,故 HF base + 兩個 adapter 檔即完整等價。
    Key 處理照 README:去 model. 前綴、跳過 embed_positions、strict=False。
    """
    model = WhisperWrapperFast(pretrained_model="breeze", lora_rank=8, head_type="bilstm")
    sd = {**torch.load(lora_path, map_location="cpu", weights_only=False),
          **torch.load(head_path, map_location="cpu", weights_only=False)}
    state = {k[len("model."):]: v for k, v in sd.items()
             if k.startswith("model.") and "embed_positions" not in k}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    n_lora = sum("lora_" in k for k in state)
    assert n_lora == 128, f"expect 128 LoRA keys, got {n_lora}"
    # 新版 transformers 需要 WhisperAttention.config(複製自 loo_cv build_model 的 patch)
    _backbone_cfg = model.backbone_model.config
    for _layer in model.backbone_model.encoder.layers:
        if hasattr(_layer, 'self_attn') and getattr(_layer.self_attn, 'config', None) is None:
            _layer.self_attn.config = _backbone_cfg
    return model.cuda().eval()


def run_inference(mp3_path, whisper_model, progress_cb=None):
    """整段音檔 → (child 時間戳, adult 時間戳, 總長秒數)。逐字複製 + 尾端 padding + 進度回呼。"""
    audio, sr = torchaudio.load(mp3_path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
    audio = audio.float()
    true_length = audio.size(1) / SAMPLE_RATE
    # 增修:尾端 zero-pad 到視窗倍數,讓最後 <10s 也被處理(結束後把時間戳夾回真實長度)
    win_samples = WINDOW_SIZE * SAMPLE_RATE
    if audio.size(1) % win_samples:
        audio = torch.nn.functional.pad(audio, (0, win_samples - audio.size(1) % win_samples))
    length = audio.size(1) / SAMPLE_RATE
    child_pred, adult_pred = [], []
    start = 0.0
    n_windows = int(length // WINDOW_SIZE)
    with torch.no_grad():
        while start + WINDOW_SIZE <= length:
            window = audio[:, int(start*SAMPLE_RATE):int((start+WINDOW_SIZE)*SAMPLE_RATE)]
            pred = whisper_model.forward_eval(window)
            pred = majority_filter(pred)
            child, adult, _ = get_timestamps(pred)
            child_pred += [(start+s, start+e) for s, e in child]
            adult_pred += [(start+s, start+e) for s, e in adult]
            start += WINDOW_SIZE
            if progress_cb:
                progress_cb(int(start // WINDOW_SIZE), n_windows)

    def merge(ivs):
        merged = []
        for s, e in sorted(ivs):
            if merged and s - merged[-1][1] <= 0.05:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        return merged

    def clip(ivs):  # 夾回真實長度、丟掉落在 padding 裡的空段
        return [(s, min(e, true_length)) for s, e in ivs if s < true_length]

    return clip(merge(child_pred)), clip(merge(adult_pred)), true_length


if __name__ == "__main__":
    path = sys.argv[1]
    print("Loading diarization model...")
    m = load_diar_model()
    print("Running inference on", path)
    child, adult, dur = run_inference(path, m,
                                      progress_cb=lambda i, n: print(f"\r{i}/{n}", end=""))
    print(f"\nduration={dur:.1f}s  child segs={len(child)}  adult segs={len(adult)}")
    for s, e in child[:10]:
        print(f"  child {s:8.2f} - {e:8.2f}")
