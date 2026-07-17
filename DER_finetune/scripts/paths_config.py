"""Central, portable path config for diarization scripts.

Design goal: point ASD_DATA_ROOT at wherever you keep data/, whisper-modeling/,
Talkbank/ on your machine; everything else is derived from it.

    ASD_DATA_ROOT   root holding data/, whisper-modeling/, Talkbank/
                    (default: "data", relative to current working directory)

Only NEW / portable scripts import this; older experiment scripts keep their
own hard-coded paths (set ASD_DATA_ROOT / edit the constants there directly).
"""
import os

# Single root that almost every path lives under.
DATA_ROOT = os.environ.get("ASD_DATA_ROOT", "data")

# ── data ──────────────────────────────────────────────────────────────────────
FINE_TUNE_DIR   = os.path.join(DATA_ROOT, "data", "Fine_tune")        # ASD sessions
RESULTS_ROOT    = os.path.join(DATA_ROOT, "data")                     # loo_results_* live here
ANALYSIS_CACHE  = os.path.join(DATA_ROOT, "data", "analysis_cache")
ANALYSIS_OUTPUT = os.path.join(DATA_ROOT, "data", "analysis_output")

# ── model / checkpoints ───────────────────────────────────────────────────────
WHISPER_MODELING_DIR = os.path.join(DATA_ROOT, "whisper-modeling")
CHECKPOINT_DIR       = os.path.join(WHISPER_MODELING_DIR, "checkpoints")
AUDIOSET_PRETRAINED  = os.path.join(
    CHECKPOINT_DIR, "whisper_base_5k", "epoch=17-val_loss=0.113.ckpt")

# ── ChangPN (TalkBank) — only needed by ChangPN-pretrain experiments ──────────
CHANGPN_AUDIO_DIR      = os.path.join(DATA_ROOT, "Talkbank", "ChangPN", "audio")
CHANGPN_TRANSCRIPT_DIR = os.path.join(DATA_ROOT, "Talkbank", "ChangPN", "transcript")


def results_dir(name: str) -> str:
    """Path to a loo_results_* directory under the data root."""
    return os.path.join(RESULTS_ROOT, name)


if __name__ == "__main__":
    # Quick sanity dump: shows what the paths resolve to in the current env.
    for k, v in sorted(globals().items()):
        if k.isupper():
            print(f"{k:20s} = {v}")
