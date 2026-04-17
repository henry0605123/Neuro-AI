#!/bin/bash
#SBATCH --job-name=playlogue-bilstm
#SBATCH --account=ACD114235
#SBATCH --partition=normal
#SBATCH --output=/home/cball3535/Neuro-AI/Fine_tune_LSTM/slurm_logs/%x-%j.out
#SBATCH --error=/home/cball3535/Neuro-AI/Fine_tune_LSTM/slurm_logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

# 可用 `sbatch --export=ALL,CONDA_ENV=ASD ...` 覆寫
CONDA_ENV="${CONDA_ENV:-playlogue-h100}"

# 依序嘗試常見 conda 初始化路徑
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "[ERROR] 找不到 conda。請先 module load 對應環境，或安裝 Miniconda/Anaconda。" >&2
  exit 1
fi

conda activate "$CONDA_ENV"

# 降低記憶體碎片風險（PyTorch 官方建議）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/cball3535/Neuro-AI/Fine_tune_LSTM/code
mkdir -p /home/cball3535/Neuro-AI/Fine_tune_LSTM/slurm_logs

# H100 上建議直接使用 AMP；預設 config.py 已指向 Playlogue split。
python main.py pipeline \
  --model-size large \
  --epochs 30 \
  --batch-size 16 \
  --lr 1e-4 \
  --lstm-hidden 128 \
  --lstm-layers 2 \
  --use-lora
