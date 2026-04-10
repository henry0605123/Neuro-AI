# 快速安裝套件腳本
# 執行方式: .\install_dependencies.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Fine-tune Pipeline 套件安裝" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 檢查 conda 環境
if ($env:CONDA_DEFAULT_ENV -ne "ASD") {
    Write-Host "⚠️  請先啟動 ASD 環境: conda activate ASD" -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ 環境: $env:CONDA_DEFAULT_ENV" -ForegroundColor Green
Write-Host ""

# 安裝 pyannote.audio
Write-Host "[1/6] 安裝 pyannote.audio..." -ForegroundColor Yellow
pip install pyannote.audio --quiet

# 安裝 WhisperX
Write-Host "[2/6] 安裝 WhisperX..." -ForegroundColor Yellow
pip install git+https://github.com/m-bain/whisperx.git --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "  從 GitHub 安裝失敗，嘗試 PyPI..." -ForegroundColor Yellow
    pip install whisperx --quiet
}

# 安裝音訊處理工具
Write-Host "[3/6] 安裝音訊處理工具..." -ForegroundColor Yellow
pip install pydub ffmpeg-python --quiet

# 安裝資料處理工具
Write-Host "[4/6] 安裝資料處理工具..." -ForegroundColor Yellow
pip install pandas pyyaml psutil --quiet

# 安裝訓練工具
Write-Host "[5/6] 安裝訓練工具..." -ForegroundColor Yellow
pip install lightning tensorboard --quiet

# 驗證安裝
Write-Host "[6/6] 驗證安裝..." -ForegroundColor Yellow
Write-Host ""

$allOk = $true

# 測試 PyTorch
try {
    $torchVersion = python -c "import torch; print(torch.__version__)" 2>&1
    Write-Host "  ✓ PyTorch: $torchVersion" -ForegroundColor Green
} catch {
    Write-Host "  ✗ PyTorch: 安裝失敗" -ForegroundColor Red
    $allOk = $false
}

# 測試 WhisperX
try {
    python -c "import whisperx" 2>&1 | Out-Null
    Write-Host "  ✓ WhisperX: OK" -ForegroundColor Green
} catch {
    Write-Host "  ✗ WhisperX: 安裝失敗" -ForegroundColor Red
    $allOk = $false
}

# 測試 Pyannote
try {
    python -c "import pyannote.audio" 2>&1 | Out-Null
    Write-Host "  ✓ Pyannote: OK" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Pyannote: 安裝失敗" -ForegroundColor Red
    $allOk = $false
}

# 測試 Lightning
try {
    python -c "import lightning" 2>&1 | Out-Null
    Write-Host "  ✓ Lightning: OK" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Lightning: 安裝失敗" -ForegroundColor Red
    $allOk = $false
}

Write-Host ""
if ($allOk) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  ✅ 所有套件安裝完成！" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "下一步:" -ForegroundColor Yellow
    Write-Host "  1. python test_performance.py  # 測試硬體效能" -ForegroundColor White
    Write-Host "  2. python run_pipeline.py      # 執行完整流程" -ForegroundColor White
} else {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  ⚠️  部分套件安裝失敗" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "請參考 SETUP_GUIDE.md 手動安裝失敗的套件" -ForegroundColor Yellow
}
