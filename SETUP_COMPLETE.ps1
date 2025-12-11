# OpenWakeWord Custom Training Environment Setup
# Windows PowerShell Script
# Requires Python 3.11.x and NVIDIA GPU with CUDA 12.4

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "OpenWakeWord Training Environment Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($pythonVersion -match "Python 3\.11\.") {
    Write-Host "[OK] $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Python 3.11.x is required. Found: $pythonVersion" -ForegroundColor Red
    exit 1
}

# Check NVIDIA GPU
Write-Host "`nChecking for NVIDIA GPU..." -ForegroundColor Yellow
$gpu = nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv,noheader 2>$null
if ($gpu) {
    Write-Host "[OK] GPU detected: $gpu" -ForegroundColor Green
} else {
    Write-Host "[WARNING] No NVIDIA GPU detected. Training will use CPU (slower)" -ForegroundColor Yellow
}

# Create virtual environment
Write-Host "`nCreating virtual environment..." -ForegroundColor Yellow
if (Test-Path "wakeword_env") {
    Write-Host "[OK] Virtual environment already exists" -ForegroundColor Green
} else {
    python -m venv wakeword_env
    Write-Host "[OK] Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "`nActivating virtual environment..." -ForegroundColor Yellow
& .\wakeword_env\Scripts\Activate.ps1

# Upgrade pip
Write-Host "`nUpgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA support
Write-Host "`nInstalling PyTorch with CUDA 12.4..." -ForegroundColor Yellow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
Write-Host "`nInstalling dependencies from requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt

# Clone OpenWakeWord repository if not exists
Write-Host "`nSetting up OpenWakeWord..." -ForegroundColor Yellow
if (Test-Path "openwakeword") {
    Write-Host "[OK] OpenWakeWord directory already exists" -ForegroundColor Green
} else {
    git clone https://github.com/dscripka/openwakeword.git
    Write-Host "[OK] OpenWakeWord cloned" -ForegroundColor Green
}

# Create necessary directories
Write-Host "`nCreating directories..." -ForegroundColor Yellow
@("clips", "audioset_16k", "fma", "mit_rirs", "my_custom_model", "test_model", "piper-sample-generator") | ForEach-Object {
    if (-not (Test-Path $_)) {
        New-Item -ItemType Directory -Path $_ | Out-Null
        Write-Host "  Created: $_" -ForegroundColor Gray
    }
}
Write-Host "[OK] Directories ready" -ForegroundColor Green

# Test PyTorch CUDA
Write-Host "`nTesting PyTorch CUDA support..." -ForegroundColor Yellow
$cudaTest = python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); print(f'PyTorch version: {torch.__version__}')" 2>&1
Write-Host $cudaTest -ForegroundColor Gray

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Activate environment: .\wakeword_env\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "2. Generate samples: python generate_samples_coqui.py" -ForegroundColor White
Write-Host "3. Train model: python train_houwme_wakeword.py" -ForegroundColor White
Write-Host ""
Write-Host "Or run a quick test (1000 samples, 10000 steps):" -ForegroundColor Yellow
Write-Host "  cd openwakeword" -ForegroundColor White
Write-Host "  python openwakeword\train.py --training_config ..\test_train.yaml --generate_clips --augment_clips --train_model" -ForegroundColor White
Write-Host ""
Write-Host "Files created:" -ForegroundColor Yellow
Write-Host "  - Model ONNX: test_model\homie_test.onnx (for Home Assistant)" -ForegroundColor White
Write-Host ""
