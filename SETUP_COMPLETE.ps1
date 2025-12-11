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
    # Show progress bar during venv creation
    $job = Start-Job -ScriptBlock { param($pythonPath) & $pythonPath -m venv wakeword_env } -ArgumentList (Get-Command python).Source
    
    $i = 0
    while ($job.State -eq 'Running') {
        $progress = ($i % 10) + 1
        Write-Progress -Activity "Creating virtual environment" -Status "Setting up Python environment..." -PercentComplete ($progress * 10)
        Start-Sleep -Milliseconds 500
        $i++
    }
    
    Wait-Job $job | Out-Null
    Receive-Job $job | Out-Null
    Remove-Job $job
    Write-Progress -Activity "Creating virtual environment" -Completed
    
    Write-Host "[OK] Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "`nActivating virtual environment..." -ForegroundColor Yellow
& .\wakeword_env\Scripts\Activate.ps1

# Upgrade pip
Write-Host "`nUpgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip wheel setuptools --quiet
Write-Host "[OK] pip, wheel, and setuptools upgraded" -ForegroundColor Green

# Install PyTorch with CUDA support
Write-Host "`nInstalling PyTorch with CUDA 12.4..." -ForegroundColor Yellow
Write-Host "This may take several minutes (downloading ~2GB)..." -ForegroundColor Gray

$job = Start-Job -ScriptBlock {
    & pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --quiet
}

$i = 0
while ($job.State -eq 'Running') {
    $progress = ($i % 9) * 10 + 10
    Write-Progress -Activity "Installing PyTorch" -Status "Downloading and installing (~2GB)..." -PercentComplete $progress
    Start-Sleep -Seconds 2
    $i++
}

Wait-Job $job | Out-Null
$jobOutput = Receive-Job $job
Remove-Job $job
Write-Progress -Activity "Installing PyTorch" -Completed

if ($LASTEXITCODE -eq 0 -or -not $LASTEXITCODE) {
    Write-Host "[OK] PyTorch with CUDA 12.4 installed" -ForegroundColor Green
} else {
    Write-Host "[ERROR] PyTorch installation failed" -ForegroundColor Red
    Write-Host $jobOutput -ForegroundColor Red
    exit 1
}

# Install other dependencies
Write-Host "`nInstalling dependencies from requirements.txt..." -ForegroundColor Yellow
Write-Host "This may take 5-10 minutes..." -ForegroundColor Gray

$job = Start-Job -ScriptBlock {
    & pip install -r requirements.txt --quiet
}

$i = 0
while ($job.State -eq 'Running') {
    $progress = ($i % 9) * 10 + 10
    Write-Progress -Activity "Installing Dependencies" -Status "Installing Python packages from requirements.txt..." -PercentComplete $progress
    Start-Sleep -Seconds 2
    $i++
}

Wait-Job $job | Out-Null
$jobOutput = Receive-Job $job
Remove-Job $job
Write-Progress -Activity "Installing Dependencies" -Completed

if ($LASTEXITCODE -eq 0 -or -not $LASTEXITCODE) {
    Write-Host "[OK] All dependencies installed" -ForegroundColor Green
} else {
    Write-Host "[WARNING] Some dependencies may have failed to install" -ForegroundColor Yellow
    Write-Host $jobOutput -ForegroundColor Gray
}

# Clone OpenWakeWord repository if not exists
Write-Host "`nSetting up OpenWakeWord..." -ForegroundColor Yellow
if (Test-Path "openwakeword") {
    Write-Host "[OK] OpenWakeWord directory already exists" -ForegroundColor Green
} else {
    $job = Start-Job -ScriptBlock {
        git clone https://github.com/dscripka/openwakeword.git 2>&1
    }
    
    $i = 0
    while ($job.State -eq 'Running') {
        $progress = ($i % 9) * 10 + 10
        Write-Progress -Activity "Cloning OpenWakeWord" -Status "Downloading from GitHub..." -PercentComplete $progress
        Start-Sleep -Milliseconds 500
        $i++
    }
    
    Wait-Job $job | Out-Null
    Receive-Job $job | Out-Null
    Remove-Job $job
    Write-Progress -Activity "Cloning OpenWakeWord" -Completed
    
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
Write-Host "2. Run automated workflow: python train_wakeword_automated.py" -ForegroundColor White
Write-Host ""
Write-Host "The automated script will:" -ForegroundColor Yellow
Write-Host "  - Guide you through wake word configuration" -ForegroundColor Gray
Write-Host "  - Let you add custom pronunciations" -ForegroundColor Gray
Write-Host "  - Generate test samples for verification" -ForegroundColor Gray
Write-Host "  - Generate training samples using multiple TTS models" -ForegroundColor Gray
Write-Host "  - Augment samples with noise and reverb" -ForegroundColor Gray
Write-Host "  - Train the model with GPU acceleration" -ForegroundColor Gray
Write-Host "  - Export ONNX model for Home Assistant" -ForegroundColor Gray
Write-Host ""
Write-Host "All dependencies and TTS models will download automatically!" -ForegroundColor Green
Write-Host ""
