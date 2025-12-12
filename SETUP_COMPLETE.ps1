# OpenWakeWord Custom Training Environment Setup
# Windows PowerShell Script
# Requires Python 3.11.x and NVIDIA GPU with CUDA 12.4

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "OpenWakeWord Training Environment Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check and install Python 3.11.x
Write-Host "Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($pythonVersion -match "Python 3\.11\.") {
    Write-Host "[OK] $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "[WARNING] Python 3.11.x not found. Current: $pythonVersion" -ForegroundColor Yellow
    Write-Host "Installing Python 3.11.9 via winget..." -ForegroundColor Yellow
    
    # Check if winget is available
    try {
        $wingetVersion = winget --version 2>&1
        Write-Host "  winget version: $wingetVersion" -ForegroundColor Gray
    } catch {
        Write-Host "[ERROR] winget not found. Please install App Installer from Microsoft Store." -ForegroundColor Red
        Write-Host "  Or download Python 3.11.9 from https://www.python.org/downloads/" -ForegroundColor Red
        exit 1
    }
    
    # Install Python 3.11.9
    Write-Host "  Downloading and installing Python 3.11.9..." -ForegroundColor Gray
    winget install Python.Python.3.11 --version 3.11.9 --silent --accept-package-agreements --accept-source-agreements
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Python 3.11.9 installed successfully" -ForegroundColor Green
        
        # Refresh environment variables
        Write-Host "  Refreshing PATH environment variable..." -ForegroundColor Gray
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        
        # Verify installation
        Start-Sleep -Seconds 2
        $pythonVersion = python --version 2>&1
        if ($pythonVersion -match "Python 3\.11\.") {
            Write-Host "[OK] Verified: $pythonVersion" -ForegroundColor Green
        } else {
            Write-Host "[WARNING] Python installed but may require terminal restart" -ForegroundColor Yellow
            Write-Host "  Please close this terminal and run the script again" -ForegroundColor Yellow
            exit 0
        }
    } else {
        Write-Host "[ERROR] Python installation failed. Exit code: $LASTEXITCODE" -ForegroundColor Red
        Write-Host "  Please manually install Python 3.11.9 from https://www.python.org/downloads/" -ForegroundColor Red
        exit 1
    }
}

# Check NVIDIA GPU
Write-Host "`nChecking for NVIDIA GPU..." -ForegroundColor Yellow
$gpu = nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv,noheader 2>$null
if ($gpu) {
    Write-Host "[OK] GPU detected: $gpu" -ForegroundColor Green
} else {
    Write-Host "[WARNING] No NVIDIA GPU detected. Training will use CPU (slower)" -ForegroundColor Yellow
}

# Check and install ffmpeg (required for AudioSet download and audio processing)
Write-Host "`nChecking for ffmpeg..." -ForegroundColor Yellow
$ffmpegInstalled = $false
try {
    $ffmpegVersion = ffmpeg -version 2>$null
    if ($ffmpegVersion) {
        Write-Host "[OK] ffmpeg is already installed" -ForegroundColor Green
        $ffmpegInstalled = $true
    }
} catch {
    # ffmpeg not found
}

if (-not $ffmpegInstalled) {
    Write-Host "[INFO] ffmpeg not found - required for AudioSet download and audio processing" -ForegroundColor Yellow
    Write-Host "Attempting to install ffmpeg using winget..." -ForegroundColor Gray
    
    try {
        # Check if winget is available
        $wingetVersion = winget --version 2>$null
        if ($wingetVersion) {
            Write-Host "Installing ffmpeg via winget..." -ForegroundColor Gray
            winget install --id Gyan.FFmpeg --silent --accept-source-agreements --accept-package-agreements 2>&1 | Out-Null
            
            # Refresh PATH to pick up new installation
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
            
            # Verify installation
            $ffmpegVersion = ffmpeg -version 2>$null
            if ($ffmpegVersion) {
                Write-Host "[OK] ffmpeg installed successfully via winget" -ForegroundColor Green
                $ffmpegInstalled = $true
            } else {
                Write-Host "[WARNING] ffmpeg installation may require shell restart" -ForegroundColor Yellow
            }
        } else {
            throw "winget not available"
        }
    } catch {
        Write-Host "[WARNING] Could not auto-install ffmpeg" -ForegroundColor Yellow
        Write-Host "Please install ffmpeg manually:" -ForegroundColor Yellow
        Write-Host "  Option 1 (Recommended): winget install ffmpeg" -ForegroundColor Gray
        Write-Host "  Option 2: Download from https://www.gyan.dev/ffmpeg/builds/" -ForegroundColor Gray
        Write-Host "            Extract and add to PATH" -ForegroundColor Gray
        Write-Host "" -ForegroundColor Gray
        Write-Host "Note: ffmpeg is only required for AudioSet download (optional)" -ForegroundColor DarkGray
    }
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

# Download training features (ACAV100M and validation)
Write-Host "`nDownloading training features..." -ForegroundColor Yellow

$acav100mFile = "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
$validationFile = "validation_set_features.npy"
$downloadErrors = @()

if (-not (Test-Path $acav100mFile)) {
    Write-Host "  Downloading ACAV100M features (~4.7GB)..." -ForegroundColor Gray
    Write-Host "  This may take 10-30 minutes depending on your connection" -ForegroundColor Gray
    
    $acav100mUrl = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    
    try {
        $ProgressPreference = 'SilentlyContinue'  # Disable progress bar for faster download
        Invoke-WebRequest -Uri $acav100mUrl -OutFile $acav100mFile -UseBasicParsing
        $ProgressPreference = 'Continue'
        
        # Verify file was downloaded and is not empty
        if ((Test-Path $acav100mFile) -and ((Get-Item $acav100mFile).Length -gt 0)) {
            Write-Host "  [OK] ACAV100M features downloaded" -ForegroundColor Green
        } else {
            throw "Downloaded file is empty or invalid"
        }
    } catch {
        Write-Host "  [ERROR] Failed to download ACAV100M features: $_" -ForegroundColor Red
        if (Test-Path $acav100mFile) {
            Remove-Item $acav100mFile -Force  # Remove partial/corrupted download
        }
        $downloadErrors += "ACAV100M features: $_"
    }
} else {
    Write-Host "  [OK] ACAV100M features already present" -ForegroundColor Green
}

if (-not (Test-Path $validationFile)) {
    Write-Host "  Downloading validation features (~56MB)..." -ForegroundColor Gray
    
    $validationUrl = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy"
    
    try {
        $ProgressPreference = 'SilentlyContinue'  # Disable progress bar for faster download
        Invoke-WebRequest -Uri $validationUrl -OutFile $validationFile -UseBasicParsing
        $ProgressPreference = 'Continue'
        
        # Verify file was downloaded and is not empty
        if ((Test-Path $validationFile) -and ((Get-Item $validationFile).Length -gt 0)) {
            Write-Host "  [OK] Validation features downloaded" -ForegroundColor Green
        } else {
            throw "Downloaded file is empty or invalid"
        }
    } catch {
        Write-Host "  [ERROR] Failed to download validation features: $_" -ForegroundColor Red
        if (Test-Path $validationFile) {
            Remove-Item $validationFile -Force  # Remove partial/corrupted download
        }
        $downloadErrors += "Validation features: $_"
    }
} else {
    Write-Host "  [OK] Validation features already present" -ForegroundColor Green
}

# Download OpenWakeWord ONNX models from official GitHub release
Write-Host "`nDownloading OpenWakeWord ONNX models..." -ForegroundColor Yellow

$modelsDir = "openwakeword\openwakeword\resources\models"
$melspecPath = "$modelsDir\melspectrogram.onnx"
$embeddingPath = "$modelsDir\embedding_model.onnx"

# Create models directory if it doesn't exist
if (!(Test-Path $modelsDir)) {
    New-Item -ItemType Directory -Path $modelsDir -Force | Out-Null
}

$baseUrl = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"
$models = @{
    "melspectrogram.onnx" = "$baseUrl/melspectrogram.onnx"
    "embedding_model.onnx" = "$baseUrl/embedding_model.onnx"
}

foreach ($modelName in $models.Keys) {
    $modelPath = Join-Path $modelsDir $modelName
    $modelUrl = $models[$modelName]
    
    if (Test-Path $modelPath) {
        $sizeMB = [math]::Round((Get-Item $modelPath).Length / 1MB, 2)
        Write-Host "  [OK] $modelName already exists ($sizeMB MB)" -ForegroundColor Green
    } else {
        Write-Host "  Downloading $modelName..." -ForegroundColor Gray
        try {
            Invoke-WebRequest -Uri $modelUrl -OutFile $modelPath -UseBasicParsing
            $sizeMB = [math]::Round((Get-Item $modelPath).Length / 1MB, 2)
            Write-Host "  [OK] $modelName downloaded ($sizeMB MB)" -ForegroundColor Green
        } catch {
            Write-Host "  [ERROR] Failed to download $modelName`: $_" -ForegroundColor Red
            $downloadErrors += "ONNX model: $modelName"
        }
    }
}

# Check if downloads failed
if ($downloadErrors.Count -gt 0) {
    Write-Host "`n[ERROR] Failed to download required training features:" -ForegroundColor Red
    foreach ($error in $downloadErrors) {
        Write-Host "  - $error" -ForegroundColor Red
    }
    Write-Host "`nYou can try:" -ForegroundColor Yellow
    Write-Host "  1. Check your internet connection" -ForegroundColor Gray
    Write-Host "  2. Download manually from:" -ForegroundColor Gray
    Write-Host "     ACAV100M: $acav100mUrl" -ForegroundColor Gray
    Write-Host "     Validation: $validationUrl" -ForegroundColor Gray
    Write-Host "  3. Run the setup again to retry download" -ForegroundColor Gray
    Write-Host "`n[WARNING] Setup completed with errors - training may not work optimally" -ForegroundColor Yellow
} else {
    Write-Host "[OK] Training features ready" -ForegroundColor Green
}

# Download/check optional background datasets for improved model quality
Write-Host "`nChecking optional background datasets..." -ForegroundColor Yellow
Write-Host "These improve model quality but are optional (training works without them)" -ForegroundColor Gray

$datasetsToDownload = @()

# Check MIT RIRs (Room Impulse Responses) - 271 files, ~50MB (direct from MIT)
$mitRirsPath = "mit_rirs"
if (-not (Test-Path $mitRirsPath)) {
    New-Item -ItemType Directory -Path $mitRirsPath | Out-Null
}
$mitRirsCount = (Get-ChildItem -Path $mitRirsPath -Filter "*.wav" -ErrorAction SilentlyContinue | Measure-Object).Count
if ($mitRirsCount -lt 250) {
    Write-Host "  [!] MIT RIRs - missing or incomplete ($mitRirsCount/271 files)" -ForegroundColor Yellow
    $datasetsToDownload += "mit_rirs"
} else {
    Write-Host "  [OK] MIT RIRs - $mitRirsCount files" -ForegroundColor Green
}

# Check MIT Environmental (HuggingFace dataset - matches original Colab) - ~300MB
$mitEnvPath = "MIT_environmental_impulse_responses"
if (-not (Test-Path $mitEnvPath)) {
    New-Item -ItemType Directory -Path $mitEnvPath | Out-Null
}
$mitEnvCount = (Get-ChildItem -Path $mitEnvPath -Filter "*.wav" -ErrorAction SilentlyContinue | Measure-Object).Count
if ($mitEnvCount -lt 250) {
    Write-Host "  [!] MIT Environmental - missing or incomplete ($mitEnvCount files)" -ForegroundColor Yellow
    Write-Host "      (Will be offered during training - requires Python environment)" -ForegroundColor Gray
} else {
    Write-Host "  [OK] MIT Environmental - $mitEnvCount files" -ForegroundColor Green
}

# Check FMA (Free Music Archive) - recommend fma_small (8000 tracks, 7.2GB)
$fmaPath = "fma"
if (-not (Test-Path $fmaPath)) {
    New-Item -ItemType Directory -Path $fmaPath | Out-Null
}
$fmaCount = (Get-ChildItem -Path $fmaPath -Filter "*.mp3" -Recurse -ErrorAction SilentlyContinue | Measure-Object).Count
if ($fmaCount -lt 100) {
    Write-Host "  [!] FMA music - missing or incomplete ($fmaCount files)" -ForegroundColor Yellow
    $datasetsToDownload += "fma"
} else {
    Write-Host "  [OK] FMA music - $fmaCount tracks" -ForegroundColor Green
}

# Check AudioSet (requires manual download from YouTube)
$audiosetPath = "audioset_16k"
if (-not (Test-Path $audiosetPath)) {
    New-Item -ItemType Directory -Path $audiosetPath | Out-Null
}
$audiosetCount = (Get-ChildItem -Path $audiosetPath -Filter "*.wav" -Recurse -ErrorAction SilentlyContinue | Measure-Object).Count
if ($audiosetCount -lt 100) {
    Write-Host "  [!] AudioSet - missing or incomplete ($audiosetCount files)" -ForegroundColor Yellow
    Write-Host "      (Will be offered during training - requires Python environment)" -ForegroundColor Gray
} else {
    Write-Host "  [OK] AudioSet - $audiosetCount files" -ForegroundColor Green
}

# Offer to download missing datasets (excluding AudioSet which requires Python)
if ($datasetsToDownload.Count -gt 0) {
    Write-Host ""
    $downloadOptional = Read-Host "Download optional datasets now? This improves model quality. (y/N)"
    
    if ($downloadOptional -eq 'y' -or $downloadOptional -eq 'Y') {
        # Download MIT RIRs (small, quick download)
        if ($datasetsToDownload -contains "mit_rirs") {
            Write-Host "`n  Downloading MIT Room Impulse Responses (~50MB)..." -ForegroundColor Cyan
            try {
                $mitRirsUrl = "https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip"
                $mitRirsZip = "mit_rirs_temp.zip"
                $tempExtract = "mit_rirs_temp"
                
                $ProgressPreference = 'SilentlyContinue'
                Invoke-WebRequest -Uri $mitRirsUrl -OutFile $mitRirsZip -UseBasicParsing
                $ProgressPreference = 'Continue'
                
                Write-Host "    Extracting MIT RIRs..." -ForegroundColor Gray
                # Extract to temp directory first
                Expand-Archive -Path $mitRirsZip -DestinationPath $tempExtract -Force
                
                # Move WAV files from nested structure to mit_rirs root
                $wavFiles = Get-ChildItem -Path $tempExtract -Filter "*.wav" -Recurse
                foreach ($file in $wavFiles) {
                    Move-Item -Path $file.FullName -Destination $mitRirsPath -Force
                }
                
                # Clean up
                Remove-Item $mitRirsZip -Force
                Remove-Item $tempExtract -Recurse -Force
                
                $extractedCount = (Get-ChildItem -Path $mitRirsPath -Filter "*.wav" | Measure-Object).Count
                Write-Host "    [OK] MIT RIRs downloaded: $extractedCount files" -ForegroundColor Green
            } catch {
                Write-Host "    [ERROR] Failed to download MIT RIRs: $_" -ForegroundColor Red
                Write-Host "    You can download manually from: https://mcdermottlab.mit.edu/Reverb/IR_Survey.html" -ForegroundColor Yellow
            }
        }

                # Download FMA small (larger download, ~7.2GB)
        if ($datasetsToDownload -contains "fma") {
            Write-Host "`n  FMA (Free Music Archive) download options:" -ForegroundColor Cyan
            Write-Host "    1. fma_small (7.2 GB, 8,000 tracks) - Recommended" -ForegroundColor Gray
            Write-Host "    2. fma_medium (22 GB, 25,000 tracks)" -ForegroundColor Gray
            Write-Host "    3. Skip FMA download (you can add music files manually later)" -ForegroundColor Gray
            $fmaChoice = Read-Host "  Choose option (1/2/3)"
            
            if ($fmaChoice -eq '1' -or $fmaChoice -eq '2') {
                $fmaUrl = if ($fmaChoice -eq '1') { 
                    "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
                } else { 
                    "https://os.unil.cloud.switch.ch/fma/fma_medium.zip"
                }
                $fmaSize = if ($fmaChoice -eq '1') { "7.2 GB" } else { "22 GB" }
                
                Write-Host "`n  Downloading FMA ($fmaSize)..." -ForegroundColor Cyan
                Write-Host "  This will take 10-60 minutes depending on your connection" -ForegroundColor Yellow
                
                try {
                    $fmaZip = "fma_temp.zip"
                    
                    # Use background job for large download with progress
                    Write-Host "  Starting download (this may take a while)..." -ForegroundColor Gray
                    $ProgressPreference = 'SilentlyContinue'
                    Invoke-WebRequest -Uri $fmaUrl -OutFile $fmaZip -UseBasicParsing
                    $ProgressPreference = 'Continue'
                    
                    Write-Host "  Extracting FMA archive..." -ForegroundColor Gray
                    Expand-Archive -Path $fmaZip -DestinationPath $fmaPath -Force
                    Remove-Item $fmaZip -Force
                    
                    $extractedCount = (Get-ChildItem -Path $fmaPath -Filter "*.mp3" -Recurse | Measure-Object).Count
                    Write-Host "  [OK] FMA downloaded: $extractedCount tracks" -ForegroundColor Green
                } catch {
                    Write-Host "  [ERROR] Failed to download FMA: $_" -ForegroundColor Red
                    Write-Host "  You can download manually from: https://github.com/mdeff/fma" -ForegroundColor Yellow
                }
            } else {
                Write-Host "  Skipped FMA download" -ForegroundColor Gray
            }
        }
    } else {
        Write-Host "`n[INFO] Skipped optional dataset downloads" -ForegroundColor Cyan
        Write-Host "Training will use synthetic augmentation (still produces good models)" -ForegroundColor Gray
        Write-Host "You can download these datasets later to improve quality" -ForegroundColor Gray
    }
} else {
    Write-Host "[OK] All background datasets available for high-quality training" -ForegroundColor Green
}

# Test PyTorch CUDA
Write-Host "`nTesting PyTorch CUDA support..." -ForegroundColor Yellow
try {
    $cudaScript = @"
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA device:', torch.cuda.get_device_name(0))
else:
    print('CUDA device: None')
print('PyTorch version:', torch.__version__)
"@
    
    $cudaTest = python -c $cudaScript 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [ERROR] PyTorch test failed!" -ForegroundColor Red
        Write-Host $cudaTest -ForegroundColor Red
        Write-Host "`n[CRITICAL] Setup cannot continue - PyTorch is not working correctly" -ForegroundColor Red
        Write-Host "Please reinstall PyTorch with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124" -ForegroundColor Yellow
        exit 1
    }
    Write-Host $cudaTest -ForegroundColor Gray
} catch {
    Write-Host "  [ERROR] Failed to test PyTorch: $_" -ForegroundColor Red
    Write-Host "`n[CRITICAL] Setup cannot continue - Python environment may be corrupted" -ForegroundColor Red
    exit 1
}

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
