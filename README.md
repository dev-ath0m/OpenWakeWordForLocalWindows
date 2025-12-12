# OpenWakeWord Custom Training for Home Assistant

Complete environment for training custom wake word models on Windows with NVIDIA GPU support.

## Credits & License

This project is based on the excellent work from the original [OpenWakeWord Training Colab Notebook](https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb) created by [David Scripka](https://github.com/dscripka).

**Original Work**: [dscripka/openWakeWord](https://github.com/dscripka/openWakeWord)  
**License**: Apache License 2.0

### Modifications in This Repository

This repository adapts the original Colab notebook to:
- ✨ Run locally on Windows machines with NVIDIA GPU support
- 🎙️ Use multiple TTS voice models (6 models for positive samples, 2 for negative samples) for greater variety
- 🔄 Replace Piper with GPU-accelerated TTS models for both positive and phoneme-based adversarial negative sample generation
- 🚀 Provide automated workflow with interactive prompts
- 🔧 Include Windows-specific compatibility fixes and optimizations
- 📊 Add visual progress bars and enhanced user feedback
- 🧬 Use OpenWakeWord's phoneme-based adversarial text generation for intelligent negative samples

**Note**: Most of the code in this repository was generated using AI assistance to adapt the original workflow for Windows compatibility and automation.

All credits for the core OpenWakeWord framework and training methodology go to the original creator, David Scripka.

## Requirements

- **Python**: 3.11.x (required)
- **OS**: Windows 10/11
- **GPU**: NVIDIA GPU with CUDA 12.4 (recommended for faster training)
- **Disk Space**: ~10-20 GB (for models, datasets, and generated samples)

## Quick Start

### 1. Clone and Setup

```powershell
# Clone this repository
cd D:\Homeassistant

# Run setup script
.\SETUP_COMPLETE.ps1
```

The setup script will:
- ✅ Check Python 3.11.x installation
- ✅ Detect NVIDIA GPU and CUDA
- ✅ Create virtual environment
- ✅ Install PyTorch with CUDA 12.4 support
- ✅ Install all dependencies
- ✅ Clone OpenWakeWord repository
- ✅ Download required training features (ACAV100M, validation sets)
- ✅ Check for optional background datasets (audioset_16k, fma, mit_rirs)
- ✅ Install all dependencies with compatible versions
- ✅ Clone OpenWakeWord repository
- ✅ Create necessary directories
- ✅ Verify CUDA functionality

### 2. Activate Environment

```powershell
.\wakeword_env\Scripts\Activate.ps1
```

### 3. Train Your Wake Word

#### Option A: Automated Workflow (Recommended)

The automated script will guide you through the entire process with interactive prompts:

```powershell
python train_wakeword_automated.py
```

This will:
- ✅ Check your environment (Python version, GPU, dependencies)
- ✅ Prompt for wake word to train
- ✅ Allow multiple pronunciation variations with test sample playback
- ✅ Generate test samples and wait for your confirmation
- ✅ Ask for training parameters (samples, steps, false activation penalty) with suggested values
- ✅ Generate positive samples using 6 TTS models (tacotron2-DDC, glow-tts, fast_pitch, jenny, your_tts male/female)
- ✅ Generate phoneme-based adversarial negative samples using TTS (replaces Piper for Windows compatibility)
- ✅ Augment all samples with background noise, music, and reverb
- ✅ Train the model with progress tracking
- ✅ Export to ONNX (and optionally TFLite)
- ✅ Provide Home Assistant integration instructions

**Interactive Prompts:**
- Wake word: Default "homie"
- Pronunciation variations: Optional multiple pronunciations with test playback
- Number of samples: Default 3000 (Quick: 1000, High quality: 5000+)
- Training steps: Default 30000 (Quick: 10000, High quality: 50000+)
- False activation penalty: Default 1000 (Range: 10-5000+, higher = stricter)

**Sample Generation:**
- Positive samples: 6 diverse TTS models create variations of your wake word
- Negative samples: Phoneme-based adversarial phrases (e.g., for "homie": "homy", "foamy", "holy", "home me", etc.) using TTS instead of Piper
- All samples: Augmented with real-world noise, music, and room acoustics

#### Option B: Quick Test (1000 samples, 10000 steps, ~30-60 minutes)

```powershell
cd openwakeword
python openwakeword\train.py --training_config ..\test_train.yaml --generate_clips --augment_clips --train_model
```

#### Option C: Manual Full Training (3000 samples, 30000 steps, ~2-4 hours)

```powershell
python train_houwme_wakeword.py
```

## What Gets Installed

### Core Dependencies (from requirements.txt)

| Package | Version | Purpose |
|---------|---------|---------|
| **TTS** | >=0.22.0 | Coqui TTS for generating wake word samples |
| **torch** | >=2.0.0 (CUDA 12.4) | PyTorch for model training |
| **audiomentations** | 0.33.0 | Audio data augmentation (Windows-compatible) |
| **numpy** | <2.0, >=1.21 | Numerical computing (v1.26.4 tested) |
| **scipy** | >=1.13, <1.17 | Scientific computing (v1.16.3 tested) |
| **onnx** | 1.14.1 | Model export to ONNX format |
| **onnxruntime** | 1.22.1 | ONNX model runtime |
| **deep-phonemizer** | 0.0.19 | Phoneme generation for adversarial samples |

Full list in `requirements.txt`

### Windows-Specific Fixes Applied

The setup script automatically patches OpenWakeWord's code for Windows compatibility. These patches are applied during `SETUP_COMPLETE.ps1` execution:

#### 1. **Piper Dependency Removal** (`train.py`)
- **Issue**: Original code requires Piper TTS which has complex dependencies
- **Fix**: Patch `train.py` to make Piper import optional
- **Impact**: Allows training to work without Piper installation

#### 2. **Memory-Mapped File Cleanup** (`utils.py`)
- **Issue**: Windows file locking prevents deletion of memory-mapped numpy files
- **Fix**: Add explicit cleanup before `trim_mmap()`:
  ```python
  # Close memory-mapped file before trimming (Windows requires this)
  del fp
  import gc
  gc.collect()
  ```
- **Location**: `openwakeword/openwakeword/utils.py` ~line 600
- **Impact**: Prevents `PermissionError: [WinError 32]` during feature augmentation

#### 3. **Memory-Mapped File Deletion** (`data.py`)
- **Issue**: Windows file locking prevents `os.remove()` on open memory-mapped files
- **Fix**: Add explicit cleanup before `os.remove()`:
  ```python
  # Close memory-mapped files before deleting (Windows requires this)
  del mmap_file1
  del mmap_file2
  import gc
  gc.collect()
  ```
- **Location**: `openwakeword/openwakeword/data.py` ~line 889
- **Impact**: Prevents file permission errors during data trimming

#### 4. **Multiprocessing Pickle Errors** (`train.py`)
- **Issue**: Lambda functions can't be pickled for Windows multiprocessing
- **Fix**: Replace lambda functions with named functions:
  ```python
  # OLD: label_transforms[key] = lambda x: [1 for i in x]
  # NEW:
  def positive_label_transform(x):
      return [1 for i in x]
  
  def negative_label_transform(x):
      return [0 for i in x]
  ```
- **Location**: `openwakeword/openwakeword/train.py` ~line 854-856
- **Impact**: Prevents `_pickle.PicklingError` during training

#### 5. **DataLoader Workers on Windows** (`train.py`)
- **Issue**: Windows multiprocessing doesn't work well with PyTorch DataLoader
- **Fix**: Platform-specific num_workers setting:
  ```python
  # On Windows, use num_workers=0 to avoid multiprocessing pickle errors
  import platform
  if platform.system() == "Windows":
      n_workers = 0
      prefetch = None
  else:
      n_workers = n_cpus//2
      prefetch = 16
  ```
- **Location**: `openwakeword/openwakeword/train.py` ~line 884
- **Impact**: Prevents DataLoader multiprocessing errors on Windows

### How Patches Are Applied

All patches are **automatically applied** by:
1. `SETUP_COMPLETE.ps1` during initial setup
2. `clone_openwakeword()` function in `train_wakeword_automated.py`

The patching functions check if changes are needed and apply them safely without breaking already-patched files.

## Project Structure

```
D:\Homeassistant\
├── wakeword_env/                    # Python virtual environment
├── openwakeword/                    # OpenWakeWord repository
├── clips/                           # Generated TTS samples (organized by voice/pronunciation)
├── my_custom_model/                 # Full training output
│   └── homie/
│       ├── homie.onnx              # Trained model (ONNX format)
│       └── homie.tflite            # Trained model (TFLite format, optional)
├── test_model/                      # Quick test output
│   └── homie_test.onnx             # Test model (200 KB)
├── audioset_16k/                    # OPTIONAL: Background noise dataset (speech, sounds)
├── fma/                             # OPTIONAL: Music dataset (Free Music Archive)
├── mit_rirs/                        # OPTIONAL: Room impulse responses (reverb simulation)
│
├── openwakeword_features_ACAV100M_2000_hrs_16bit.npy  # Required: Downloaded by setup (~4.7GB)
├── validation_set_features.npy                         # Required: Downloaded by setup (~56MB)
│
├── requirements.txt                 # Python dependencies
├── SETUP_COMPLETE.ps1              # Automated setup script
├── train_wakeword_automated.py     # Automated training workflow
├── generate_samples_coqui.py       # TTS sample generation (legacy)
├── test_train.yaml                 # Test training configuration
├── my_model.yaml                   # Full training configuration
└── README.md                       # This file
```

## Optional Background Datasets (For Improved Model Quality)

The training process can use optional background datasets to improve model robustness. **These are NOT required** - training will work without them using synthetic augmentation only.

### Automatic Download Available

Both the setup script and training script will **offer to download** these datasets automatically:

| Dataset | Purpose | Size | Auto-Download |
|---------|---------|------|---------------|
| **mit_rirs** | Room impulse responses (reverb simulation) | ~50 MB (271 files) | ✅ Yes |
| **mit_environmental** | MIT Environmental Impulse Responses (HuggingFace) | ~300 MB | ✅ Yes |
| **fma** | Background music (Free Music Archive) | 7.2 GB - 22 GB | ✅ Yes |
| **audioset_16k** | Background noise (speech, environmental sounds) | 20-50 GB (~40K files) | ⚠️ Yes (6-24 hours) |

### How It Works

**During Setup (SETUP_COMPLETE.ps1):**
- Script checks if datasets are present
- Offers to download MIT RIRs (~50MB) - direct from MIT
- Offers to download MIT Environmental dataset (~300MB) - from HuggingFace (used in original Colab)
- Offers choice between FMA small (7.2GB) or medium (22GB)
- Offers AudioSet Balanced+Eval download (20-50GB, requires yt-dlp)

**During Training (train_wakeword_automated.py):**
- Same automatic download options
- Downloads include progress bars
- Graceful fallback if downloads fail
- AudioSet download can be interrupted and resumed

### What Each Dataset Does

**MIT RIRs (Room Impulse Responses):**
- **Purpose**: Simulates different room acoustics and reverb (271 files from various spaces)
- **Impact**: Makes model robust to different environments (bathroom, living room, etc.)
- **Source**: Direct download from MIT McDermott Lab
- **Recommendation**: Quick download, worth having

**MIT Environmental (HuggingFace dataset):**
- **Purpose**: Additional environmental impulse responses (used in original Colab training)
- **Impact**: Provides even more reverb variations for training
- **Source**: HuggingFace dataset `davidscripka/MIT_environmental_impulse_responses`
- **Recommendation**: Download for maximum authenticity to original training process

**FMA (Free Music Archive):**
- **Purpose**: Background music for training robustness
- **Impact**: Reduces false triggers when music is playing
- **Recommendation**: fma_small (7.2GB) is sufficient for most cases

**AudioSet (Balanced + Eval subsets):**
- **Purpose**: Real-world background noise (speech, traffic, nature, music, etc.)
- **Impact**: Significantly improves robustness to environmental noise
- **Recommendation**: 
  - ✅ **Download if**: You have time (6-24 hours) and want maximum quality
  - ⚠️ **Requirements**: yt-dlp (auto-installed), ffmpeg (manual install), stable internet
  - ℹ️ **Note**: Downloads ~40,000 clips from YouTube (some may be unavailable)

### AudioSet Download Details

AudioSet provides the highest quality background noise dataset with 527 different sound categories. The automatic download:

1. **Downloads metadata** (CSV files with YouTube video IDs)
2. **Installs yt-dlp** (if not already installed)
3. **Requires ffmpeg** (must be installed manually - see setup messages)
4. **Downloads audio** from YouTube (Balanced + Eval = ~40,000 clips)
5. **Extracts segments** (10-second clips at 16kHz mono)

**Expected Results:**
- Success rate: 70-90% (some videos unavailable/region-locked)
- Final dataset: ~25,000-35,000 audio clips
- Size: 20-50 GB depending on success rate
- Time: 6-24 hours depending on internet speed

**Manual AudioSet Download:**
You can also run the download script manually:
```bash
# Full download
python download_audioset.py

# Test mode (10 samples only)
python download_audioset.py --test
```

Check `audioset_download.log` for detailed progress.

### When to Download

**Download them if:**
- You want **highest quality** models
- Your wake word will be used in **noisy environments**
- You have **disk space** (30-70 GB for FMA + AudioSet)
- You want **maximum robustness** to music/background sounds
- You have **time** for AudioSet download (6-24 hours)

**Skip them if:**
- You're doing a **quick test**
- **Disk space** is limited
- You want **faster setup**
- Your environment is **relatively quiet**

### Manual Download (Optional)

If automatic download fails, you can download manually:

1. **MIT Room Impulse Responses**
   - Download: https://mcdermottlab.mit.edu/Reverb/IR_Survey.html
   - Extract WAV files to `mit_rirs/` directory

2. **FMA (Free Music Archive)**
   - Download: https://github.com/mdeff/fma
   - Choose fma_small.zip (7.2 GB) or fma_medium.zip (22 GB)
   - Extract to `fma/` directory

3. **AudioSet**
   - Info: https://research.google.com/audioset/download.html
   - Requires downloading YouTube videos and extracting audio
   - Complex process, skip unless absolutely needed

## Training Configuration

### Test Configuration (`test_train.yaml`)
- **Samples**: 1000 positive + negative adversarial
- **Training Steps**: 10,000
- **Time**: ~30-60 minutes on GPU
- **Output**: `test_model/homie_test.onnx`

### Full Configuration (`my_model.yaml`)
- **Samples**: 3000 positive + negative adversarial  
- **Training Steps**: 30,000
- **Time**: ~2-4 hours on GPU
- **Output**: `my_custom_model/homie/homie.onnx`

## Generated Files

### Model Files (Ready for Home Assistant)

- **ONNX Format**: `homie_test.onnx` or `homie.onnx` (~200 KB)
  - ✅ Fully compatible with OpenWakeWord
  - ✅ Works with Home Assistant Wyoming protocol
  - ✅ Recommended format

- **TFLite Format**: `homie_test.tflite` or `homie.tflite` (optional)
  - ⚠️ Requires additional TensorFlow dependencies
  - ⚠️ Has compatibility issues on some systems
  - ℹ️ ONNX is sufficient for Home Assistant

## Using with Home Assistant

### 1. Copy Model to Home Assistant

```powershell
# Copy the trained model
Copy-Item "test_model\homie_test.onnx" "\\homeassistant\config\custom_wakewords\homie.onnx"
```

### 2. Configure Wyoming OpenWakeWord

Add to your Home Assistant configuration:

```yaml
# configuration.yaml
wyoming:
  - uri: tcp://localhost:10400
    models:
      - homie  # Your custom wake word
```

### 3. Test the Wake Word

Say "Homie" to activate your Home Assistant voice assistant!

## Automated Workflow Features

The `train_wakeword_automated.py` script provides:

### Environment Validation
- Python 3.11.x version check
- Virtual environment verification
- NVIDIA GPU detection and CUDA availability
- Dependency installation status
- Automatic installation of missing packages

### Interactive Configuration
- Wake word input with validation
- Test sample generation and playback
- User confirmation before proceeding
- Training parameter selection with recommended values
- Configuration summary before training starts

### Progress Tracking
- Colored terminal output for better visibility
- Step-by-step progress indicators
- Success/warning/error messages with context
- Estimated time for each phase
- Final summary with model locations

### Error Handling
- Graceful failure recovery
- User prompts to continue or abort
- Detailed error messages
- Option to skip failed steps when safe

### Output
- ONNX model (primary, fully compatible)
- TFLite model (optional, attempts conversion)
- Training configuration saved for future reference
- Home Assistant integration instructions

## Troubleshooting

### Python Version Issues

```powershell
# Check Python version
python --version  # Should be 3.11.x

# If wrong version, install Python 3.11 from python.org
```

### CUDA Not Detected

```powershell
# Check NVIDIA driver
nvidia-smi

# If not working, update NVIDIA drivers
# Download from: https://www.nvidia.com/Download/index.aspx
```

### Dependency Conflicts

```powershell
# Clean install
Remove-Item -Recurse -Force wakeword_env
.\SETUP_COMPLETE.ps1
```

### Training Errors

1. **File Locking Issues**: Fixed automatically (trim skipped on Windows)
2. **Multiprocessing Errors**: Fixed automatically (num_workers=0)
3. **Unicode Errors**: Fixed automatically (ASCII filtering)
4. **Out of Memory**: Reduce `n_samples` in configuration file

## Performance Notes

| Hardware | Quick Test (1000/10000) | Full Training (3000/30000) |
|----------|------------------------|----------------------------|
| **RTX A3000 Laptop GPU** | ~30-60 min | ~2-4 hours |
| **CPU Only** | ~2-3 hours | ~8-12 hours |

## Tested Configuration

- ✅ Windows 11
- ✅ Python 3.11.0
- ✅ NVIDIA RTX A3000 Laptop GPU
- ✅ CUDA 12.4
- ✅ PyTorch 2.6.0+cu124
- ✅ audiomentations 0.33.0
- ✅ numpy 1.26.4
- ✅ scipy 1.16.3

## Additional Resources

- [OpenWakeWord GitHub](https://github.com/dscripka/openwakeword)
- [Home Assistant Voice](https://www.home-assistant.io/voice_control/)
- [Coqui TTS](https://github.com/coqui-ai/TTS)

## License

This project uses:
- OpenWakeWord (Apache 2.0)
- Coqui TTS (MPL 2.0)
- Home Assistant (Apache 2.0)

## Support

For issues:
1. Check this README
2. Review error messages in terminal
3. Check OpenWakeWord documentation
4. File issue with full error log

---

**Training completed successfully?** 🎉

Your `homie_test.onnx` model is ready for Home Assistant!
