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

1. **Piper Replacement**: Uses GPU-accelerated TTS models (fast_pitch, glow-tts) instead of Piper for negative sample generation
2. **Phoneme-Based Adversarial Samples**: Leverages OpenWakeWord's `generate_adversarial_texts()` to create phonetically similar negative samples
3. **File Trimming Skip**: Disabled on Windows due to file locking issues
4. **Multiprocessing Disabled**: DataLoader uses `num_workers=0` to avoid pickling errors
5. **Unicode Handling**: ASCII filtering for console output compatibility
6. **Progress Visualization**: Real-time progress bars with sample statistics during generation

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
├── audioset_16k/                    # Background noise dataset (auto-downloaded)
├── fma/                             # Music dataset (auto-downloaded)
├── mit_rirs/                        # Room impulse responses (auto-downloaded)
│
├── requirements.txt                 # Python dependencies
├── SETUP_COMPLETE.ps1              # Automated setup script
├── train_houwme_wakeword.py        # Main training script
├── generate_samples_coqui.py       # TTS sample generation
├── test_train.yaml                 # Test training configuration
├── my_model.yaml                   # Full training configuration
└── README_SETUP.md                 # This file
```

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
