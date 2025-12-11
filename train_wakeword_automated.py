#!/usr/bin/env python3
"""
Automated Wake Word Training Workflow
Complete pipeline from environment check to model export
"""

import os
import sys
import subprocess
import platform
import yaml
import shutil
from pathlib import Path
from typing import Optional, Tuple
import json

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(70)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}[OK] {text}{Colors.ENDC}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.FAIL}[ERROR] {text}{Colors.ENDC}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}[!] {text}{Colors.ENDC}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.OKCYAN}[INFO] {text}{Colors.ENDC}")

def check_python_version() -> bool:
    """Check if Python version is 3.11.x"""
    print_header("Checking Python Version")
    version = sys.version_info
    print_info(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor == 11:
        print_success("Python 3.11.x detected")
        return True
    else:
        print_error(f"Python 3.11.x required, found {version.major}.{version.minor}.{version.micro}")
        return False

def check_gpu() -> Tuple[bool, Optional[str]]:
    """Check for NVIDIA GPU and CUDA availability"""
    print_header("Checking GPU Availability")
    
    # Check nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Extract GPU name
            for line in result.stdout.split('\n'):
                if 'NVIDIA' in line and '|' in line:
                    parts = line.split('|')
                    if len(parts) > 1:
                        gpu_name = parts[1].strip().split('  ')[0]
                        print_success(f"NVIDIA GPU detected: {gpu_name}")
                        break
            
            # Check PyTorch CUDA
            try:
                import torch
                if torch.cuda.is_available():
                    cuda_version = torch.version.cuda
                    gpu_name = torch.cuda.get_device_name(0)
                    print_success(f"PyTorch CUDA available: {cuda_version}")
                    print_success(f"GPU: {gpu_name}")
                    return True, gpu_name
                else:
                    print_warning("PyTorch installed but CUDA not available")
                    return False, None
            except ImportError:
                print_warning("PyTorch not installed yet")
                return True, "NVIDIA GPU (PyTorch pending)"
        else:
            print_warning("nvidia-smi command failed")
            return False, None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print_warning("nvidia-smi not found - no NVIDIA GPU detected")
        return False, None

def check_virtual_env() -> bool:
    """Check if running in virtual environment"""
    print_header("Checking Virtual Environment")
    
    in_venv = sys.prefix != sys.base_prefix
    if in_venv:
        print_success(f"Virtual environment active: {sys.prefix}")
        return True
    else:
        print_error("Not running in virtual environment")
        print_info("Please activate virtual environment:")
        print_info("  .\\wakeword_env\\Scripts\\Activate.ps1")
        return False

def check_dependencies() -> dict:
    """Check if required packages are installed"""
    print_header("Checking Dependencies")
    
    required = {
        'torch': 'PyTorch (Deep Learning)',
        'TTS': 'Coqui TTS (Sample Generation)',
        'audiomentations': 'Audio Augmentation',
        'numpy': 'Numerical Computing',
        'scipy': 'Scientific Computing',
        'onnx': 'Model Export',
        'onnxruntime': 'ONNX Runtime',
        'pydub': 'Audio Processing',
        'librosa': 'Audio Analysis',
        'soundfile': 'Audio I/O',
        'yaml': 'Configuration Files',
    }
    
    status = {}
    missing = []
    
    for package, description in required.items():
        try:
            __import__(package)
            print_success(f"{package:20s} - {description}")
            status[package] = True
        except ImportError:
            print_error(f"{package:20s} - {description} (MISSING)")
            status[package] = False
            missing.append(package)
    
    if missing:
        print_warning(f"\nMissing packages: {', '.join(missing)}")
        print_info("Install with: pip install -r requirements.txt")
    
    return status

def clone_openwakeword(base_dir: Path) -> bool:
    """Clone OpenWakeWord repository if not present"""
    openwakeword_dir = base_dir / "openwakeword"
    
    if openwakeword_dir.exists():
        print_success(f"OpenWakeWord already present: {openwakeword_dir}")
        return True
    
    print_header("Cloning OpenWakeWord")
    print_info("Downloading OpenWakeWord from GitHub...")
    
    try:
        subprocess.run(
            ['git', 'clone', 'https://github.com/dscripka/openWakeWord.git', 'openwakeword'],
            cwd=str(base_dir),
            check=True
        )
        print_success("OpenWakeWord cloned successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Git clone failed: {e}")
        print_info("Make sure git is installed: https://git-scm.com/download/win")
        return False
    except FileNotFoundError:
        print_error("Git not found")
        print_info("Install git from: https://git-scm.com/download/win")
        return False

def install_dependencies() -> bool:
    """Install missing dependencies"""
    print_header("Installing Dependencies")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    if not requirements_file.exists():
        print_error("requirements.txt not found")
        return False
    
    print_info("Installing packages from requirements.txt...")
    print_info("This may take several minutes...")
    
    try:
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)],
            check=True
        )
        print_success("All dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Installation failed: {e}")
        return False

def get_user_input(prompt: str, default: str = None, input_type: type = str):
    """Get user input with optional default value"""
    if default:
        full_prompt = f"{Colors.OKCYAN}{prompt} [{default}]: {Colors.ENDC}"
    else:
        full_prompt = f"{Colors.OKCYAN}{prompt}: {Colors.ENDC}"
    
    while True:
        user_input = input(full_prompt).strip()
        
        if not user_input and default:
            return input_type(default)
        elif not user_input:
            print_warning("Input required, please try again")
            continue
        
        try:
            return input_type(user_input)
        except ValueError:
            print_warning(f"Invalid input, expected {input_type.__name__}")

def get_yes_no(prompt: str, default: bool = True) -> bool:
    """Get yes/no input from user"""
    default_str = "Y/n" if default else "y/N"
    full_prompt = f"{Colors.OKCYAN}{prompt} [{default_str}]: {Colors.ENDC}"
    
    while True:
        response = input(full_prompt).strip().lower()
        
        if not response:
            return default
        elif response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print_warning("Please answer 'y' or 'n'")

def test_sample_generation(wake_word: str, base_dir: Path) -> bool:
    """Generate a test sample and wait for user confirmation"""
    print_header("Testing Sample Generation")
    
    print_info(f"Generating test sample for '{wake_word}'...")
    print_info("This will use the default TTS model to create one sample")
    
    # Create temporary test directory
    test_dir = base_dir / "test_sample"
    test_dir.mkdir(exist_ok=True)
    
    try:
        # Import TTS
        from TTS.api import TTS
        import soundfile as sf
        
        # Initialize TTS (using fast model for quick test)
        print_info("Loading TTS model...")
        tts = TTS("tts_models/en/ljspeech/fast_pitch")
        
        # Generate sample
        test_file = test_dir / f"{wake_word}_test.wav"
        print_info(f"Generating: {test_file}")
        tts.tts_to_file(text=wake_word, file_path=str(test_file))
        
        print_success(f"Test sample generated: {test_file}")
        print_info("\nPlease listen to the sample to verify it sounds correct")
        print_info(f"Location: {test_file}")
        
        # Ask user to confirm
        if platform.system() == 'Windows':
            print_info("\nAttempting to play sample...")
            try:
                subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "{test_file}").PlaySync()'], 
                             timeout=10)
            except:
                print_warning("Auto-play failed, please play manually")
        
        confirmed = get_yes_no("\nDoes the sample sound correct?", default=True)
        
        if confirmed:
            print_success("Sample confirmed - proceeding with training")
            return True
        else:
            print_warning("Sample rejected - you may need to adjust pronunciation")
            print_info("Consider creating a custom pronunciation in generate_samples_coqui.py")
            
            retry = get_yes_no("Continue anyway?", default=False)
            return retry
            
    except Exception as e:
        print_error(f"Sample generation failed: {e}")
        print_info("You can still continue, but verify samples manually later")
        return get_yes_no("Continue anyway?", default=True)
    finally:
        # Cleanup test directory
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)

def create_training_config(
    wake_word: str,
    n_samples: int,
    training_steps: int,
    false_activation_penalty: float,
    base_dir: Path,
    use_gpu: bool
) -> Path:
    """Create training configuration YAML file"""
    print_header("Creating Training Configuration")
    
    # Determine paths
    model_name = wake_word.lower().replace(' ', '_')
    clips_dir = base_dir / "clips" / "generated" / model_name
    output_dir = base_dir / "trained_models" / model_name
    acav100m_features = base_dir / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build configuration
    config = {
        'model_name': model_name,
        'wake_word': wake_word,
        
        # Training parameters
        'n_samples': n_samples,
        'steps': training_steps,
        'false_activation_penalty': false_activation_penalty,
        
        # Paths
        'positive_audio_dir': str(clips_dir),
        'output_dir': str(output_dir),
        
        # Feature settings
        'feature_extraction': {
            'n_fft': 2048,
            'hop_length': 512,
            'n_mels': 96,
            'sample_rate': 16000
        },
        
        # Model settings
        'model': {
            'target_phrase_weight': 1.0,
            'negative_phrase_weight': false_activation_penalty,
        },
        
        # GPU settings
        'device': 'cuda' if use_gpu else 'cpu',
        'batch_size': 128 if use_gpu else 32,
        
        # Augmentation
        'augmentation': {
            'background_noise_dir': str(base_dir / "audioset_16k"),
            'music_dir': str(base_dir / "fma"),
            'room_impulse_dir': str(base_dir / "mit_rirs"),
        }
    }
    
    # Add ACAV100M features if available
    if acav100m_features.exists():
        config['ACAV100M_features_path'] = str(acav100m_features)
        config['ACAV100M_sample'] = 1024
        print_success(f"ACAV100M features found: {acav100m_features}")
    else:
        print_warning("ACAV100M features not found - adversarial sampling disabled")
        print_info("Download from: https://github.com/dscripka/openWakeWord")
    
    # Save configuration
    config_file = base_dir / f"training_config_{model_name}.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print_success(f"Configuration saved: {config_file}")
    print_info(f"  Wake word: {wake_word}")
    print_info(f"  Samples: {n_samples}")
    print_info(f"  Steps: {training_steps}")
    print_info(f"  False activation penalty: {false_activation_penalty}")
    print_info(f"  Device: {'GPU (CUDA)' if use_gpu else 'CPU'}")
    print_info(f"  Output: {output_dir}")
    
    return config_file

def generate_samples(wake_word: str, n_samples: int, base_dir: Path) -> bool:
    """Generate TTS samples for training"""
    print_header("Generating TTS Samples")
    
    model_name = wake_word.lower().replace(' ', '_')
    clips_dir = base_dir / "clips" / "generated" / model_name
    clips_dir.mkdir(parents=True, exist_ok=True)
    
    print_info(f"Generating {n_samples} samples for '{wake_word}'")
    print_info(f"Output directory: {clips_dir}")
    
    # Import and run sample generator
    try:
        # Check if custom generator exists
        generator_script = base_dir / "generate_samples_coqui.py"
        if generator_script.exists():
            print_info("Using custom sample generator (generate_samples_coqui.py)")
            result = subprocess.run(
                [sys.executable, str(generator_script), 
                 '--wake-word', wake_word,
                 '--samples', str(n_samples),
                 '--output', str(clips_dir)],
                cwd=str(base_dir),
                capture_output=False
            )
            return result.returncode == 0
        else:
            # Use piper-sample-generator
            print_info("Using piper-sample-generator")
            piper_gen = base_dir / "piper-sample-generator" / "generate_samples.py"
            if piper_gen.exists():
                result = subprocess.run(
                    [sys.executable, str(piper_gen),
                     '--wake-word', wake_word,
                     '--number', str(n_samples),
                     '--output-dir', str(clips_dir)],
                    cwd=str(base_dir),
                    capture_output=False
                )
                return result.returncode == 0
            else:
                print_error("No sample generator found")
                return False
                
    except Exception as e:
        print_error(f"Sample generation failed: {e}")
        return False

def augment_samples(config_file: Path, base_dir: Path) -> bool:
    """Augment samples with background noise and room impulse responses"""
    print_header("Augmenting Samples")
    
    print_info("Applying audio augmentation (noise, music, reverb)")
    print_info("This may take several minutes...")
    
    openwakeword_dir = base_dir / "openwakeword"
    train_script = openwakeword_dir / "openwakeword" / "train.py"
    
    if not train_script.exists():
        print_error(f"Training script not found: {train_script}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(train_script),
             '--training_config', str(config_file),
             '--augment_clips'],
            cwd=str(openwakeword_dir),
            capture_output=False
        )
        
        if result.returncode == 0:
            print_success("Augmentation completed")
            return True
        else:
            print_error("Augmentation failed")
            return False
            
    except Exception as e:
        print_error(f"Augmentation failed: {e}")
        return False

def train_model(config_file: Path, base_dir: Path) -> bool:
    """Train the wake word model"""
    print_header("Training Model")
    
    print_info("Starting model training")
    print_info("This may take 30 minutes to several hours depending on settings...")
    
    openwakeword_dir = base_dir / "openwakeword"
    train_script = openwakeword_dir / "openwakeword" / "train.py"
    
    if not train_script.exists():
        print_error(f"Training script not found: {train_script}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(train_script),
             '--training_config', str(config_file),
             '--train_model'],
            cwd=str(openwakeword_dir),
            capture_output=False
        )
        
        if result.returncode == 0:
            print_success("Training completed")
            return True
        else:
            print_error("Training failed")
            return False
            
    except Exception as e:
        print_error(f"Training failed: {e}")
        return False

def export_to_onnx(model_dir: Path, model_name: str) -> Optional[Path]:
    """Export model to ONNX format"""
    print_header("Exporting to ONNX")
    
    # Model should already be exported by train.py
    onnx_file = model_dir / f"{model_name}.onnx"
    
    if onnx_file.exists():
        size_kb = onnx_file.stat().st_size / 1024
        print_success(f"ONNX model found: {onnx_file}")
        print_info(f"Size: {size_kb:.2f} KB")
        return onnx_file
    else:
        print_warning("ONNX model not found - may need manual export")
        return None

def convert_to_tflite(onnx_file: Path) -> Optional[Path]:
    """Convert ONNX model to TFLite format"""
    print_header("Converting to TFLite")
    
    print_warning("TFLite conversion has known dependency issues on Windows")
    print_info("ONNX format is fully supported by Home Assistant")
    
    if not get_yes_no("Attempt TFLite conversion?", default=False):
        print_info("Skipping TFLite conversion")
        return None
    
    tflite_file = onnx_file.parent / f"{onnx_file.stem}.tflite"
    
    try:
        # Try using onnx2tf (most reliable)
        print_info("Attempting conversion with onnx2tf...")
        result = subprocess.run(
            ['onnx2tf', '-i', str(onnx_file), '-o', str(onnx_file.parent), '-osd'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0 and tflite_file.exists():
            size_kb = tflite_file.stat().st_size / 1024
            print_success(f"TFLite model created: {tflite_file}")
            print_info(f"Size: {size_kb:.2f} KB")
            return tflite_file
        else:
            print_error("onnx2tf conversion failed")
            print_info(result.stderr)
            
    except subprocess.TimeoutExpired:
        print_error("Conversion timeout (5 minutes)")
    except FileNotFoundError:
        print_error("onnx2tf not found - install with: pip install onnx2tf")
    except Exception as e:
        print_error(f"Conversion failed: {e}")
    
    print_info("TFLite conversion failed - using ONNX model only")
    return None

def print_summary(wake_word: str, model_dir: Path, onnx_file: Optional[Path], 
                  tflite_file: Optional[Path], n_samples: int, training_steps: int):
    """Print training summary"""
    print_header("Training Complete!")
    
    print(f"{Colors.OKGREEN}{Colors.BOLD}Wake Word:{Colors.ENDC} {wake_word}")
    print(f"{Colors.OKGREEN}{Colors.BOLD}Samples:{Colors.ENDC} {n_samples}")
    print(f"{Colors.OKGREEN}{Colors.BOLD}Training Steps:{Colors.ENDC} {training_steps}")
    print(f"{Colors.OKGREEN}{Colors.BOLD}Output Directory:{Colors.ENDC} {model_dir}")
    
    print(f"\n{Colors.BOLD}Generated Models:{Colors.ENDC}")
    if onnx_file:
        size_kb = onnx_file.stat().st_size / 1024
        print(f"  {Colors.OKGREEN}[OK]{Colors.ENDC} ONNX: {onnx_file} ({size_kb:.2f} KB)")
    else:
        print(f"  {Colors.FAIL}[X]{Colors.ENDC} ONNX: Not found")
    
    if tflite_file:
        size_kb = tflite_file.stat().st_size / 1024
        print(f"  {Colors.OKGREEN}[OK]{Colors.ENDC} TFLite: {tflite_file} ({size_kb:.2f} KB)")
    else:
        print(f"  {Colors.WARNING}[-]{Colors.ENDC} TFLite: Not created (ONNX is sufficient)")
    
    print(f"\n{Colors.BOLD}Next Steps:{Colors.ENDC}")
    print(f"  1. Test the model with OpenWakeWord")
    print(f"  2. Copy to Home Assistant: {onnx_file if onnx_file else 'model.onnx'}")
    print(f"  3. Configure Wyoming OpenWakeWord service")
    print(f"  4. Test wake word detection in Home Assistant")
    
    if onnx_file:
        print(f"\n{Colors.OKCYAN}Copy command for Home Assistant:{Colors.ENDC}")
        print(f"  Copy-Item \"{onnx_file}\" \"\\\\homeassistant\\config\\custom_wakewords\\{wake_word}.onnx\"")

def main():
    """Main automated workflow"""
    print_header("Automated Wake Word Training Workflow")
    print(f"{Colors.BOLD}OpenWakeWord Custom Training System{Colors.ENDC}\n")
    
    # Get base directory
    base_dir = Path(__file__).parent.resolve()
    print_info(f"Working directory: {base_dir}")
    
    # Step 1: Environment checks
    if not check_python_version():
        sys.exit(1)
    
    if not check_virtual_env():
        sys.exit(1)
    
    has_gpu, gpu_name = check_gpu()
    
    # Step 2: Dependency checks
    deps_status = check_dependencies()
    all_deps_ok = all(deps_status.values())
    
    if not all_deps_ok:
        print_warning("\nSome dependencies are missing")
        if get_yes_no("Install missing dependencies now?", default=True):
            if not install_dependencies():
                print_error("Dependency installation failed")
                sys.exit(1)
            print_success("Dependencies installed - please restart script")
            sys.exit(0)
        else:
            print_error("Cannot proceed without dependencies")
            sys.exit(1)
    
    # Step 2.5: Clone OpenWakeWord if needed
    if not clone_openwakeword(base_dir):
        print_error("OpenWakeWord repository required")
        sys.exit(1)
    
    print_success("\nAll environment checks passed!")
    
    # Step 3: Get wake word
    print_header("Wake Word Configuration")
    wake_word = get_user_input("Enter wake word to train", default="homie")
    
    # Step 4: Test sample generation
    if get_yes_no("Generate test sample for verification?", default=True):
        if not test_sample_generation(wake_word, base_dir):
            if not get_yes_no("Continue anyway?", default=False):
                print_info("Training cancelled")
                sys.exit(0)
    
    # Step 5: Get training parameters
    print_header("Training Parameters")
    
    print_info("Number of samples to generate:")
    print_info("  - Quick test: 1000 samples (~30-60 min)")
    print_info("  - Standard: 3000 samples (~2-4 hours)")
    print_info("  - High quality: 5000+ samples (4+ hours)")
    n_samples = get_user_input("Number of samples", default="3000", input_type=int)
    
    print_info("\nTraining steps:")
    print_info("  - Quick test: 10000 steps")
    print_info("  - Standard: 30000 steps")
    print_info("  - High quality: 50000+ steps")
    training_steps = get_user_input("Training steps", default="30000", input_type=int)
    
    print_info("\nFalse activation penalty:")
    print_info("  - Lower (0.5-1.0): More sensitive, may have false activations")
    print_info("  - Medium (1.0-2.0): Balanced")
    print_info("  - Higher (2.0-5.0): Less sensitive, fewer false activations")
    false_activation_penalty = get_user_input("False activation penalty", default="1.0", input_type=float)
    
    # Step 6: Confirm settings
    print_header("Configuration Summary")
    print(f"{Colors.BOLD}Wake Word:{Colors.ENDC} {wake_word}")
    print(f"{Colors.BOLD}Samples:{Colors.ENDC} {n_samples}")
    print(f"{Colors.BOLD}Training Steps:{Colors.ENDC} {training_steps}")
    print(f"{Colors.BOLD}False Activation Penalty:{Colors.ENDC} {false_activation_penalty}")
    print(f"{Colors.BOLD}GPU Acceleration:{Colors.ENDC} {'Yes - ' + gpu_name if has_gpu else 'No (CPU only)'}")
    
    if not get_yes_no("\nProceed with training?", default=True):
        print_info("Training cancelled")
        sys.exit(0)
    
    # Step 7: Create configuration
    config_file = create_training_config(
        wake_word=wake_word,
        n_samples=n_samples,
        training_steps=training_steps,
        false_activation_penalty=false_activation_penalty,
        base_dir=base_dir,
        use_gpu=has_gpu
    )
    
    # Step 8: Generate samples
    if not generate_samples(wake_word, n_samples, base_dir):
        print_error("Sample generation failed")
        if not get_yes_no("Continue with existing samples?", default=False):
            sys.exit(1)
    
    # Step 9: Augment samples
    if not augment_samples(config_file, base_dir):
        print_error("Sample augmentation failed")
        if not get_yes_no("Continue anyway?", default=False):
            sys.exit(1)
    
    # Step 10: Train model
    if not train_model(config_file, base_dir):
        print_error("Model training failed")
        sys.exit(1)
    
    # Step 11: Export models
    model_name = wake_word.lower().replace(' ', '_')
    model_dir = base_dir / "trained_models" / model_name
    
    onnx_file = export_to_onnx(model_dir, model_name)
    tflite_file = convert_to_tflite(onnx_file) if onnx_file else None
    
    # Step 12: Print summary
    print_summary(wake_word, model_dir, onnx_file, tflite_file, n_samples, training_steps)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Training interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
