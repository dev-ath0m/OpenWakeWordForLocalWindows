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
    """Clone OpenWakeWord repository if not present and install it"""
    openwakeword_dir = base_dir / "openwakeword"
    
    # Check if already cloned
    if openwakeword_dir.exists():
        print_success(f"OpenWakeWord already present: {openwakeword_dir}")
    else:
        print_header("Cloning OpenWakeWord")
        print_info("Downloading OpenWakeWord from GitHub...")
        
        try:
            subprocess.run(
                ['git', 'clone', 'https://github.com/dscripka/openWakeWord.git', 'openwakeword'],
                cwd=str(base_dir),
                check=True
            )
            print_success("OpenWakeWord cloned successfully")
        except subprocess.CalledProcessError as e:
            print_error(f"Git clone failed: {e}")
            print_info("Make sure git is installed: https://git-scm.com/download/win")
            return False
        except FileNotFoundError:
            print_error("Git not found")
            print_info("Install git from: https://git-scm.com/download/win")
            return False
    
    # Install OpenWakeWord as editable package
    print_info("Installing OpenWakeWord package...")
    try:
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-e', str(openwakeword_dir)],
            check=True,
            capture_output=True
        )
        print_success("OpenWakeWord package installed")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"OpenWakeWord installation failed: {e}")
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

def get_pronunciations(wake_word: str) -> list:
    """Get list of pronunciation variations from user"""
    print_header("Pronunciation Variations")
    
    print_info("You can provide multiple pronunciation variations to improve model accuracy")
    print_info("Examples for 'homie': homie, houw mee, ho mee, home ee")
    print_info("")
    print_info("Enter pronunciations one per line (press Enter twice when done)")
    print_info(f"First pronunciation (default): {wake_word}")
    
    pronunciations = [wake_word]  # Always include the original
    
    while True:
        user_input = input(f"{Colors.OKCYAN}Pronunciation #{len(pronunciations) + 1} (or press Enter to finish): {Colors.ENDC}").strip()
        
        if not user_input:
            break
        
        if user_input.lower() != wake_word.lower() and user_input not in pronunciations:
            pronunciations.append(user_input)
            print_success(f"Added: {user_input}")
        elif user_input in pronunciations:
            print_warning("Already added, skipping")
        else:
            print_warning("Same as original, skipping")
    
    print_success(f"\nTotal pronunciations: {len(pronunciations)}")
    for i, p in enumerate(pronunciations, 1):
        print_info(f"  {i}. {p}")
    
    return pronunciations

def test_sample_generation(wake_word: str, pronunciations: list, base_dir: Path) -> list:
    """Generate test samples for each pronunciation and get user feedback"""
    print_header("Testing Sample Generation")
    
    print_info(f"Generating test samples for {len(pronunciations)} pronunciation(s)")
    print_info("This will use the default TTS model to create one sample for each")
    
    # Create temporary test directory
    test_dir = base_dir / "test_sample"
    test_dir.mkdir(exist_ok=True)
    
    approved_pronunciations = []
    test_files = []
    
    try:
        # Import TTS
        from TTS.api import TTS
        
        # Initialize TTS (using fast model for quick test)
        print_info("\nLoading TTS model...")
        tts = TTS("tts_models/en/ljspeech/fast_pitch")
        
        # Generate samples for each pronunciation
        for i, pronunciation in enumerate(pronunciations, 1):
            print_info(f"\n[{i}/{len(pronunciations)}] Generating sample for: '{pronunciation}'")
            
            safe_name = pronunciation.replace(' ', '_').replace('/', '_')
            test_file = test_dir / f"{safe_name}_test.wav"
            
            try:
                tts.tts_to_file(text=pronunciation, file_path=str(test_file))
                test_files.append((pronunciation, test_file))
                print_success(f"Generated: {test_file}")
            except Exception as e:
                print_error(f"Failed to generate sample: {e}")
                continue
        
        # Play and get user feedback for each sample
        print_header("Review Generated Samples")
        print_info("Please listen to each sample and decide if the pronunciation sounds good")
        
        for pronunciation, test_file in test_files:
            print(f"\n{Colors.BOLD}Pronunciation: {pronunciation}{Colors.ENDC}")
            print_info(f"File: {test_file}")
            
            # Auto-play on Windows
            if platform.system() == 'Windows':
                print_info("Playing sample...")
                try:
                    subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "{test_file}").PlaySync()'], 
                                 timeout=10)
                except:
                    print_warning("Auto-play failed, please play manually")
            
            # Get user decision
            keep = get_yes_no(f"Keep this pronunciation '{pronunciation}' for training?", default=True)
            
            if keep:
                approved_pronunciations.append(pronunciation)
                print_success(f"✓ Kept: {pronunciation}")
            else:
                print_warning(f"✗ Skipped: {pronunciation}")
        
        # Summary
        print_header("Pronunciation Selection Summary")
        if approved_pronunciations:
            print_success(f"Selected {len(approved_pronunciations)} pronunciation(s) for training:")
            for i, p in enumerate(approved_pronunciations, 1):
                print_info(f"  {i}. {p}")
        else:
            print_warning("No pronunciations selected!")
            
        return approved_pronunciations
            
    except Exception as e:
        print_error(f"Sample generation failed: {e}")
        print_info("You can still continue, but verify samples manually later")
        return get_yes_no("Continue anyway?", default=True)
    finally:
        # Cleanup test directory
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)

def cleanup_incomplete_training(wake_word: str, base_dir: Path) -> None:
    """Clean up incomplete training artifacts that could interfere with new training"""
    print_header("Pre-Training Cleanup")
    
    model_name = wake_word.lower().replace(' ', '_')
    model_dir = base_dir / "trained_models" / model_name
    clips_dir = base_dir / "clips" / "generated" / model_name
    
    cleaned_items = []
    
    # Remove incomplete feature files (.npy)
    if model_dir.exists():
        npy_files = list(model_dir.glob("*.npy"))
        if npy_files:
            print_info(f"Removing {len(npy_files)} incomplete feature files...")
            for npy_file in npy_files:
                try:
                    npy_file.unlink()
                    cleaned_items.append(f"Feature file: {npy_file.name}")
                except Exception as e:
                    print_warning(f"Could not remove {npy_file.name}: {e}")
        
        # Remove incomplete negative sample directories
        negative_dirs = [d for d in model_dir.glob("negative_*") if d.is_dir()]
        if negative_dirs:
            print_info(f"Removing {len(negative_dirs)} incomplete negative sample directories...")
            for neg_dir in negative_dirs:
                try:
                    shutil.rmtree(neg_dir)
                    cleaned_items.append(f"Negative dir: {neg_dir.name}")
                except Exception as e:
                    print_warning(f"Could not remove {neg_dir.name}: {e}")
        
        # Remove partial model files
        partial_models = list(model_dir.glob("checkpoint_*.pt")) + list(model_dir.glob("*.pth"))
        if partial_models:
            print_info(f"Removing {len(partial_models)} checkpoint files...")
            for model_file in partial_models:
                try:
                    model_file.unlink()
                    cleaned_items.append(f"Checkpoint: {model_file.name}")
                except Exception as e:
                    print_warning(f"Could not remove {model_file.name}: {e}")
    
    # Remove incomplete augmented samples
    if clips_dir.exists():
        augmented_dirs = [d for d in clips_dir.glob("*_augmented") if d.is_dir()]
        if augmented_dirs:
            print_info(f"Removing {len(augmented_dirs)} incomplete augmented sample directories...")
            for aug_dir in augmented_dirs:
                try:
                    shutil.rmtree(aug_dir)
                    cleaned_items.append(f"Augmented dir: {aug_dir.name}")
                except Exception as e:
                    print_warning(f"Could not remove {aug_dir.name}: {e}")
    
    if cleaned_items:
        print_success(f"Cleaned up {len(cleaned_items)} incomplete training artifacts")
        for item in cleaned_items[:5]:  # Show first 5
            print_info(f"  - {item}")
        if len(cleaned_items) > 5:
            print_info(f"  ... and {len(cleaned_items) - 5} more")
    else:
        print_success("No incomplete training artifacts found")

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
        },
        
        # Disable Piper generator (not compatible with Windows)
        'use_piper': False
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

def generate_samples(wake_word: str, pronunciations: list, n_samples: int, base_dir: Path) -> bool:
    """Generate TTS samples for training using multiple models with automatic downloading"""
    print_header("Generating TTS Samples")
    
    model_name = wake_word.lower().replace(' ', '_')
    clips_dir = base_dir / "clips" / "generated" / model_name
    clips_dir.mkdir(parents=True, exist_ok=True)
    
    print_info(f"Generating {n_samples} samples for '{wake_word}'")
    print_info(f"Using {len(pronunciations)} pronunciation(s): {', '.join(pronunciations)}")
    print_info(f"Output directory: {clips_dir}")
    
    # Check if custom generator exists (provides more control and variety)
    generator_script = base_dir / "generate_samples_coqui.py"
    if generator_script.exists():
        print_info("Using custom multi-model generator (generate_samples_coqui.py)")
        print_info("This will use 6 different TTS models for maximum voice variety")
        print_info("Models will be downloaded to 'tts/' folder if not cached")
        
        # Note: Custom generator needs to be updated to accept pronunciations
        print_warning("Note: Custom generator will use its built-in pronunciations")
        
        try:
            result = subprocess.run(
                [sys.executable, str(generator_script), 
                 '--samples', str(n_samples)],
                cwd=str(base_dir),
                capture_output=False
            )
            
            if result.returncode == 0:
                # Move generated samples to target directory
                source_clips = base_dir / "clips"
                if source_clips.exists():
                    import shutil
                    for item in source_clips.iterdir():
                        if item.is_dir():
                            dest = clips_dir / item.name
                            if dest.exists():
                                shutil.rmtree(dest)
                            shutil.move(str(item), str(dest))
                        elif item.suffix == '.wav':
                            shutil.move(str(item), str(clips_dir / item.name))
                return True
            else:
                print_warning("Custom generator failed, trying built-in method...")
        except Exception as e:
            print_warning(f"Custom generator error: {e}")
            print_info("Falling back to built-in sample generation...")
    
    # Built-in sample generation with multiple TTS models
    print_info("Using built-in multi-model TTS generator")
    print_info("Models will be downloaded and cached in 'tts/' folder automatically")
    
    # Generate both positive and negative samples
    positive_success = _generate_positive_samples(wake_word, pronunciations, n_samples, clips_dir, base_dir)
    negative_success = _generate_negative_samples(wake_word, n_samples, base_dir)
    
    return positive_success


def _generate_positive_samples(wake_word: str, pronunciations: list, n_samples: int, clips_dir: Path, base_dir: Path) -> bool:
    """Generate positive samples (wake word pronunciations) using TTS"""
    print_header("Generating Positive Samples")
    
    model_name = wake_word.lower().replace(' ', '_')
    
    try:
        from TTS.api import TTS
        import librosa
        from scipy.io import wavfile
        import numpy as np
        
        # Set TTS cache directory
        import os
        os.environ['TTS_HOME'] = str(base_dir)
        
        # TTS models to use for variety
        tts_models = [
            {"model": "tts_models/en/ljspeech/tacotron2-DDC", "gender": "female"},
            {"model": "tts_models/en/ljspeech/glow-tts", "gender": "female"},
            {"model": "tts_models/en/ljspeech/fast_pitch", "gender": "female"},
            {"model": "tts_models/en/jenny/jenny", "gender": "female"},
            {"model": "tts_models/multilingual/multi-dataset/your_tts", "speaker": "male-en-2", "language": "en"},
            {"model": "tts_models/multilingual/multi-dataset/your_tts", "speaker": "female-en-5", "language": "en"},
        ]
        
        # Use user-provided pronunciations
        variations = pronunciations
        
        # Check GPU availability
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
        
        samples_per_combo = n_samples // (len(tts_models) * len(variations))
        
        print_info(f"Using {len(tts_models)} TTS models with {len(variations)} pronunciations")
        print_info(f"Generating ~{samples_per_combo} samples per combination")
        print_info(f"Device: {device.upper()}")
        print_info("Starting sample generation...\n")
        
        # Suppress verbose TTS logging completely
        import logging
        logging.getLogger('TTS').setLevel(logging.CRITICAL)
        logging.getLogger('TTS.tts.utils.synthesis').setLevel(logging.CRITICAL)
        logging.getLogger('TTS.tts.models').setLevel(logging.CRITICAL)
        logging.getLogger('TTS.utils').setLevel(logging.CRITICAL)
        
        # Suppress stdout from TTS
        import sys as _sys
        import io
        
        class SuppressOutput:
            def __enter__(self):
                self._original_stdout = _sys.stdout
                self._original_stderr = _sys.stderr
                _sys.stdout = io.StringIO()
                _sys.stderr = io.StringIO()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                _sys.stdout = self._original_stdout
                _sys.stderr = self._original_stderr
        
        valid_count = 0
        failed_count = 0
        
        for model_config in tts_models:
            model_path = model_config["model"]
            model_short = model_path.split('/')[-1]
            
            print_info(f"\nLoading model: {model_short}...")
            try:
                with SuppressOutput():
                    tts = TTS(model_name=model_path).to(device)
                
                for variation in variations:
                    # Create subfolder for this model/variation
                    gender = model_config.get("gender", "voice")
                    speaker = model_config.get("speaker", "")
                    speaker_suffix = f"_{speaker}" if speaker else ""
                    subfolder = clips_dir / f"{model_short}{speaker_suffix}_{gender}" / variation.replace(" ", "_")
                    subfolder.mkdir(parents=True, exist_ok=True)
                    
                    # Generate samples for this combination
                    for i in range(samples_per_combo):
                        output_file = subfolder / f"{model_name}_{valid_count}.wav"
                        
                        try:
                            # Generate with slight speed variation
                            speed = 1.0 + np.random.uniform(-0.1, 0.1)
                            
                            kwargs = {
                                'text': variation,
                                'file_path': str(output_file),
                                'speed': speed
                            }
                            
                            if 'speaker' in model_config:
                                kwargs['speaker'] = model_config['speaker']
                            if 'language' in model_config:
                                kwargs['language'] = model_config['language']
                            
                            # Generate sample with suppressed output
                            with SuppressOutput():
                                tts.tts_to_file(**kwargs)
                            
                            # Validate and process
                            sr, audio = wavfile.read(str(output_file))
                            duration = len(audio) / sr
                            
                            # Skip if too long (>4 seconds)
                            if duration > 4.0:
                                output_file.unlink()
                                failed_count += 1
                                continue
                            
                            # Trim silence
                            audio_float = audio.astype(np.float32)
                            audio_trimmed, _ = librosa.effects.trim(audio_float, top_db=30)
                            
                            # Resample to 16kHz if needed
                            if sr != 16000:
                                from scipy import signal
                                num_samples = int(len(audio_trimmed) * 16000 / sr)
                                audio_resampled = signal.resample(audio_trimmed, num_samples)
                                audio = audio_resampled.astype(np.int16)
                            else:
                                audio = audio_trimmed.astype(np.int16)
                            
                            # Save final version
                            wavfile.write(str(output_file), 16000, audio)
                            valid_count += 1
                            
                            # Show progress bar with statistics
                            progress_pct = (valid_count / n_samples) * 100
                            total_attempts = valid_count + failed_count
                            fail_pct = (failed_count / total_attempts * 100) if total_attempts > 0 else 0
                            
                            # Create progress bar (50 chars wide)
                            bar_width = 50
                            filled = int(bar_width * valid_count / n_samples)
                            bar = '█' * filled + '░' * (bar_width - filled)
                            
                            # Format variation name (truncate if too long)
                            var_display = variation if len(variation) <= 20 else variation[:17] + '...'
                            
                            print(f"\r{Colors.OKCYAN}[{bar}] {progress_pct:5.1f}% | {valid_count}/{n_samples} valid | "
                                  f"Failed: {fail_pct:4.1f}% | Current: '{var_display}' ({model_short}){Colors.ENDC}", 
                                  end='', flush=True)
                                
                        except Exception as e:
                            failed_count += 1
                            if output_file.exists():
                                output_file.unlink()
                            continue
                        
                        if valid_count >= n_samples:
                            break
                    
                    if valid_count >= n_samples:
                        break
                
                # Print newline after progress bar
                print()
                total_attempts = valid_count + failed_count
                fail_pct = (failed_count / total_attempts * 100) if total_attempts > 0 else 0
                print_success(f"Completed model '{model_short}': {valid_count} valid samples, {failed_count} failed ({fail_pct:.1f}%)")
                        
            except Exception as e:
                print_warning(f"Model {model_short} failed: {e}")
                continue
            
            if valid_count >= n_samples:
                break
        
        print_success(f"\nGenerated {valid_count} valid positive samples")
        total_attempts = valid_count + failed_count
        if failed_count > 0:
            fail_pct = (failed_count / total_attempts * 100)
            print_info(f"Total attempts: {total_attempts} (Success rate: {100-fail_pct:.1f}%)")
        print_info(f"TTS models cached in: {base_dir / 'tts'}")
        return valid_count > 0
        
    except Exception as e:
        print_error(f"Positive sample generation failed: {e}")
        return False


def _generate_negative_samples(wake_word: str, n_samples: int, base_dir: Path) -> bool:
    """Generate negative samples (phonetically similar non-wake-words) using TTS as Piper replacement"""
    print_header("Generating Negative Samples")
    
    model_name = wake_word.lower().replace(' ', '_')
    output_dir = base_dir / "trained_models" / model_name
    negative_train_dir = output_dir / "negative_train"
    negative_train_dir.mkdir(parents=True, exist_ok=True)
    
    print_info("Generating phonetically similar adversarial phrases")
    print_info("These teach the model what NOT to trigger on")
    
    try:
        # Import OpenWakeWord's adversarial text generator
        import sys
        openwakeword_path = base_dir / "openwakeword"
        if str(openwakeword_path) not in sys.path:
            sys.path.insert(0, str(openwakeword_path))
        
        from openwakeword.data import generate_adversarial_texts
        
        # Generate adversarial texts based on the wake word
        print_info(f"Analyzing phonemes for '{wake_word}'...")
        adversarial_texts = generate_adversarial_texts(
            input_text=wake_word,
            N=n_samples,
            include_partial_phrase=1.0,  # Include partial phrases (e.g., "ho" from "homie")
            include_input_words=0.2      # Sometimes include actual wake word parts
        )
        
        print_success(f"Generated {len(adversarial_texts)} phonetically similar phrases")
        print_info(f"Examples: {', '.join(adversarial_texts[:5])}")
        
        # Now generate audio samples for these adversarial texts using TTS
        from TTS.api import TTS
        import librosa
        from scipy.io import wavfile
        import numpy as np
        import os
        
        os.environ['TTS_HOME'] = str(base_dir)
        
        # Use faster models for negative samples (same as Piper approach)
        tts_models = [
            "tts_models/en/ljspeech/fast_pitch",
            "tts_models/en/ljspeech/glow-tts",
        ]
        
        # Check GPU
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
        
        # Suppress logging
        import logging
        import sys as _sys
        import io
        
        logging.getLogger('TTS').setLevel(logging.CRITICAL)
        
        class SuppressOutput:
            def __enter__(self):
                self._original_stdout = _sys.stdout
                self._original_stderr = _sys.stderr
                _sys.stdout = io.StringIO()
                _sys.stderr = io.StringIO()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                _sys.stdout = self._original_stdout
                _sys.stderr = self._original_stderr
        
        valid_count = 0
        failed_count = 0
        samples_per_model = n_samples // len(tts_models)
        
        for model_name in tts_models:
            model_short = model_name.split('/')[-1]
            print_info(f"\nLoading model: {model_short}...")
            
            try:
                with SuppressOutput():
                    tts = TTS(model_name=model_name).to(device)
                
                # Cycle through adversarial texts
                for i, text in enumerate(adversarial_texts):
                    if valid_count >= samples_per_model * (tts_models.index(model_name) + 1):
                        break
                    
                    output_file = negative_train_dir / f"neg_{valid_count}.wav"
                    
                    try:
                        # Generate with speed variation
                        speed = 1.0 + np.random.uniform(-0.15, 0.15)
                        
                        with SuppressOutput():
                            tts.tts_to_file(text=text, file_path=str(output_file), speed=speed)
                        
                        # Process audio
                        sr, audio = wavfile.read(str(output_file))
                        
                        # Skip if too long
                        if len(audio) / sr > 4.0:
                            output_file.unlink()
                            failed_count += 1
                            continue
                        
                        # Trim and resample
                        audio_float = audio.astype(np.float32)
                        audio_trimmed, _ = librosa.effects.trim(audio_float, top_db=30)
                        
                        if sr != 16000:
                            from scipy import signal
                            num_samples_resampled = int(len(audio_trimmed) * 16000 / sr)
                            audio_resampled = signal.resample(audio_trimmed, num_samples_resampled)
                            audio = audio_resampled.astype(np.int16)
                        else:
                            audio = audio_trimmed.astype(np.int16)
                        
                        wavfile.write(str(output_file), 16000, audio)
                        valid_count += 1
                        
                        # Show progress
                        if valid_count % 50 == 0 or valid_count == n_samples:
                            progress_pct = (valid_count / n_samples) * 100
                            fail_pct = (failed_count / (valid_count + failed_count) * 100) if (valid_count + failed_count) > 0 else 0
                            print_info(f"Progress: {valid_count}/{n_samples} ({progress_pct:.1f}%) | Failed: {fail_pct:.1f}%")
                        
                    except Exception as e:
                        failed_count += 1
                        if output_file.exists():
                            output_file.unlink()
                        continue
                
                if valid_count >= n_samples:
                    break
                    
            except Exception as e:
                print_warning(f"Model {model_short} failed: {e}")
                continue
        
        print_success(f"\nGenerated {valid_count} negative samples in {negative_train_dir}")
        if failed_count > 0:
            print_info(f"Failed: {failed_count} samples ({failed_count/(valid_count+failed_count)*100:.1f}%)")
        return valid_count > 0
        
    except Exception as e:
        print_error(f"Negative sample generation failed: {e}")
        print_warning("Training will use ACAV100M features only for negative samples")
        import traceback
        traceback.print_exc()
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
    
    # Step 3.5: Get pronunciation variations
    if get_yes_no("\nAdd custom pronunciation variations?", default=True):
        pronunciations = get_pronunciations(wake_word)
    else:
        pronunciations = [wake_word]
        print_info(f"Using single pronunciation: {wake_word}")
    
    # Step 4: Test sample generation
    if get_yes_no("\nGenerate test samples for verification?", default=True):
        approved_pronunciations = test_sample_generation(wake_word, pronunciations, base_dir)
        
        if not approved_pronunciations:
            print_error("No pronunciations approved!")
            if not get_yes_no("Continue anyway with original wake word?", default=False):
                print_info("Training cancelled")
                sys.exit(0)
            approved_pronunciations = [wake_word]
        
        pronunciations = approved_pronunciations
    
    print_success(f"\nUsing {len(pronunciations)} pronunciation(s) for training")
    
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
    
    print_info("\nFalse activation penalty (max_negative_weight):")
    print_info("  - Lower (10-100): More sensitive, may trigger on similar sounds")
    print_info("  - Medium (100-1000): Balanced - recommended starting point")
    print_info("  - Higher (1000-5000): Very strict, fewer false activations")
    print_info("  - Very High (5000+): Maximum strictness, may miss some activations")
    false_activation_penalty = get_user_input("False activation penalty", default="1000", input_type=float)
    
    # Step 6: Confirm settings
    print_header("Configuration Summary")
    print(f"{Colors.BOLD}Wake Word:{Colors.ENDC} {wake_word}")
    print(f"{Colors.BOLD}Pronunciations:{Colors.ENDC} {', '.join(pronunciations)}")
    print(f"{Colors.BOLD}Samples:{Colors.ENDC} {n_samples}")
    print(f"{Colors.BOLD}Training Steps:{Colors.ENDC} {training_steps}")
    print(f"{Colors.BOLD}False Activation Penalty:{Colors.ENDC} {false_activation_penalty}")
    print(f"{Colors.BOLD}GPU Acceleration:{Colors.ENDC} {'Yes - ' + gpu_name if has_gpu else 'No (CPU only)'}")
    
    if not get_yes_no("\nProceed with training?", default=True):
        print_info("Training cancelled")
        sys.exit(0)
    
    # Step 6.5: Cleanup incomplete training from previous runs
    cleanup_incomplete_training(wake_word, base_dir)
    
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
    if not generate_samples(wake_word, pronunciations, n_samples, base_dir):
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
