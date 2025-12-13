#!/usr/bin/env python3
"""
Automated Wake Word Training Workflow
Complete pipeline from environment check to model export

Features:
- Environment validation (Python version, GPU, dependencies)
- Background dataset management (MIT RIRs, FMA, AudioSet)
- TTS-based sample generation with multiple voices
- Custom voice recording from microphone (optional)
- Audio augmentation with noise and reverb
- Model training with OpenWakeWord
- Export to ONNX and TFLite formats
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
import psutil
import importlib.util
import warnings

# Import console logging functions
from scripts.console_logger import (
    Colors, 
    print_header, 
    print_success, 
    print_error, 
    print_warning, 
    print_info,
    print_summary
)

# Import script modules
from scripts.setup_environment import (
    check_python_version,
    check_gpu,
    check_virtual_env,
    check_dependencies,
    install_dependencies as _install_dependencies
)
from scripts.download_training_features import download_training_features
from scripts.download_background_datasets import (
    download_mit_rirs,
    download_mit_environmental,
    download_fma
)
from scripts.clone_openwakeword_repo import clone_openwakeword, check_onnx_models
from scripts.test_sample_generation import test_sample_generation
from scripts.record_custom_samples import record_custom_samples
from scripts.generate_positive_samples import generate_positive_samples
from scripts.generate_negative_samples import generate_negative_samples
from scripts.augment_samples import augment_samples as _augment_samples, run_subprocess_with_logging
from scripts.convert_audio_to_16khz import check_and_fix_audio_sample_rates
from scripts.patch_openwakeword_for_windows import apply_patches

# Suppress FutureWarning about pynvml deprecation (PyTorch CUDA still uses old import)
# The nvidia-ml-py package is installed and will be used automatically
warnings.filterwarnings('ignore', message='.*pynvml package is deprecated.*')


def install_dependencies() -> bool:
    """Install missing dependencies"""
    requirements_file = Path(__file__).parent / "requirements.txt"
    return _install_dependencies(requirements_file)

def check_background_datasets(base_dir: Path) -> dict:
    """Check and optionally download background datasets for improved model quality"""
    print_header("Checking Background Datasets")
    print_info("These improve model quality but are optional (training works without them)")
    
    # Check MIT RIRs (direct download)
    mit_rirs_path = base_dir / "mit_rirs"
    mit_rirs_path.mkdir(exist_ok=True)
    mit_rirs_count = len(list(mit_rirs_path.glob("*.wav")))
    
    if mit_rirs_count < 250:
        print_warning(f"MIT RIRs - missing or incomplete ({mit_rirs_count}/271 files)")
    else:
        print_success(f"MIT RIRs - {mit_rirs_count} files")
    
    # Check MIT Environmental (HuggingFace - used in original Colab)
    mit_env_path = base_dir / "MIT_environmental_impulse_responses"
    mit_env_path.mkdir(exist_ok=True)
    mit_env_count = len(list(mit_env_path.glob("*.wav")))
    
    if mit_env_count < 250:
        print_warning(f"MIT Environmental - missing or incomplete ({mit_env_count} files)")
    else:
        print_success(f"MIT Environmental - {mit_env_count} files")
    
    # Check FMA
    fma_path = base_dir / "fma"
    fma_path.mkdir(exist_ok=True)
    fma_count = len(list(fma_path.rglob("*.mp3")))
    
    if fma_count < 100:
        print_warning(f"FMA music - missing or incomplete ({fma_count} files)")
    else:
        print_success(f"FMA music - {fma_count} tracks")
    
    # Check AudioSet
    audioset_path = base_dir / "audioset_16k"
    audioset_path.mkdir(exist_ok=True)
    audioset_count = len(list(audioset_path.rglob("*.wav")))
    
    if audioset_count < 100:
        print_warning(f"AudioSet - missing or incomplete ({audioset_count} files)")
    else:
        print_success(f"AudioSet - {audioset_count} files")
    
    # Report summary
    missing = []
    if mit_rirs_count < 250:
        missing.append('mit_rirs')
    if mit_env_count < 250:
        missing.append('mit_environmental')
    if fma_count < 100:
        missing.append('fma')
    if audioset_count < 100:
        missing.append('audioset')
    
    if missing:
        print_info("\nSome optional datasets are missing (can download later)")
        print_info("You'll be prompted to download them before training starts")
    else:
        print_success("\nAll background datasets available for high-quality training!")
    
    return {
        'mit_rirs': mit_rirs_count >= 250,
        'mit_environmental': mit_env_count >= 250,
        'fma': fma_count >= 100,
        'audioset': audioset_count >= 100
    }

def get_user_input(prompt: str, default: str = None, input_type: type = str):
    """Get user input with optional default value"""
    if default:
        full_prompt = f"{Colors.OKCYAN}{prompt} [{default}]: {Colors.ENDC}"
    else:
        full_prompt = f"{Colors.OKCYAN}{prompt}: {Colors.ENDC}"
    
    while True:
        print(full_prompt, end='', flush=True)  # Display prompt immediately
        user_input = input().strip()
        
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
        print(full_prompt, end='', flush=True)  # Display prompt immediately
        response = input().strip().lower()
        
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
        sys.stdout.flush()
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
    # Note: output_dir should NOT include model_name - train.py adds it
    output_dir = base_dir / "trained_models"
    acav100m_features = base_dir / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    
    # Create output directory (train.py will create model_name subdirectory)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build configuration matching OpenWakeWord's expected format
    config = {
        'model_name': model_name,
        'model_type': 'dnn',
        'target_phrase': [wake_word],  # Must be a list
        
        # Sample counts
        'n_samples': n_samples,
        'n_samples_val': n_samples // 10,  # 10% for validation
        
        # Training parameters
        'steps': training_steps,
        'max_negative_weight': false_activation_penalty,
        'target_accuracy': 0.5,
        'target_recall': 0.25,
        'target_false_positives_per_hour': 0.2,
        'layer_size': 32,
        
        # Paths - must match what train.py expects
        'output_dir': str(output_dir),
        'rir_paths': [
            str(base_dir / "mit_rirs"),
            str(base_dir / "MIT_environmental_impulse_responses")
        ],
        'background_paths': [
            str(base_dir / "audioset_16k"),
            str(base_dir / "fma")
        ],
        'background_paths_duplication_rate': [1, 1],
        
        # Augmentation settings
        'augmentation_batch_size': 16,
        'augmentation_rounds': 1,
        'tts_batch_size': 50,
        
        # Batch settings for training
        'batch_n_per_class': {
            'positive': 50,
            'adversarial_negative': 50,
            'ACAV100M_sample': 1024
        },
        
        # Custom negative phrases (empty, we use adversarial generation)
        'custom_negative_phrases': [],
    }
    
    # Add ACAV100M features if available
    acav100m_features = base_dir / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    validation_features = base_dir / "validation_set_features.npy"
    
    if acav100m_features.exists():
        config['feature_data_files'] = {
            'ACAV100M_sample': str(acav100m_features)
        }
        size_mb = acav100m_features.stat().st_size / (1024 * 1024)
        print_success(f"ACAV100M features found: {acav100m_features} ({size_mb:.1f} MB)")
    else:
        print_warning("ACAV100M features not found - adversarial sampling disabled")
        print_info("These features are automatically downloaded during setup")
        print_info("If download failed, you can retry the script or download manually from:")
        print_info("https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy")
        # Remove ACAV100M from batch settings if not available
        del config['batch_n_per_class']['ACAV100M_sample']
    
    if validation_features.exists():
        config['false_positive_validation_data_path'] = str(validation_features)
        size_mb = validation_features.stat().st_size / (1024 * 1024)
        print_success(f"Validation features found: {validation_features} ({size_mb:.1f} MB)")
    else:
        print_warning("Validation features not found - using training data for validation")
        print_info("These features are automatically downloaded during setup")
        print_info("If download failed, you can retry the script or download manually from:")
        print_info("https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy")
    
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

def generate_samples(wake_word: str, pronunciations: list, n_samples: int, base_dir: Path, custom_samples_count: int = 0) -> bool:
    """Generate TTS samples for training using extracted modules
    
    Args:
        wake_word: The wake word to generate samples for
        pronunciations: List of pronunciation variations
        n_samples: Number of TTS samples to generate
        base_dir: Base directory for the project
        custom_samples_count: Number of custom voice samples recorded
    """
    # Calculate validation samples (10% of training samples)
    n_samples_val = max(1, n_samples // 10)
    
    # If we have custom samples, reduce TTS generation accordingly
    # Custom samples are already high quality, so we need fewer TTS samples
    if custom_samples_count > 0:
        print_info(f"\nAdjusting TTS generation: {custom_samples_count} custom samples already recorded")
        # Reduce TTS samples by the number of custom samples (they're worth more!)
        adjusted_n_samples = max(100, n_samples - (custom_samples_count * 2))
        print_info(f"Generating {adjusted_n_samples} TTS samples (reduced from {n_samples})")
        print_info(f"Custom samples count as ~{custom_samples_count * 2} TTS samples due to higher quality")
        print_success(f"\n✓ Training dataset will include:")
        print_info(f"  • {adjusted_n_samples} TTS-generated samples")
        print_info(f"  • {custom_samples_count} custom voice samples (already saved)")
        print_info(f"  • Total: {adjusted_n_samples + custom_samples_count} positive samples\n")
    else:
        adjusted_n_samples = n_samples
    
    # Generate both positive and negative samples using extracted modules
    positive_success = generate_positive_samples(wake_word, pronunciations, adjusted_n_samples, n_samples_val, base_dir)
    if not positive_success:
        print_error("Positive sample generation failed - cannot continue")
        return False
    
    negative_success = generate_negative_samples(wake_word, adjusted_n_samples, n_samples_val, base_dir)
    if not negative_success:
        print_error("Negative sample generation failed - cannot continue")
        print_error("Training requires both positive and negative samples")
        return False
    
    return True

def train_model(config_file: Path, base_dir: Path) -> bool:
    """Train the wake word model with progress monitoring"""
    
    print_header("Training Model")
    
    # Apply compatibility patches to OpenWakeWord
    if not apply_patches(base_dir, verbose=False):
        print_error("Failed to apply OpenWakeWord patches")
        return False
    
    print_info("Starting model training")
    print_info("This may take 30 minutes to several hours depending on settings...")
    
    openwakeword_dir = base_dir / "openwakeword"
    train_script = openwakeword_dir / "openwakeword" / "train.py"
    
    if not train_script.exists():
        print_error(f"Training script not found: {train_script}")
        return False
    
    train_log_path = base_dir / "training_output.log"
    
    print_info("Training in progress - check log for detailed progress")
    print_info(f"Log file: {train_log_path}")
    print(f"\n{Colors.BOLD}Training steps:{Colors.ENDC}")
    print(f"  1. Loading models and features")
    print(f"  2. Training neural network (this takes the longest)")
    print(f"  3. Evaluating on validation data")
    print(f"  4. Saving final model\n")
    
    result = run_subprocess_with_logging(
        command=[sys.executable, str(train_script),
                '--training_config', str(config_file),
                '--train_model'],
        log_file_path=train_log_path,
        cwd=openwakeword_dir,
        description="Training"
    )
    
    if result:
        print_success("Training completed successfully")
    
    return result

def main():
    """Main automated workflow"""
    print_header("Automated Wake Word Training Workflow")
    print(f"{Colors.BOLD}OpenWakeWord Custom Training System{Colors.ENDC}\n")
    
    # Verify ONNX models exist (downloaded by setup script)
    check_onnx_models()
    
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
    
    # Step 2.6: Download training features (ACAV100M and validation)
    if not download_training_features(base_dir):
        print_error("Training feature download failed")
        if not get_yes_no("Continue without all features? (training quality may be reduced)", default=False):
            sys.exit(1)
        print_warning("Continuing without all training features - expect reduced model quality")
    
    # Step 2.7: Check background datasets (optional but recommended)
    background_status = check_background_datasets(base_dir)
    
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
    
    # Step 4.5: Record custom voice samples (optional)
    custom_samples_count = 0
    if get_yes_no("\nRecord your own voice samples?", default=False):
        print_info("\nRecording your own voice adds your unique pronunciation to the training data")
        print_info("This can significantly improve accuracy for your specific voice")
        print_info("Press SPACE to record each sample, ENTER when done (recommend 10+ samples)\n")
        
        success, count = record_custom_samples(wake_word, base_dir, target_count=10)
        if success:
            custom_samples_count = count
            print_success(f"Added {count} custom voice samples to training data")
        else:
            print_info("Skipping custom voice samples")
    else:
        print_info("Skipping custom voice recording - using TTS-generated samples only")
    
    # Step 4.6: Get training parameters BEFORE downloads
    print_header("Training Parameters")
    
    print_info("Number of wake word examples to generate:")
    print_info("  Controls the variety and robustness of the model")
    print_info("  - Quick test: 1,000 samples (usually produces good results)")
    print_info("  - Standard: 3,000 samples (recommended)")
    print_info("  - Best quality: 30,000-50,000 samples (often produces best results)")
    print_info("  Time: ~1-2 min per 100 samples on GPU")
    n_samples = get_user_input("Number of samples", default="3000", input_type=int)
    
    print_info("\nNumber of training steps:")
    print_info("  Controls how long to train the model")
    print_info("  - Minimum: 100 steps (required)")
    print_info("  - Quick test: 10,000 steps (usually works well)")
    print_info("  - Standard: 30,000 steps (recommended)")
    print_info("  - Best quality: 50,000+ steps (training longer usually helps)")
    print_info("  Time: ~1-2 sec per 100 steps on GPU")
    
    while True:
        training_steps = get_user_input("Training steps", default="30000", input_type=int)
        if training_steps >= 100:
            break
        print_error(f"Training steps must be at least 100 (you entered {training_steps})")
        print_info("Low step counts cause numerical errors in the warmup calculation")
    
    print_info("\nFalse activation penalty (max_negative_weight):")
    print_info("  Controls how strongly false activations are penalized")
    print_info("  Higher values = less likely to activate incorrectly")
    print_info("  - Lower (10-100): More sensitive, may trigger on similar sounds")
    print_info("  - Medium (100-1000): Balanced - good starting point")
    print_info("  - Higher (1000-5000): Stricter, fewer false activations")
    print_info("  - Very High (5000+): Maximum strictness, but may miss unclear speech with noise")
    false_activation_penalty = get_user_input("False activation penalty", default="1000", input_type=float)
    
    print(f"\n{Colors.OKBLUE}[DEBUG] About to show Background Dataset Configuration...{Colors.ENDC}")
    print(f"[DEBUG] background_status: {background_status}")
    
    # Background dataset configuration
    print_header("Background Dataset Configuration")
    print_info("Background datasets add noise/music for realistic training")
    
    # FMA music dataset
    if not background_status['fma']:
        print_info("\nFMA (Free Music Archive) - background music:")
        print_info("  1. fma_small (7.2 GB, 8,000 tracks) - Recommended")
        print_info("  2. fma_medium (22 GB, 25,000 tracks) - More variety")
        print_info("  3. Skip FMA download")
        fma_choice = get_user_input("FMA dataset choice", default="1")
    else:
        fma_choice = "0"  # Already have it
        fma_count = len(list((base_dir / "fma").rglob("*.mp3")))
        print_info(f"\nFMA already available: {fma_count} tracks")
    
    # AudioSet background noise - ALWAYS ask (even if some exist)
    audioset_path = base_dir / "audioset_16k"
    audioset_count = len(list(audioset_path.rglob("*.wav"))) if audioset_path.exists() else 0
    
    print_info("\nAudioSet - diverse environmental sounds:")
    print_info(f"  Current: {audioset_count} samples")
    print_info("  Recommended: 1,000+ samples for good quality")
    print_info("  Downloads from YouTube (requires yt-dlp and ffmpeg)")
    
    samples_needed = max(0, 1000 - audioset_count)
    default_target = str(samples_needed) if samples_needed > 0 else "0"
    audioset_target = get_user_input(
        f"Additional AudioSet samples to download (0 to skip)",
        default=default_target,
        input_type=int
    )
    
    # Step 5: Configuration Summary and Final Confirmation
    print_header("Configuration Summary")
    print(f"{Colors.BOLD}Wake Word:{Colors.ENDC} {wake_word}")
    print(f"{Colors.BOLD}Pronunciations:{Colors.ENDC} {', '.join(pronunciations)}")
    print(f"{Colors.BOLD}Samples:{Colors.ENDC} {n_samples}")
    print(f"{Colors.BOLD}Training Steps:{Colors.ENDC} {training_steps}")
    print(f"{Colors.BOLD}False Activation Penalty:{Colors.ENDC} {false_activation_penalty}")
    
    # Show background dataset choices
    fma_labels = {"0": "Already downloaded", "1": "FMA Small (7.2 GB)", "2": "FMA Medium (22 GB)", "3": "Skip"}
    print(f"{Colors.BOLD}FMA Dataset:{Colors.ENDC} {fma_labels.get(fma_choice, fma_choice)}")
    
    if audioset_target > 0:
        total_after = audioset_count + audioset_target
        print(f"{Colors.BOLD}AudioSet:{Colors.ENDC} Download {audioset_target} more (current: {audioset_count}, target: {total_after})")
    else:
        print(f"{Colors.BOLD}AudioSet:{Colors.ENDC} Current: {audioset_count} samples (no download)")
    
    print(f"{Colors.BOLD}GPU Acceleration:{Colors.ENDC} {'Yes - ' + gpu_name if has_gpu else 'No (CPU only)'}")
    
    print()
    print_info("Once you proceed, the following will run automatically:")
    print_info("  1. Download missing background datasets (if any)")
    print_info("  2. Generate wake word samples")
    print_info("  3. Augment samples with noise/reverb")
    print_info("  4. Train the model")
    print_info("  5. Export to ONNX and TFLite formats")
    print()
    print_warning("This process may take several hours depending on your settings")
    print_warning("You can leave the machine unattended - no further prompts will appear")
    print()
    
    if not get_yes_no("Proceed with automated training?", default=True):
        print_info("Training cancelled")
        sys.exit(0)
    
    # Step 6: Download background datasets if needed (fully automated, no prompts)
    print_header("Preparing Background Datasets")
    
    if not background_status['mit_rirs'] or not background_status['mit_environmental'] or not background_status['fma'] or not background_status['audioset']:
        print_info("Downloading missing background datasets automatically...")
        print_info("This improves model quality and runs unattended")
        print()
        
        # Download MIT RIRs (small, quick - direct from MIT)
        if not background_status['mit_rirs']:
            print_info("Downloading MIT Room Impulse Responses (~50MB)...")
            mit_rirs_path = base_dir / "mit_rirs"
            if download_mit_rirs(mit_rirs_path):
                mit_rirs_count = len(list(mit_rirs_path.glob("*.wav")))
                print_success(f"MIT RIRs downloaded: {mit_rirs_count} files")
            else:
                print_warning("MIT RIRs download failed - continuing without")
        
        # Download MIT Environmental (from HuggingFace - matches original Colab)
        if not background_status['mit_environmental']:
            print_info("Downloading MIT Environmental Impulse Responses from HuggingFace (~300MB)...")
            mit_env_path = base_dir / "MIT_environmental_impulse_responses"
            if download_mit_environmental(mit_env_path):
                mit_env_count = len(list(mit_env_path.glob("*.wav")))
                print_success(f"MIT Environmental downloaded: {mit_env_count} files")
            else:
                print_warning("MIT Environmental download failed - continuing without")
        
        # Download FMA based on user choice
        if not background_status['fma'] and fma_choice in ['1', '2']:
            fma_size = "small (7.2 GB, 8,000 tracks)" if fma_choice == '1' else "medium (22 GB, 25,000 tracks)"
            print_info(f"Downloading FMA {fma_size}...")
            print_warning("This will take a while depending on your internet speed")
            fma_path = base_dir / "fma"
            if download_fma(fma_path, fma_choice):
                fma_count = len(list(fma_path.rglob("*.mp3")))
                print_success(f"FMA downloaded: {fma_count} tracks")
            else:
                print_warning("FMA download failed - continuing without")
        elif fma_choice == '3':
            print_info("Skipping FMA download (user choice)")
        
        # Download AudioSet based on user's target
        if audioset_target > 0:
            audioset_path = base_dir / "audioset_16k"
            current_count = len(list(audioset_path.rglob("*.wav"))) if audioset_path.exists() else 0
            
            print_info(f"Downloading {audioset_target} additional AudioSet samples...")
            print_info(f"Current: {current_count} samples, Target: {current_count + audioset_target} total")
            print_warning("This will take several hours - downloading from YouTube")
            
            download_script = Path("scripts/download_audioset.py")
            if download_script.exists():
                import subprocess
                try:
                    # Pass total target (current + additional) since script downloads from scratch
                    total_target = current_count + audioset_target
                    result = subprocess.run(
                        [sys.executable, str(download_script), str(audioset_path), "--max-samples", str(total_target)],
                        check=False
                    )
                    
                    new_count = len(list(audioset_path.rglob("*.wav")))
                    print_success(f"AudioSet download complete: {new_count} samples")
                    
                except KeyboardInterrupt:
                    print_warning("\nAudioSet download interrupted")
                    new_count = len(list(audioset_path.rglob("*.wav")))
                    print_info(f"Partial download: {new_count} samples available")
                except Exception as e:
                    print_warning(f"AudioSet download error: {e}")
                    print_info("Continuing without AudioSet")
            else:
                print_warning("scripts/download_audioset.py not found - skipping AudioSet")
        elif audioset_target == 0 and not background_status['audioset']:
            print_info("Skipping AudioSet download (user choice)")
    else:
        print_success("All background datasets already available")
    
    # Step 7: Cleanup and create configuration
    cleanup_incomplete_training(wake_word, base_dir)
    
    config_file = create_training_config(
        wake_word=wake_word,
        n_samples=n_samples,
        training_steps=training_steps,
        false_activation_penalty=false_activation_penalty,
        base_dir=base_dir,
        use_gpu=has_gpu
    )
    
    # Step 8: Generate samples (automated, no prompts)
    print_header("Generating Wake Word Samples")
    if not generate_samples(wake_word, pronunciations, n_samples, base_dir, custom_samples_count):
        print_error("Sample generation failed - cannot continue")
        sys.exit(1)
    
    # Step 8.5: Verify all audio files are 16kHz (generated samples + background audio)
    print_header("Audio Sample Rate Verification")
    print_info("Checking and converting ALL audio files to 16kHz...")
    print_info("This includes generated TTS samples and background audio")
    print_info("Corrupted files will be automatically removed")
    
    # Create a config dict with all audio paths (generated + background)
    model_name = wake_word.lower().replace(' ', '_')
    audio_config = {
        'background_paths': [
            str(base_dir / "audioset_16k"),
            str(base_dir / "fma"),
            str(base_dir / "trained_models" / model_name / "positive_train"),
            str(base_dir / "trained_models" / model_name / "positive_test"),
            str(base_dir / "trained_models" / model_name / "negative_train"),
            str(base_dir / "trained_models" / model_name / "negative_test")
        ],
        'rir_paths': [
            str(base_dir / "mit_rirs"),
            str(base_dir / "MIT_environmental_impulse_responses")
        ]
    }
    
    if not check_and_fix_audio_sample_rates(audio_config, remove_corrupted=True):
        print_warning("Some audio files could not be converted to 16kHz")
        if not get_yes_no("Continue anyway? (may cause training errors)", default=True):
            sys.exit(1)
    
    # Step 9: Augment samples
    if not _augment_samples(config_file, base_dir):
        print_error("Sample augmentation failed")
        if not get_yes_no("Continue anyway?", default=False):
            sys.exit(1)
    
    # Step 10: Train model
    if not train_model(config_file, base_dir):
        print_error("Model training failed - cannot proceed")
        print_error("Check the training log for details")
        sys.exit(1)
    
    # Step 11: Verify training output and export models
    model_name = wake_word.lower().replace(' ', '_')
    model_dir = base_dir / "trained_models"
    
    # Check if ONNX model was created (sign of successful training)
    onnx_file = model_dir / f"{model_name}.onnx"
    if not onnx_file.exists():
        print_error(f"Training output ONNX file not found: {onnx_file}")
        print_error("Training may have failed silently - check logs")
        sys.exit(1)
    
    # Verify model file size is reasonable
    onnx_size_kb = onnx_file.stat().st_size / 1024
    if onnx_size_kb < 50:
        print_error(f"ONNX file is too small ({onnx_size_kb:.2f} KB) - likely corrupted")
        print_error("Training completed but produced invalid model")
        sys.exit(1)
    
    print_success(f"Training output verified: {onnx_file} ({onnx_size_kb:.2f} KB)")
    
    # ONNX file already exists from training, just return it
    tflite_file = model_dir / f"{model_name}.tflite"
    if tflite_file.exists():
        tflite_size_kb = tflite_file.stat().st_size / 1024
        print_success(f"TFLite model also created: {tflite_file} ({tflite_size_kb:.2f} KB)")
    else:
        tflite_file = None
        print_info("TFLite model not created (optional - ONNX is sufficient)")
    
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
