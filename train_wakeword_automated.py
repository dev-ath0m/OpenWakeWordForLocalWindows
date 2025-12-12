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
import psutil
import importlib.util
import warnings

# Suppress FutureWarning about pynvml deprecation (PyTorch CUDA still uses old import)
# The nvidia-ml-py package is installed and will be used automatically
warnings.filterwarnings('ignore', message='.*pynvml package is deprecated.*')

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
    sys.stdout.flush()  # Ensure header is displayed before any input prompts

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}[OK] {text}{Colors.ENDC}")
    sys.stdout.flush()

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.FAIL}[ERROR] {text}{Colors.ENDC}")
    sys.stdout.flush()

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}[!] {text}{Colors.ENDC}")
    sys.stdout.flush()

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.OKCYAN}[INFO] {text}{Colors.ENDC}")
    sys.stdout.flush()

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

def download_training_features(base_dir: Path) -> bool:
    """Download ACAV100M and validation features from Hugging Face"""
    print_header("Downloading Training Features")
    
    # Feature file URLs (from Hugging Face openwakeword_features dataset)
    acav100m_url = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    validation_url = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy"
    
    acav100m_file = base_dir / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    validation_file = base_dir / "validation_set_features.npy"
    
    download_errors = []
    
    # Download ACAV100M features if not present
    if not acav100m_file.exists():
        print_info("Downloading ACAV100M features (~4.7GB)...")
        print_info("This may take 10-30 minutes depending on your connection")
        print_info(f"URL: {acav100m_url}")
        
        try:
            import urllib.request
            
            def progress_callback(block_count, block_size, total_size):
                if total_size > 0:
                    downloaded = block_count * block_size
                    percent = min(100, (downloaded / total_size) * 100)
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    
                    bar_width = 50
                    filled = int(bar_width * downloaded / total_size)
                    bar = '█' * filled + '░' * (bar_width - filled)
                    
                    print(f"\r{Colors.OKCYAN}[{bar}] {percent:5.1f}% | {mb_downloaded:6.1f}/{mb_total:.1f} MB{Colors.ENDC}", 
                          end='', flush=True)
            
            urllib.request.urlretrieve(acav100m_url, str(acav100m_file), progress_callback)
            print()  # New line after progress bar
            
            # Verify file was downloaded and is not empty
            if acav100m_file.exists() and acav100m_file.stat().st_size > 0:
                print_success(f"ACAV100M features downloaded: {acav100m_file}")
            else:
                raise Exception("Downloaded file is empty or invalid")
                
        except Exception as e:
            print_error(f"Failed to download ACAV100M features: {e}")
            if acav100m_file.exists():
                acav100m_file.unlink()  # Remove partial/corrupted download
            download_errors.append(f"ACAV100M features: {e}")
    else:
        print_success(f"ACAV100M features already present: {acav100m_file}")
    
    # Download validation features if not present
    if not validation_file.exists():
        print_info("Downloading validation features (~56MB)...")
        print_info(f"URL: {validation_url}")
        
        try:
            import urllib.request
            
            def progress_callback(block_count, block_size, total_size):
                if total_size > 0:
                    downloaded = block_count * block_size
                    percent = min(100, (downloaded / total_size) * 100)
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    
                    bar_width = 50
                    filled = int(bar_width * downloaded / total_size)
                    bar = '█' * filled + '░' * (bar_width - filled)
                    
                    print(f"\r{Colors.OKCYAN}[{bar}] {percent:5.1f}% | {mb_downloaded:6.1f}/{mb_total:.1f} MB{Colors.ENDC}", 
                          end='', flush=True)
            
            urllib.request.urlretrieve(validation_url, str(validation_file), progress_callback)
            print()  # New line after progress bar
            
            # Verify file was downloaded and is not empty
            if validation_file.exists() and validation_file.stat().st_size > 0:
                print_success(f"Validation features downloaded: {validation_file}")
            else:
                raise Exception("Downloaded file is empty or invalid")
                
        except Exception as e:
            print_error(f"Failed to download validation features: {e}")
            if validation_file.exists():
                validation_file.unlink()  # Remove partial/corrupted download
            download_errors.append(f"Validation features: {e}")
    else:
        print_success(f"Validation features already present: {validation_file}")
    
    # Check if downloads failed
    if download_errors:
        print_error("\nFailed to download required training features:")
        for error in download_errors:
            print_error(f"  - {error}")
        print_info("\nYou can try:")
        print_info("  1. Check your internet connection")
        print_info("  2. Download manually from:")
        print_info(f"     ACAV100M: {acav100m_url}")
        print_info(f"     Validation: {validation_url}")
        print_info("  3. Run the script again to retry download")
        return False
    
    print_success("\nAll training features ready!")
    return True

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

def download_mit_rirs(output_path: Path) -> bool:
    """Download MIT Room Impulse Responses (~50MB, 271 files)"""
    try:
        import urllib.request
        import zipfile
        import shutil
        
        url = "https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip"
        zip_file = output_path.parent / "mit_rirs_temp.zip"
        temp_extract = output_path.parent / "mit_rirs_temp"
        
        print_info("Downloading MIT RIRs...")
        urllib.request.urlretrieve(url, str(zip_file))
        
        print_info("Extracting MIT RIRs...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_extract)
        
        # Move all WAV files from nested structure to output root
        for wav_file in temp_extract.rglob("*.wav"):
            shutil.move(str(wav_file), str(output_path / wav_file.name))
        
        # Clean up
        zip_file.unlink()
        shutil.rmtree(temp_extract)
        return True
        
    except Exception as e:
        print_error(f"Failed to download MIT RIRs: {e}")
        print_info("You can download manually from: https://mcdermottlab.mit.edu/Reverb/IR_Survey.html")
        return False

def download_mit_environmental(output_path: Path) -> bool:
    """Download MIT Environmental Impulse Responses from HuggingFace (~300MB)"""
    try:
        print_info("Loading HuggingFace datasets library...")
        import datasets
        from scipy.io import wavfile
        import numpy as np
        from tqdm import tqdm
        
        print_info("Downloading MIT Environmental dataset from HuggingFace...")
        print_info("This uses the same dataset as the original Colab training notebook")
        
        # Load dataset from HuggingFace (streaming to avoid loading all in memory)
        try:
            rir_dataset = datasets.load_dataset(
                "davidscripka/MIT_environmental_impulse_responses",
                split="train",
                streaming=True
            )
        except Exception as ds_error:
            error_msg = str(ds_error)
            if "pyarrow" in error_msg.lower() or "PyExtensionType" in error_msg:
                print_error("PyArrow compatibility issue detected")
                print_info("Attempting to fix: pip install --upgrade pyarrow")
                import subprocess
                try:
                    subprocess.run(["pip", "install", "--upgrade", "pyarrow>=12.0.0,<15.0.0"], 
                                 check=True, capture_output=True)
                    print_success("PyArrow updated, retrying download...")
                    # Retry after upgrade
                    rir_dataset = datasets.load_dataset(
                        "davidscripka/MIT_environmental_impulse_responses",
                        split="train",
                        streaming=True
                    )
                except Exception as retry_error:
                    print_error(f"Still failed after pyarrow upgrade: {retry_error}")
                    print_warning("Skipping MIT Environmental - MIT RIRs alone provides good reverb")
                    return False
            else:
                raise ds_error
        
        # Save clips to 16-bit PCM wav files
        file_count = 0
        for row in tqdm(rir_dataset, desc="Downloading files"):
            name = row['audio']['path'].split('/')[-1]
            output_file = output_path / name
            
            # Convert float32 to int16 PCM format
            audio_data = (row['audio']['array'] * 32767).astype(np.int16)
            wavfile.write(str(output_file), 16000, audio_data)
            file_count += 1
        
        print_success(f"Downloaded {file_count} MIT Environmental files")
        return True
        
    except Exception as e:
        print_error(f"Failed to download MIT Environmental: {e}")
        print_info("Make sure 'datasets' package is installed: pip install datasets")
        print_info("This dataset is optional but matches the original Colab training")
        return False

def download_fma(output_path: Path, size: str) -> bool:
    """Download FMA dataset (7.2GB for small, 22GB for medium)"""
    try:
        import urllib.request
        import zipfile
        from tqdm import tqdm
        
        if size == '1':
            url = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
            size_str = "7.2 GB"
        else:
            url = "https://os.unil.cloud.switch.ch/fma/fma_medium.zip"
            size_str = "22 GB"
        
        zip_file = output_path.parent / "fma_temp.zip"
        
        print_info(f"Downloading FMA ({size_str})...")
        print_info("This will take 10-60 minutes depending on your connection")
        print_info("Starting download (this may take a while)...")
        
        def progress_callback(block_count, block_size, total_size):
            if total_size > 0:
                downloaded = block_count * block_size
                percent = min(100, (downloaded / total_size) * 100)
                gb_downloaded = downloaded / (1024 * 1024 * 1024)
                gb_total = total_size / (1024 * 1024 * 1024)
                
                bar_width = 50
                filled = int(bar_width * downloaded / total_size)
                bar = '█' * filled + '░' * (bar_width - filled)
                
                print(f"\r{Colors.OKCYAN}[{bar}] {percent:5.1f}% | {gb_downloaded:4.2f}/{gb_total:.2f} GB{Colors.ENDC}", 
                      end='', flush=True)
        
        urllib.request.urlretrieve(url, str(zip_file), progress_callback)
        print()  # New line after progress
        
        print_info("Extracting FMA archive (this may take several minutes)...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            members = zip_ref.namelist()
            with tqdm(total=len(members), desc="Extracting", unit="files", 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} files [{elapsed}<{remaining}]') as pbar:
                for member in members:
                    zip_ref.extract(member, output_path)
                    pbar.update(1)
        
        zip_file.unlink()
        return True
        
    except Exception as e:
        print_error(f"Failed to download FMA: {e}")
        print_info("You can download manually from: https://github.com/mdeff/fma")
        return False

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
    except subprocess.CalledProcessError as e:
        print_error(f"OpenWakeWord installation failed: {e}")
        return False
    
    # Patch train.py to make Piper optional
    if not patch_openwakeword_train_script(openwakeword_dir):
        print_warning("Failed to patch train.py - Piper will be required")
        print_warning("Training may fail if Piper is not available")
    
    # Patch utils.py for Windows compatibility
    if not patch_openwakeword_utils(openwakeword_dir):
        print_warning("Failed to patch utils.py - may have file permission errors on Windows")
    
    # Patch data.py for Windows compatibility
    if not patch_openwakeword_data(openwakeword_dir):
        print_warning("Failed to patch data.py - may have file permission errors on Windows")
    
    # Patch train.py for Windows multiprocessing
    if not patch_openwakeword_train_multiprocessing(openwakeword_dir):
        print_warning("Failed to patch train.py multiprocessing - training may fail on Windows")
    
    return True

def patch_openwakeword_train_script(openwakeword_dir: Path) -> bool:
    """Patch OpenWakeWord's train.py to make Piper dependency optional"""
    train_script = openwakeword_dir / "openwakeword" / "train.py"
    
    if not train_script.exists():
        print_error(f"train.py not found at {train_script}")
        return False
    
    print_info("Patching train.py to make Piper optional...")
    
    try:
        import re
        
        # Read the file
        content = train_script.read_text(encoding='utf-8')
        original_content = content
        
        # Patch 1: Make Piper import conditional
        # Find: sys.path.insert(0, os.path.abspath(config["piper_sample_generator_path"]))
        #       from generate_samples import generate_samples
        # Replace with conditional import
        pattern1 = r'(\s+# imports Piper for synthetic sample generation\s+)sys\.path\.insert\(0, os\.path\.abspath\(config\["piper_sample_generator_path"\]\)\)\s+from generate_samples import generate_samples'
        replacement1 = r'''\1# imports Piper for synthetic sample generation (only if path is provided)
    if "piper_sample_generator_path" in config and config["piper_sample_generator_path"]:
        sys.path.insert(0, os.path.abspath(config["piper_sample_generator_path"]))
        from generate_samples import generate_samples
    else:
        # Piper not available - samples must be pre-generated
        generate_samples = None'''
        
        content = re.sub(pattern1, replacement1, content, count=1)
        
        # Patch 2: Add check before generate_samples calls
        # Pattern to find generate_samples( calls and add None check before them
        # We need to be careful to only patch calls within the --generate_clips section
        
        # For positive training samples
        pattern2 = r'(if n_current_samples <= 0\.95\*config\["n_samples"\]:\s+)(generate_samples\(\s+text=config\["target_phrase"\])'
        replacement2 = r'''\1if generate_samples is None:
                logging.error("Piper sample generator not available and positive samples not pre-generated!")
                logging.error("Please generate samples manually or provide piper_sample_generator_path in config")
                raise RuntimeError("Cannot generate clips without Piper or pre-generated samples")
            \2'''
        
        content = re.sub(pattern2, replacement2, content, count=1)
        
        # For positive test samples
        pattern3 = r'(if n_current_samples <= 0\.95\*config\["n_samples_val"\]:\s+)(generate_samples\(text=config\["target_phrase"\], max_samples=config\["n_samples_val"\])'
        replacement3 = r'''\1if generate_samples is None:
                logging.error("Piper sample generator not available and positive test samples not pre-generated!")
                raise RuntimeError("Cannot generate clips without Piper or pre-generated samples")
            \2'''
        
        content = re.sub(pattern3, replacement3, content, count=1)
        
        # For negative training samples
        pattern4 = r'(include_input_words=0\.2\)\)\s+)(generate_samples\(text=adversarial_texts, max_samples=config\["n_samples"\]-n_current_samples,\s+batch_size=config\["tts_batch_size"\]//7,)'
        replacement4 = r'''\1if generate_samples is None:
                logging.error("Piper sample generator not available and negative samples not pre-generated!")
                raise RuntimeError("Cannot generate clips without Piper or pre-generated samples")
            \2'''
        
        content = re.sub(pattern4, replacement4, content, count=1)
        
        # For negative test samples  
        pattern5 = r'(include_input_words=0\.2\)\)\s+)(generate_samples\(text=adversarial_texts, max_samples=config\["n_samples_val"\]-n_current_samples,\s+batch_size=config\["tts_batch_size"\]//7,)'
        replacement5 = r'''\1if generate_samples is None:
                logging.error("Piper sample generator not available and negative test samples not pre-generated!")
                raise RuntimeError("Cannot generate clips without Piper or pre-generated samples")
            \2'''
        
        content = re.sub(pattern5, replacement5, content, count=1)
        
        # Check if any changes were made
        if content == original_content:
            print_warning("No changes made - train.py may already be patched or format has changed")
            print_info("Manual verification recommended")
            return True  # Don't fail, just warn
        
        # Write the patched content back
        train_script.write_text(content, encoding='utf-8')
        print_success("train.py patched successfully - Piper is now optional")
        return True
        
    except Exception as e:
        print_error(f"Patching failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def patch_openwakeword_utils(openwakeword_dir: Path) -> bool:
    """Patch OpenWakeWord's utils.py to fix Windows file permission errors"""
    utils_script = openwakeword_dir / "openwakeword" / "utils.py"
    
    if not utils_script.exists():
        print_error(f"utils.py not found at {utils_script}")
        return False
    
    print_info("Patching utils.py for Windows compatibility...")
    
    try:
        content = utils_script.read_text(encoding='utf-8')
        original_content = content
        
        # Patch: Close memory-mapped file before trimming (Windows requires this)
        # Find the line: "    # Trip empty rows from the mmapped array"
        # Add cleanup before trim_mmap call
        pattern = r'(\s+fp\.flush\(\)\s+)(# Trip empty rows from the mmapped array\s+trim_mmap\(output_file\))'
        replacement = r'''\1# Close memory-mapped file before trimming (Windows requires this)
    del fp
    import gc
    gc.collect()
    
    \2'''
        
        content = content.replace(
            '    fp.flush()\n\n    # Trip empty rows from the mmapped array\n    trim_mmap(output_file)',
            '''    fp.flush()

    # Close memory-mapped file before trimming (Windows requires this)
    del fp
    import gc
    gc.collect()
    
    # Trip empty rows from the mmapped array
    trim_mmap(output_file)'''
        )
        
        if content == original_content:
            print_warning("No changes made - utils.py may already be patched or format has changed")
            return True
        
        utils_script.write_text(content, encoding='utf-8')
        print_success("utils.py patched successfully - Windows file handling fixed")
        return True
        
    except Exception as e:
        print_error(f"Patching utils.py failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def patch_openwakeword_data(openwakeword_dir: Path) -> bool:
    """Patch OpenWakeWord's data.py to fix Windows file permission errors"""
    data_script = openwakeword_dir / "openwakeword" / "data.py"
    
    if not data_script.exists():
        print_error(f"data.py not found at {data_script}")
        return False
    
    print_info("Patching data.py for Windows compatibility...")
    
    try:
        content = data_script.read_text(encoding='utf-8')
        original_content = content
        
        # Patch: Close memory-mapped files before deleting (Windows requires this)
        # Find the section in trim_mmap function where files are removed
        pattern = r'(\s+mmap_file2\.flush\(\)\s+)(# Remove old mmaped file\s+os\.remove\(mmap_path\))'
        replacement = r'''\1# Close memory-mapped files before deleting (Windows requires this)
    del mmap_file1
    del mmap_file2
    import gc
    gc.collect()
    
    \2'''
        
        content = content.replace(
            '            mmap_file2.flush()\n\n    # Remove old mmaped file\n    os.remove(mmap_path)',
            '''            mmap_file2.flush()

    # Close memory-mapped files before deleting (Windows requires this)
    del mmap_file1
    del mmap_file2
    import gc
    gc.collect()
    
    # Remove old mmaped file
    os.remove(mmap_path)'''
        )
        
        if content == original_content:
            print_warning("No changes made - data.py may already be patched or format has changed")
            return True
        
        data_script.write_text(content, encoding='utf-8')
        print_success("data.py patched successfully - Windows file handling fixed")
        return True
        
    except Exception as e:
        print_error(f"Patching data.py failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def patch_openwakeword_train_multiprocessing(openwakeword_dir: Path) -> bool:
    """Patch OpenWakeWord's train.py to fix Windows multiprocessing issues"""
    train_script = openwakeword_dir / "openwakeword" / "train.py"
    
    if not train_script.exists():
        print_error(f"train.py not found at {train_script}")
        return False
    
    print_info("Patching train.py for Windows multiprocessing compatibility...")
    
    try:
        content = train_script.read_text(encoding='utf-8')
        original_content = content
        
        # Fix 1: Replace lambda functions with named functions (for pickling)
        content = content.replace(
            '''        label_transforms = {}
        for key in ["positive"] + list(config["feature_data_files"].keys()) + ["adversarial_negative"]:
            if key == "positive":
                label_transforms[key] = lambda x: [1 for i in x]
            else:
                label_transforms[key] = lambda x: [0 for i in x]''',
            '''        # Define label transform functions (Windows-compatible, no lambda)
        def positive_label_transform(x):
            return [1 for i in x]
        
        def negative_label_transform(x):
            return [0 for i in x]
        
        label_transforms = {}
        for key in ["positive"] + list(config["feature_data_files"].keys()) + ["adversarial_negative"]:
            if key == "positive":
                label_transforms[key] = positive_label_transform
            else:
                label_transforms[key] = negative_label_transform'''
        )
        
        # Fix 2: Set num_workers=0 on Windows (multiprocessing doesn't work well)
        content = content.replace(
            '''        n_cpus = os.cpu_count()
        if n_cpus is None:
            n_cpus = 1
        else:
            n_cpus = n_cpus//2
        X_train = torch.utils.data.DataLoader(IterDataset(batch_generator),
                                              batch_size=None, num_workers=n_cpus, prefetch_factor=16)''',
            '''        # On Windows, use num_workers=0 to avoid multiprocessing pickle errors
        import platform
        if platform.system() == "Windows":
            n_workers = 0
            prefetch = None
        else:
            n_cpus = os.cpu_count()
            if n_cpus is None:
                n_workers = 1
            else:
                n_workers = n_cpus//2
            prefetch = 16
        
        X_train = torch.utils.data.DataLoader(IterDataset(batch_generator),
                                              batch_size=None, num_workers=n_workers, 
                                              prefetch_factor=prefetch)'''
        )
        
        if content == original_content:
            print_warning("No changes made - train.py may already be patched or format has changed")
            return True
        
        train_script.write_text(content, encoding='utf-8')
        print_success("train.py patched successfully - Windows multiprocessing fixed")
        return True
        
    except Exception as e:
        print_error(f"Patching train.py multiprocessing failed: {e}")
        import traceback
        traceback.print_exc()
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
    
    # Calculate validation samples (10% of training samples)
    n_samples_val = max(1, n_samples // 10)
    
    # Generate both positive and negative samples
    positive_success = _generate_positive_samples(wake_word, pronunciations, n_samples, n_samples_val, base_dir)
    if not positive_success:
        print_error("Positive sample generation failed - cannot continue")
        return False
    
    negative_success = _generate_negative_samples(wake_word, n_samples, n_samples_val, base_dir)
    if not negative_success:
        print_error("Negative sample generation failed - cannot continue")
        print_error("Training requires both positive and negative samples")
        return False
    
    return True


def _get_tts_models_config():
    """Get the unified TTS model configuration for both positive and negative samples
    
    Models are categorized by speed:
    - Fast models (1-2 sec/sample): Used for bulk generation
    - Slow models (8-12 sec/sample): Limited to 20 samples for voice variety
    """
    return [
        # Fast models (1-2 seconds per sample on GPU) - no limit
        {"model": "tts_models/en/ljspeech/vits", "gender": "female", "max_samples": None},
        {"model": "tts_models/en/ljspeech/fast_pitch", "gender": "female", "max_samples": None},
        {"model": "tts_models/en/ljspeech/glow-tts", "gender": "female", "max_samples": None},
        {"model": "tts_models/en/vctk/vits", "speaker": "p225", "gender": "female", "max_samples": None},
        {"model": "tts_models/en/vctk/vits", "speaker": "p226", "gender": "male", "max_samples": None},
        {"model": "tts_models/en/jenny/jenny", "gender": "female", "max_samples": None},
        {"model": "tts_models/multilingual/multi-dataset/your_tts", "speaker": "male-en-2", "language": "en", "max_samples": None},
        {"model": "tts_models/multilingual/multi-dataset/your_tts", "speaker": "female-en-5", "language": "en", "max_samples": None},
        
        # Slow but high-quality models (8-12 seconds per sample) - limited to 20 samples for variety
        {"model": "tts_models/en/ljspeech/tacotron2-DDC", "gender": "female", "max_samples": 20},
        {"model": "tts_models/en/ljspeech/tacotron2-DCA", "gender": "female", "max_samples": 20},
        {"model": "tts_models/en/ljspeech/neural_hmm", "gender": "female", "max_samples": 20},
        {"model": "tts_models/en/sam/tacotron-DDC", "gender": "male", "max_samples": 20},
    ]


def _setup_tts_environment(base_dir: Path):
    """Setup TTS environment and return device"""
    import os
    os.environ['TTS_HOME'] = str(base_dir)
    
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    
    # Suppress verbose TTS logging
    import logging
    logging.getLogger('TTS').setLevel(logging.CRITICAL)
    logging.getLogger('TTS.tts.utils.synthesis').setLevel(logging.CRITICAL)
    logging.getLogger('TTS.tts.models').setLevel(logging.CRITICAL)
    logging.getLogger('TTS.utils').setLevel(logging.CRITICAL)
    
    return device


class _SuppressOutput:
    """Context manager to suppress TTS library output (Windows-compatible)"""
    def __init__(self, log_file_path=None):
        self.log_file_path = log_file_path
        
    def __enter__(self):
        import sys
        import os
        
        # Save original streams
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        
        # Redirect to log file or null device using Python-level redirection only
        # (OS-level redirection with os.dup2 causes issues with tqdm on Windows)
        if self.log_file_path:
            self.redirect_file = open(self.log_file_path, 'a', encoding='utf-8', buffering=1)
            sys.stdout = self.redirect_file
            sys.stderr = self.redirect_file
        else:
            import io
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import sys
        
        # Restore original streams
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        
        # Close redirect file if it was opened
        if hasattr(self, 'redirect_file'):
            try:
                self.redirect_file.close()
            except:
                pass


def _run_subprocess_with_logging(
    command: list,
    log_file_path: Path,
    cwd: Path = None,
    description: str = "Process"
) -> bool:
    """
    Run a subprocess with real-time output streaming to console and log file.
    
    Args:
        command: Command and arguments to execute
        log_file_path: Path to log file for output capture
        cwd: Working directory for subprocess (optional)
        description: Description of the process for error messages
    
    Returns:
        True if process succeeded (exit code 0), False otherwise
    """
    import traceback
    
    try:
        print_info(f"Running {description}...")
        print_info(f"Output will be logged to: {log_file_path}")
        sys.stdout.flush()
        
        with open(log_file_path, 'w') as log_file:
            process = subprocess.Popen(
                command,
                cwd=str(cwd) if cwd else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Stream output to both console and log file in real-time
            for line in process.stdout:
                print(line, end='', flush=True)
                log_file.write(line)
                log_file.flush()
            
            process.wait()
        
        if process.returncode == 0:
            print_success(f"{description} completed successfully")
            return True
        else:
            print_error(f"{description} failed with exit code {process.returncode}")
            print_info(f"Check log file for details: {log_file_path}")
            return False
            
    except Exception as e:
        print_error(f"{description} failed: {e}")
        print_info(f"Check log file for details: {log_file_path}")
        
        # Log the exception to file
        try:
            with open(log_file_path, 'a') as f:
                f.write(f"\n\n=== Exception ===\n")
                f.write(f"Error: {str(e)}\n")
                f.write(traceback.format_exc())
        except:
            pass
        
        return False


def _get_process_usage():
    """Get current process CPU and GPU usage for progress monitoring"""
    try:
        # Get CPU usage for current process
        process = psutil.Process()
        cpu_percent = process.cpu_percent(interval=0.1)
        
        # Get GPU usage if available
        try:
            import pynvml
            gpu_util = "N/A"
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = f"{util.gpu}%"
            except:
                pass
        except ImportError:
            # pynvml not available, try nvidia-smi
            try:
                import torch
                if torch.cuda.is_available():
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits', '--id=0'],
                        capture_output=True, text=True, timeout=0.5
                    )
                    if result.returncode == 0:
                        gpu_util = f"{result.stdout.strip()}%"
                    else:
                        gpu_util = "N/A"
                else:
                    gpu_util = "N/A"
            except:
                gpu_util = "N/A"
        
        return cpu_percent, gpu_util
    except:
        return 0.0, "N/A"


def _setup_tts_environment(base_dir: Path):
    """Setup TTS environment with PyTorch 2.6 compatibility fixes"""
    import os
    os.environ['TTS_HOME'] = str(base_dir)
    
    # Fix PyTorch 2.6 weights_only issue for TTS models
    import torch
    from TTS.utils.radam import RAdam
    from collections import defaultdict
    torch.serialization.add_safe_globals([RAdam, defaultdict, dict])
    
    # Force CUDA device selection
    if torch.cuda.is_available():
        # Set default CUDA device
        torch.cuda.set_device(0)  # Use first NVIDIA GPU
        # Force PyTorch to use CUDA
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        device = "cuda:0"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        device = "cpu"
        print("CUDA not available, using CPU")
    
    # Suppress verbose TTS logging
    import logging
    logging.getLogger('TTS').setLevel(logging.CRITICAL)
    
    return device


def _generate_tts_samples_with_model(
    tts,
    model_config: dict,
    texts: list,
    output_train_dir: Path,
    output_test_dir: Path,
    n_samples: int,
    n_samples_val: int,
    train_count: int,
    test_count: int,
    failed_count: int,
    base_dir: Path,
    sample_type: str = "Positive"
) -> tuple[int, int, int]:
    """
    Generate TTS samples for a single model configuration.
    
    Args:
        tts: Loaded TTS model instance
        model_config: Model configuration dict
        texts: List of texts to synthesize
        output_train_dir: Directory for training samples
        output_test_dir: Directory for test samples
        n_samples: Total training samples needed
        n_samples_val: Total test samples needed
        train_count: Current training sample count
        test_count: Current test sample count
        failed_count: Current failed sample count
        base_dir: Base directory for project
        sample_type: "Positive" or "Negative" for progress bar label
    
    Returns:
        Tuple of (train_count, test_count, failed_count)
    """
    import numpy as np
    import librosa
    import soundfile as sf
    from tqdm import tqdm
    import sys
    
    tts_log_path = base_dir / "tts_output.log"
    model_path = model_config["model"]
    model_short = model_path.split('/')[-1]
    
    # Check if this model has a sample limit (for slow models)
    model_limit = model_config.get("max_samples", None)
    if model_limit is not None:
        model_train_target = min(model_limit, n_samples - train_count)
        model_test_target = min(model_limit // 10, n_samples_val - test_count)
    else:
        # Calculate remaining samples needed (not the global total)
        model_train_target = n_samples - train_count
        model_test_target = n_samples_val - test_count
    
    # Get native sample rate
    try:
        if hasattr(tts, 'synthesizer') and hasattr(tts.synthesizer, 'output_sample_rate'):
            native_sr = tts.synthesizer.output_sample_rate
        else:
            native_sr = 22050
    except:
        native_sr = 22050
    
    print_info(f"Starting generation with {len(texts)} text variant(s)...")
    print_info(f"Native TTS sample rate: {native_sr}Hz → Resampling to 16kHz")
    
    # Setup progress bar (show only this model's target, not global total)
    need_resample = (native_sr != 16000)
    gender = model_config.get("gender", "voice")
    speaker = model_config.get("speaker", "")
    speaker_suffix = f"_{speaker}_{gender}" if speaker else f"_{gender}"
    model_desc = f"{model_short}{speaker_suffix}"
    
    # Calculate this model's contribution to the total (not the global total)
    model_total_target = model_train_target + model_test_target
    pbar = tqdm(total=model_total_target, desc=f"{sample_type} ({model_desc})", 
               unit="samples", initial=0,  # Start at 0 for this model
               file=sys.stdout, mininterval=0.5, dynamic_ncols=True,
               bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    # Force initial display
    pbar.refresh()
    
    progress_update_interval = 10
    text_idx = 0
    model_train_count = 0
    model_test_count = 0  # Initialize test counter at start
    
    # Generate training samples
    samples_per_text = max(1, (n_samples - train_count) // len(texts))
    for text in texts:
        combo_count = 0
        while train_count < n_samples and combo_count < samples_per_text and model_train_count < model_train_target:
            output_file = output_train_dir / f"{model_short}_{train_count}.wav"
            
            try:
                # Generate with speed variation
                speed = 1.0 + np.random.uniform(-0.1, 0.1)
                
                kwargs = {
                    'text': text,
                    'speed': speed
                }
                
                if 'speaker' in model_config:
                    kwargs['speaker'] = model_config['speaker']
                if 'language' in model_config:
                    kwargs['language'] = model_config['language']
                
                # Generate directly to numpy array (in-memory, no temp file)
                with _SuppressOutput(log_file_path=str(tts_log_path)):
                    if hasattr(tts, 'tts') and callable(tts.tts):
                        audio = tts.tts(**kwargs)
                        # Convert to numpy array if it's a list
                        if isinstance(audio, list):
                            audio = np.array(audio)
                    else:
                        # Fallback to file-based if in-memory not supported
                        temp_file = output_file.parent / f"temp_{output_file.name}"
                        kwargs['file_path'] = str(temp_file)
                        tts.tts_to_file(**kwargs)
                        audio, _ = librosa.load(str(temp_file), sr=native_sr, mono=True)
                        temp_file.unlink()
                
                # Trim silence
                audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)
                
                # Skip if too long
                if len(audio_trimmed) / native_sr > 4.0:
                    failed_count += 1
                    combo_count += 1
                    continue
                
                # Resample to 16kHz only if needed
                if need_resample:
                    audio_16k = librosa.resample(audio_trimmed, orig_sr=native_sr, target_sr=16000)
                else:
                    audio_16k = audio_trimmed
                
                # Normalize
                if len(audio_16k) > 0:
                    max_val = np.abs(audio_16k).max()
                    if max_val > 0:
                        audio_16k = audio_16k / max_val * 0.95
                
                # Save directly as 16kHz PCM_16
                sf.write(str(output_file), audio_16k, 16000, subtype='PCM_16')
                
                train_count += 1
                model_train_count += 1
                combo_count += 1
                
            except Exception as e:
                # Log detailed error to file for debugging
                with open(tts_log_path, 'a') as f:
                    import traceback
                    f.write(f"\n=== Error generating {sample_type.lower()} sample {train_count} ===\n")
                    f.write(f"Model: {model_short}, Text: {text}\n")
                    f.write(f"Error: {str(e)}\n")
                    f.write(traceback.format_exc())
                    f.write("\n")
                failed_count += 1
                combo_count += 1
                if output_file.exists():
                    output_file.unlink()
            
            # Update progress every 10 attempts (successful or failed) - runs regardless of exception
            total_attempts = model_train_count + model_test_count + failed_count
            if total_attempts == 1 or total_attempts % 10 == 0:
                cpu_usage, gpu_usage = _get_process_usage()
                success_rate = ((model_train_count + model_test_count) / total_attempts * 100) if total_attempts > 0 else 0
                pbar.n = model_train_count + model_test_count
                pbar.set_postfix({
                    'Train': model_train_count,
                    'Test': model_test_count,
                    'Success': f'{success_rate:.1f}%',
                    'CPU': f'{cpu_usage:.0f}%',
                    'GPU': gpu_usage
                })
                pbar.refresh()
    
    # Generate test samples
    text_idx = 0
    test_samples_per_text = max(1, model_test_target // len(texts))
    
    while test_count < n_samples_val and model_test_count < model_test_target and text_idx < len(texts) * 2:
        text = texts[text_idx % len(texts)]
        text_idx += 1
        
        output_file = output_test_dir / f"{model_short}_test_{test_count}.wav"
        temp_file = output_file.parent / f"temp_{output_file.name}"
        
        try:
            # Generate with speed variation
            speed = 1.0 + np.random.uniform(-0.1, 0.1)
            
            kwargs = {
                'text': text,
                'file_path': str(temp_file),
                'speed': speed
            }
            
            if 'speaker' in model_config:
                kwargs['speaker'] = model_config['speaker']
            if 'language' in model_config:
                kwargs['language'] = model_config['language']
            
            # Generate sample with suppressed output
            with _SuppressOutput(log_file_path=str(tts_log_path)):
                tts.tts_to_file(**kwargs)
            
            # Load, resample, trim, and save in one efficient step
            audio, sr = librosa.load(str(temp_file), sr=native_sr, mono=True)
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)
            
            if len(audio_trimmed) / native_sr > 4.0:
                temp_file.unlink()
                failed_count += 1
                continue
            
            if native_sr != 16000:
                audio_16k = librosa.resample(audio_trimmed, orig_sr=native_sr, target_sr=16000)
            else:
                audio_16k = audio_trimmed
            
            if len(audio_16k) > 0:
                max_val = np.abs(audio_16k).max()
                if max_val > 0:
                    audio_16k = audio_16k / max_val * 0.95
            
            sf.write(str(output_file), audio_16k, 16000, subtype='PCM_16')
            temp_file.unlink()
            
            test_count += 1
            model_test_count += 1
            
        except Exception as e:
            # Log detailed error to file for debugging
            with open(tts_log_path, 'a') as f:
                import traceback
                f.write(f"\n=== Error generating {sample_type.lower()} test sample {test_count} ===\n")
                f.write(f"Model: {model_short}, Text: {text}\n")
                f.write(f"Error: {str(e)}\n")
                f.write(traceback.format_exc())
                f.write("\n")
            failed_count += 1
            if temp_file.exists():
                temp_file.unlink()
            if output_file.exists():
                output_file.unlink()
        
        # Update progress every 10 attempts (successful or failed) - runs regardless of exception
        total_attempts = model_train_count + model_test_count + failed_count
        if total_attempts % 10 == 0 or total_attempts == 1:
            cpu_usage, gpu_usage = _get_process_usage()
            success_rate = ((model_train_count + model_test_count) / total_attempts * 100) if total_attempts > 0 else 0
            pbar.n = model_train_count + model_test_count
            pbar.set_postfix({
                'Train': model_train_count,
                'Test': model_test_count,
                'Success': f'{success_rate:.1f}%',
                'CPU': f'{cpu_usage:.0f}%',
                'GPU': gpu_usage
            })
            pbar.refresh()
    
    # Close progress bar
    pbar.close()
    
    # Report completion using model-specific counts
    if failed_count > 0:
        total_attempts = model_train_count + model_test_count + failed_count
        fail_pct = (failed_count / total_attempts * 100) if total_attempts > 0 else 0
        print_success(f"Completed model '{model_short}': {model_train_count} train + {model_test_count} test, {failed_count} failed ({fail_pct:.1f}%)")
    else:
        print_success(f"Completed model '{model_short}': {model_train_count} train + {model_test_count} test")
    
    return train_count, test_count, failed_count


def _generate_positive_samples(wake_word: str, pronunciations: list, n_samples: int, n_samples_val: int, base_dir: Path) -> bool:
    """Generate positive samples (wake word pronunciations) using TTS"""
    print_header("Generating Positive Samples")
    
    model_name = wake_word.lower().replace(' ', '_')
    output_dir = base_dir / "trained_models" / model_name
    positive_train_dir = output_dir / "positive_train"
    positive_test_dir = output_dir / "positive_test"
    positive_train_dir.mkdir(parents=True, exist_ok=True)
    positive_test_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from TTS.api import TTS
        import numpy as np
        
        # Setup environment
        device = _setup_tts_environment(base_dir)
        tts_models = _get_tts_models_config()
        
        # Calculate samples per model
        samples_per_model = (n_samples // len(tts_models)) + 1
        samples_per_combo = max(1, samples_per_model // len(pronunciations))
        
        print_info(f"Using {len(tts_models)} TTS models with {len(pronunciations)} pronunciations")
        print_info(f"Target: {n_samples} training samples + {n_samples_val} test samples")
        print_info(f"Generating ~{samples_per_combo} samples per model/variation combination")
        print_info(f"Device: {device.upper()}")
        print_info("Starting sample generation...\n")
        
        train_count = 0
        test_count = 0
        failed_count = 0
        
        for model_config in tts_models:
            model_path = model_config["model"]
            model_short = model_path.split('/')[-1]
            
            # Clear any previous progress line
            print("\r" + " " * 120 + "\r", end='', flush=True)
            print_info(f"Loading model: {model_short}...")
            
            try:
                # Setup TTS log file for redirecting verbose output
                tts_log_path = base_dir / "tts_output.log"
                
                # Redirect TTS library logging to file
                import logging
                tts_logger = logging.getLogger('TTS')
                tts_logger.setLevel(logging.WARNING)
                file_handler = logging.FileHandler(tts_log_path, mode='a')
                file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                tts_logger.addHandler(file_handler)
                
                # Load TTS model with GPU support (suppress verbose output)
                import warnings
                warnings.filterwarnings('ignore', message='.*gpu.*will be deprecated.*')
                
                # Suppress TTS library output during model loading
                with _SuppressOutput(log_file_path=str(tts_log_path)):
                    tts = TTS(model_name=model_path)
                
                # Manually force synthesizer and vocoder to GPU
                if device == "cuda:0":
                    if hasattr(tts, 'synthesizer') and tts.synthesizer is not None:
                        if hasattr(tts.synthesizer, 'tts_model'):
                            tts.synthesizer.tts_model = tts.synthesizer.tts_model.to(device)
                    if hasattr(tts, 'vocoder') and tts.vocoder is not None:
                        if hasattr(tts.vocoder, 'model'):
                            tts.vocoder.model = tts.vocoder.model.to(device)
                
                print_success(f"Model {model_short} initialized on {device.upper()}\")")
                print_success(f"Model {model_short} loaded successfully on {device.upper()}")
                
                # Use common function to generate samples
                train_count, test_count, failed_count = _generate_tts_samples_with_model(
                    tts=tts,
                    model_config=model_config,
                    texts=pronunciations,
                    output_train_dir=positive_train_dir,
                    output_test_dir=positive_test_dir,
                    n_samples=n_samples,
                    n_samples_val=n_samples_val,
                    train_count=train_count,
                    test_count=test_count,
                    failed_count=failed_count,
                    base_dir=base_dir,
                    sample_type="Positive"
                )
                        
            except Exception as e:
                # Log full error details to file
                tts_log_path = base_dir / "tts_output.log"
                with open(tts_log_path, 'a') as f:
                    import traceback
                    f.write(f"\n=== Model {model_short} failed ===\n")
                    f.write(f"Error: {str(e)}\n")
                    f.write(traceback.format_exc())
                    f.write("\n")
                print_warning(f"Model {model_short} failed: {e}")
                print_info(f"Full error details logged to: {tts_log_path}")
                continue
            
            if train_count >= n_samples and test_count >= n_samples_val:
                break
        
        print_success(f"\nGenerated {train_count} training samples and {test_count} test samples")
        total_valid = train_count + test_count
        total_attempts = total_valid + failed_count
        if failed_count > 0:
            fail_pct = (failed_count / total_attempts * 100)
            print_info(f"Total attempts: {total_attempts} (Success rate: {100-fail_pct:.1f}%)")
        print_info(f"TTS models cached in: {base_dir / 'tts'}")
        return train_count > 0 and test_count > 0
        
    except Exception as e:
        print_error(f"Positive sample generation failed: {e}")
        return False


def _generate_negative_samples(wake_word: str, n_samples: int, n_samples_val: int, base_dir: Path) -> bool:
    """Generate negative samples (phonetically similar non-wake-words) using TTS as Piper replacement"""
    print_header("Generating Negative Samples")
    
    model_name = wake_word.lower().replace(' ', '_')
    output_dir = base_dir / "trained_models" / model_name
    negative_train_dir = output_dir / "negative_train"
    negative_test_dir = output_dir / "negative_test"
    negative_train_dir.mkdir(parents=True, exist_ok=True)
    negative_test_dir.mkdir(parents=True, exist_ok=True)
    
    print_info("Generating phonetically similar adversarial phrases")
    print_info("These teach the model what NOT to trigger on")
    print_info(f"Target: {n_samples} training samples + {n_samples_val} test samples")
    
    try:
        # Import OpenWakeWord's adversarial text generator
        import sys
        openwakeword_path = base_dir / "openwakeword"
        if str(openwakeword_path) not in sys.path:
            sys.path.insert(0, str(openwakeword_path))
        
        from openwakeword.data import generate_adversarial_texts
        
        # Patch torch.load to use weights_only=False for deep-phonemizer compatibility
        # This is safe as we're loading from the trusted deep-phonemizer package
        print_info(f"Analyzing phonemes for '{wake_word}'...")
        print_info("Note: Temporarily allowing unsafe pickle loading for deep-phonemizer (PyTorch 2.6 compatibility)")
        
        import torch
        original_torch_load = torch.load
        
        def patched_torch_load(*args, **kwargs):
            """Temporary patch for PyTorch 2.6 compatibility with deep-phonemizer"""
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        
        try:
            # Apply patch
            torch.load = patched_torch_load
            
            # Generate adversarial texts with patched torch.load
            print_info("Generating adversarial text phrases...")
            print_info("This may take 1-2 minutes on first run (downloading phonemizer models)")
            print_info("Please wait...")
            
            adversarial_texts = generate_adversarial_texts(
                input_text=wake_word,
                N=n_samples,
                include_partial_phrase=1.0,  # Include partial phrases (e.g., "ho" from "homie")
                include_input_words=0.2      # Sometimes include actual wake word parts
            )
            
            print_success("Adversarial text generation complete!")
        finally:
            # Restore original torch.load
            torch.load = original_torch_load
        
        print_success(f"Generated {len(adversarial_texts)} phonetically similar phrases")
        print_info(f"Examples: {', '.join(adversarial_texts[:5])}")
        
        # Now generate audio samples for these adversarial texts using TTS
        from TTS.api import TTS
        import librosa
        from scipy.io import wavfile
        import numpy as np
        
        # Setup TTS environment
        device = _setup_tts_environment(base_dir)
        
        # Setup TTS log file for redirecting verbose output
        tts_log_path = base_dir / "tts_output.log"
        
        # Redirect TTS library logging to file
        import logging
        tts_logger = logging.getLogger('TTS')
        tts_logger.setLevel(logging.WARNING)
        file_handler = logging.FileHandler(tts_log_path, mode='a')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        tts_logger.addHandler(file_handler)
        
        # Use the same TTS models as positive sample generation
        tts_models_config = _get_tts_models_config()
        
        train_count = 0
        test_count = 0
        failed_count = 0
        
        # Generate samples using each model configuration
        import threading
        import time as time_module
        
        for model_config in tts_models_config:
            if train_count >= n_samples and test_count >= n_samples_val:
                break
            
            # Check if this model has a sample limit (for slow models)
            model_limit = model_config.get("max_samples", None)
            if model_limit is not None:
                model_train_target = min(model_limit, n_samples - train_count)
                model_test_target = min(model_limit // 10, n_samples_val - test_count)
            else:
                model_train_target = n_samples
                model_test_target = n_samples_val
            
            model_name = model_config["model"]
            model_short = model_name.split('/')[-1]
            
            try:
                # Load model with spinner feedback (same as positive samples)
                init_done = threading.Event()
                def init_spinner():
                    spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
                    idx = 0
                    start_time = time_module.time()
                    while not init_done.is_set():
                        elapsed = time_module.time() - start_time
                        print(f"\r{Colors.OKCYAN}  {spinner_chars[idx]} Loading {model_short}... (elapsed: {int(elapsed)}s){Colors.ENDC}", 
                              end='', flush=True)
                        idx = (idx + 1) % len(spinner_chars)
                        time_module.sleep(0.1)
                    print("\r" + " " * 80 + "\r", end='', flush=True)
                
                init_thread = threading.Thread(target=init_spinner)
                init_thread.daemon = True
                init_thread.start()
                
                # Load model without deprecated gpu parameter (suppress verbose output)
                import warnings
                warnings.filterwarnings('ignore', message='.*gpu.*will be deprecated.*')
                
                # Suppress TTS library output during model loading
                with _SuppressOutput(log_file_path=str(tts_log_path)):
                    tts = TTS(model_name=model_name)
                
                init_done.set()
                init_thread.join(timeout=0.5)
                
                # GPU transfer spinner (note: TTS uses gpu=True in constructor)
                gpu_done = threading.Event()
                def gpu_spinner():
                    spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
                    idx = 0
                    start_time = time_module.time()
                    while not gpu_done.is_set():
                        elapsed = time_module.time() - start_time
                        print(f"\r{Colors.OKCYAN}  {spinner_chars[idx]} Model ready on {device.upper()}... (elapsed: {int(elapsed)}s){Colors.ENDC}", 
                              end='', flush=True)
                        idx = (idx + 1) % len(spinner_chars)
                        time_module.sleep(0.1)
                    print("\r" + " " * 80 + "\r", end='', flush=True)
                
                gpu_thread = threading.Thread(target=gpu_spinner)
                gpu_thread.daemon = True
                gpu_thread.start()
                
                # Manually force synthesizer and vocoder to GPU
                if device == "cuda:0":
                    if hasattr(tts, 'synthesizer') and tts.synthesizer is not None:
                        if hasattr(tts.synthesizer, 'tts_model'):
                            tts.synthesizer.tts_model = tts.synthesizer.tts_model.to(device)
                    if hasattr(tts, 'vocoder') and tts.vocoder is not None:
                        if hasattr(tts.vocoder, 'model'):
                            tts.vocoder.model = tts.vocoder.model.to(device)
                
                gpu_done.set()
                gpu_thread.join(timeout=0.5)
                
                print_success(f"Model {model_short} loaded successfully on {device.upper()}")
                
                # Get native sample rate
                try:
                    if hasattr(tts, 'synthesizer') and hasattr(tts.synthesizer, 'output_sample_rate'):
                        native_sr = tts.synthesizer.output_sample_rate
                    else:
                        native_sr = 22050
                except:
                    native_sr = 22050
                
                # Initialize progress bar for this model
                from tqdm import tqdm
                import librosa
                import soundfile as sf
                
                gender = model_config.get("gender", "voice")
                speaker = model_config.get("speaker", "")
                speaker_suffix = f"_{speaker}_{gender}" if speaker else f"_{gender}"
                model_desc = f"{model_short}{speaker_suffix}"
                total_target = n_samples + n_samples_val
                pbar = tqdm(total=total_target, desc=f"Negative ({model_desc})", 
                           unit="samples", initial=0,
                           file=sys.stdout, mininterval=0.5, dynamic_ncols=True,
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
                
                try:
                    # Cycle through adversarial texts for training samples
                    text_idx = 0
                    model_train_count = 0  # Track samples for this model
                    while train_count < n_samples and text_idx < len(adversarial_texts) and model_train_count < model_train_target:
                        text = adversarial_texts[text_idx % len(adversarial_texts)]
                        text_idx += 1
                        
                        output_file = negative_train_dir / f"neg_{train_count}.wav"
                        temp_file = output_file.parent / f"temp_{output_file.name}"
                        
                        try:
                            # Generate with speed variation
                            speed = 1.0 + np.random.uniform(-0.1, 0.1)
                            
                            kwargs = {
                                'text': text,
                                'file_path': str(temp_file),
                                'speed': speed
                            }
                            
                            if 'speaker' in model_config:
                                kwargs['speaker'] = model_config['speaker']
                            if 'language' in model_config:
                                kwargs['language'] = model_config['language']
                            
                            # Generate sample with suppressed output
                            with _SuppressOutput():
                                tts.tts_to_file(**kwargs)
                            
                            # Load, resample, trim, and save
                            audio, sr = librosa.load(str(temp_file), sr=native_sr, mono=True)
                            audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)
                            
                            if len(audio_trimmed) / native_sr > 4.0:
                                temp_file.unlink()
                                failed_count += 1
                                continue
                            
                            if native_sr != 16000:
                                audio_16k = librosa.resample(audio_trimmed, orig_sr=native_sr, target_sr=16000)
                            else:
                                audio_16k = audio_trimmed
                            
                            if len(audio_16k) > 0:
                                max_val = np.abs(audio_16k).max()
                                if max_val > 0:
                                    audio_16k = audio_16k / max_val * 0.95
                            
                            sf.write(str(output_file), audio_16k, 16000, subtype='PCM_16')
                            temp_file.unlink()
                            
                            train_count += 1
                            model_train_count += 1
                            
                            # Update progress on first sample and every 10 samples to reduce overhead
                            if model_train_count == 1 or model_train_count % 10 == 0:
                                cpu_usage, gpu_usage = _get_process_usage()
                                text_display = text if len(text) <= 30 else text[:27] + '...'
                                pbar.n = model_train_count + model_test_count
                                pbar.set_postfix({
                                    'Train': model_train_count,
                                    'Test': model_test_count,
                                    'Failed': failed_count,
                                    'CPU': f'{cpu_usage:.0f}%',
                                    'GPU': gpu_usage,
                                    'Text': text_display
                                })
                                pbar.refresh()
                            
                        except Exception as e:
                            # Log detailed error to file for debugging
                            with open(tts_log_path, 'a') as f:
                                import traceback
                                f.write(f"\n=== Error generating negative sample {train_count} ===\n")
                                f.write(f"Model: {model_short}, Text: {text}\n")
                                f.write(f"Error: {str(e)}\n")
                                f.write(traceback.format_exc())
                                f.write("\n")
                            failed_count += 1
                            if temp_file.exists():
                                temp_file.unlink()
                            if output_file.exists():
                                output_file.unlink()
                            continue
                    
                    # Generate test samples
                    text_idx = 0
                    model_test_count = 0  # Track test samples for this model
                    while test_count < n_samples_val and text_idx < len(adversarial_texts) * 2 and model_test_count < model_test_target:
                        text = adversarial_texts[text_idx % len(adversarial_texts)]
                        text_idx += 1
                        
                        output_file = negative_test_dir / f"neg_test_{test_count}.wav"
                        temp_file = output_file.parent / f"temp_{output_file.name}"
                        
                        try:
                            # Generate with speed variation
                            speed = 1.0 + np.random.uniform(-0.1, 0.1)
                            
                            kwargs = {
                                'text': text,
                                'file_path': str(temp_file),
                                'speed': speed
                            }
                            
                            if 'speaker' in model_config:
                                kwargs['speaker'] = model_config['speaker']
                            if 'language' in model_config:
                                kwargs['language'] = model_config['language']
                            
                            # Generate sample with suppressed output
                            with _SuppressOutput():
                                tts.tts_to_file(**kwargs)
                            
                            # Load, resample, trim, and save
                            audio, sr = librosa.load(str(temp_file), sr=native_sr, mono=True)
                            audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)
                            
                            if len(audio_trimmed) / native_sr > 4.0:
                                temp_file.unlink()
                                failed_count += 1
                                continue
                            
                            if native_sr != 16000:
                                audio_16k = librosa.resample(audio_trimmed, orig_sr=native_sr, target_sr=16000)
                            else:
                                audio_16k = audio_trimmed
                            
                            if len(audio_16k) > 0:
                                max_val = np.abs(audio_16k).max()
                                if max_val > 0:
                                    audio_16k = audio_16k / max_val * 0.95
                            
                            sf.write(str(output_file), audio_16k, 16000, subtype='PCM_16')
                            temp_file.unlink()
                            
                            test_count += 1
                            model_test_count += 1
                            
                            # Update progress bar (test samples are fewer, update every time)
                            cpu_usage, gpu_usage = _get_process_usage()
                            text_display = text if len(text) <= 30 else text[:27] + '...'
                            pbar.n = model_train_count + model_test_count
                            pbar.set_postfix({
                                'Train': model_train_count,
                                'Test': model_test_count,
                                'Failed': failed_count,
                                'CPU': f'{cpu_usage:.0f}%',
                                'GPU': gpu_usage,
                                'Text': text_display
                            })
                            pbar.refresh()
                            
                        except Exception as e:
                            # Log detailed error to file for debugging
                            with open(tts_log_path, 'a') as f:
                                import traceback
                                f.write(f"\n=== Error generating negative test sample {test_count} ===\n")
                                f.write(f"Model: {model_short}, Text: {text}\n")
                                f.write(f"Error: {str(e)}\n")
                                f.write(traceback.format_exc())
                                f.write("\n")
                            failed_count += 1
                            if temp_file.exists():
                                temp_file.unlink()
                            if output_file.exists():
                                output_file.unlink()
                            continue
                    
                    if train_count >= n_samples and test_count >= n_samples_val:
                        break
                    
                    # Close progress bar
                    pbar.close()
                        
                except Exception as model_error:
                    # Model-level error (not individual sample error)
                    # Close progress bar on error
                    if 'pbar' in locals():
                        pbar.close()
                    # Log full error details to file
                    with open(tts_log_path, 'a') as f:
                        import traceback
                        f.write(f"\n=== Negative samples: Model {model_short} failed ===\n")
                        f.write(f"Error: {str(model_error)}\n")
                        f.write(traceback.format_exc())
                        f.write("\n")
                    print_warning(f"Model {model_short} failed: {model_error}")
                    print_info(f"Full error details logged to: {tts_log_path}")
                    continue  # Try next model instead of aborting
                
            except Exception as e:
                # Model loading error
                print_warning(f"Model {model_short} failed to load: {e}")
                continue
            
            if train_count >= n_samples and test_count >= n_samples_val:
                break
        
        # Final validation - require at least 80% of target samples (allow some model failures)
        total_valid = train_count + test_count
        total_target = n_samples + n_samples_val
        if total_valid < total_target * 0.8:
            print_error(f"\nInsufficient negative samples generated: {train_count} train + {test_count} test / {total_target} target")
            print_error("Training requires at least 80% of target samples")
            return False
        
        print_success(f"\nGenerated {train_count} training samples and {test_count} test samples")
        if failed_count > 0:
            total_attempts = total_valid + failed_count
            print_info(f"Failed: {failed_count} samples ({failed_count/total_attempts*100:.1f}%)")
        
        if train_count < n_samples or test_count < n_samples_val:
            print_warning(f"Generated {train_count}/{n_samples} train + {test_count}/{n_samples_val} test (some models may have failed)")
        
        return True
        
    except Exception as e:
        print_error(f"Negative sample generation failed: {e}")
        print_error("This is a critical error - training cannot proceed without negative samples")
        print_info("The error is likely due to PyTorch 2.6 changing torch.load security defaults")
        print_info("Possible solutions:")
        print_info("  1. Downgrade PyTorch to 2.5 or earlier")
        print_info("  2. Wait for deep-phonemizer to update their checkpoint format")
        import traceback
        traceback.print_exc()
        return False


def _patch_openwakeword_train_py(base_dir: Path) -> bool:
    """Apply runtime patches to OpenWakeWord's train.py for compatibility fixes.
    
    Patches:
    1. Fix TFLite conversion for TensorFlow 2.16+ (use onnx2tf instead of onnx-tf)
    2. Fix Windows file locking in trim_mmap function
    
    Returns:
        True if patching succeeded, False otherwise
    """
    train_py_path = base_dir / "openwakeword" / "openwakeword" / "train.py"
    data_py_path = base_dir / "openwakeword" / "openwakeword" / "data.py"
    
    if not train_py_path.exists():
        print_error(f"Cannot patch: {train_py_path} not found")
        return False
    
    if not data_py_path.exists():
        print_error(f"Cannot patch: {data_py_path} not found")
        return False
    
    try:
        # Read train.py
        with open(train_py_path, 'r', encoding='utf-8') as f:
            train_content = f.read()
        
        # Read data.py
        with open(data_py_path, 'r', encoding='utf-8') as f:
            data_content = f.read()
        
        # Check if patches already applied
        if 'onnx2tf' in train_content and 'explicit close + gc' in data_content:
            return True  # Already patched
        
        print_info("Applying compatibility patches to OpenWakeWord...")
        
        # Patch 1: Fix TFLite conversion in train.py
        old_convert_func = '''# Separate function to convert onnx models to tflite format
def convert_onnx_to_tflite(onnx_model_path, output_path):
    """Converts an ONNX version of an openwakeword model to the Tensorflow tflite format."""
    # imports
    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf

    # Convert to tflite from onnx model
    onnx_model = onnx.load(onnx_model_path)
    tf_rep = prepare(onnx_model, device="CPU")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tf_rep.export_graph(os.path.join(tmp_dir, "tf_model"))
        converter = tf.lite.TFLiteConverter.from_saved_model(os.path.join(tmp_dir, "tf_model"))
        tflite_model = converter.convert()

        logging.info(f"####\\nSaving tflite mode to '{output_path}'")
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

    return None'''
        
        new_convert_func = '''# Separate function to convert onnx models to tflite format
def convert_onnx_to_tflite(onnx_model_path, output_path):
    """Converts an ONNX version of an openwakeword model to the Tensorflow tflite format.
    
    Uses onnx2tf for conversion (compatible with TensorFlow 2.16+).
    Falls back to onnx-tf if onnx2tf fails (requires TF 2.12-2.14).
    """
    import tensorflow as tf
    
    # Try onnx2tf first (compatible with TF 2.16+)
    try:
        import onnx2tf
        import shutil
        
        logging.info(f"Converting {os.path.basename(onnx_model_path)} to TFLite using onnx2tf...")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # onnx2tf creates output files in a directory
            # Use minimal parameters - API varies by version
            onnx2tf.convert(
                input_onnx_file_path=onnx_model_path,
                output_folder_path=tmp_dir
            )
            
            # Find the generated tflite file
            import glob
            tflite_files = glob.glob(os.path.join(tmp_dir, '*.tflite'))
            if tflite_files:
                shutil.copy(tflite_files[0], output_path)
                logging.info(f"####\\nSaving tflite model to '{output_path}'")
                return None
            else:
                raise RuntimeError("onnx2tf did not generate a TFLite file")
                
    except Exception as e:
        logging.warning(f"onnx2tf conversion failed: {e}")
        logging.info("Attempting fallback to onnx-tf (requires TensorFlow 2.12-2.14)...")
        
        # Fallback to onnx-tf (only works with TF 2.12-2.14)
        try:
            import onnx
            from onnx_tf.backend import prepare
            
            onnx_model = onnx.load(onnx_model_path)
            tf_rep = prepare(onnx_model, device="CPU")
            with tempfile.TemporaryDirectory() as tmp_dir:
                tf_rep.export_graph(os.path.join(tmp_dir, "tf_model"))
                converter = tf.lite.TFLiteConverter.from_saved_model(os.path.join(tmp_dir, "tf_model"))
                tflite_model = converter.convert()

                logging.info(f"####\\nSaving tflite model to '{output_path}'")
                with open(output_path, 'wb') as f:
                    f.write(tflite_model)
                    
        except Exception as fallback_error:
            logging.error(f"Both onnx2tf and onnx-tf conversion failed.")
            logging.error(f"onnx-tf error: {fallback_error}")
            logging.info("TFLite conversion requires either:")
            logging.info("  - onnx2tf with TensorFlow 2.16+ (pip install onnx2tf)")
            logging.info("  - onnx-tf with TensorFlow 2.12-2.14 (pip install onnx-tf)")
            raise RuntimeError("TFLite conversion failed with all methods") from fallback_error

    return None'''
        
        if old_convert_func in train_content:
            train_content = train_content.replace(old_convert_func, new_convert_func)
            print_success("  ✓ Patched TFLite conversion for TensorFlow 2.16 compatibility")
        else:
            print_warning("  ! TFLite conversion function not found (may already be patched)")
        
        # Patch 2: Fix Windows file locking in data.py
        old_trim_code = '''    # Close memory-mapped files before deleting (Windows requires this)
    del mmap_file1
    del mmap_file2
    import gc
    gc.collect()
    
    # Remove old mmaped file
    os.remove(mmap_path)'''
        
        new_trim_code = '''    # Close memory-mapped files before deleting (Windows requires explicit close + gc)
    if hasattr(mmap_file1, '_mmap'):
        mmap_file1._mmap.close()
    if hasattr(mmap_file2, '_mmap'):
        mmap_file2._mmap.close()
    del mmap_file1
    del mmap_file2
    import gc
    gc.collect()
    
    # Windows file locking: retry with delays
    import time
    max_retries = 5
    for attempt in range(max_retries):
        try:
            os.remove(mmap_path)
            break
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(0.5)  # Wait for file handle to release
                gc.collect()  # Force garbage collection again
            else:
                raise  # Re-raise on final attempt'''
        
        if old_trim_code in data_content:
            data_content = data_content.replace(old_trim_code, new_trim_code)
            print_success("  ✓ Patched Windows file locking in trim_mmap")
        else:
            print_warning("  ! File locking code not found (may already be patched)")
        
        # Write patched files
        with open(train_py_path, 'w', encoding='utf-8') as f:
            f.write(train_content)
        
        with open(data_py_path, 'w', encoding='utf-8') as f:
            f.write(data_content)
        
        print_success("OpenWakeWord compatibility patches applied successfully")
        return True
        
    except Exception as e:
        print_error(f"Failed to patch OpenWakeWord: {e}")
        import traceback
        traceback.print_exc()
        return False


def augment_samples(config_file: Path, base_dir: Path) -> bool:
    """Augment samples with background noise and room impulse responses"""
    print_header("Augmenting Samples")
    
    # Apply compatibility patches to OpenWakeWord
    if not _patch_openwakeword_train_py(base_dir):
        print_error("Failed to apply OpenWakeWord patches")
        return False
    
    print_info("Applying audio augmentation (noise, music, reverb)")
    print_info("This may take several minutes...")
    
    openwakeword_dir = base_dir / "openwakeword"
    train_script = openwakeword_dir / "openwakeword" / "train.py"
    
    if not train_script.exists():
        print_error(f"Training script not found: {train_script}")
        return False
    
    aug_log_path = base_dir / "augmentation_output.log"
    
    return _run_subprocess_with_logging(
        command=[sys.executable, str(train_script),
                '--training_config', str(config_file),
                '--augment_clips'],
        log_file_path=aug_log_path,
        cwd=openwakeword_dir,
        description="Augmentation"
    )

def train_model(config_file: Path, base_dir: Path) -> bool:
    """Train the wake word model with progress monitoring"""
    print_header("Training Model")
    
    # Apply compatibility patches to OpenWakeWord
    if not _patch_openwakeword_train_py(base_dir):
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
    
    result = _run_subprocess_with_logging(
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



def check_and_fix_audio_sample_rates(config: dict, remove_corrupted: bool = True) -> bool:
    """Check and convert all background/RIR files to 16kHz using parallel processing
    
    Args:
        config: Configuration dict with 'background_paths' and 'rir_paths'
        remove_corrupted: If True, delete files that cannot be loaded or converted
    
    Returns:
        True if all files were processed successfully, False if errors occurred
    """
    from audio_utils import scan_and_convert_audio_files
    
    print_header("Checking Audio File Sample Rates")
    
    target_sr = 16000
    all_audio_paths = []
    
    # Collect all audio file paths from config
    if 'background_paths' in config:
        for bg_path in config['background_paths']:
            bg_path = Path(bg_path)
            if bg_path.exists():
                all_audio_paths.extend(list(bg_path.glob("**/*.wav")))
                all_audio_paths.extend(list(bg_path.glob("**/*.mp3")))
    
    if 'rir_paths' in config:
        for rir_path in config['rir_paths']:
            rir_path = Path(rir_path)
            if rir_path.exists():
                all_audio_paths.extend(list(rir_path.glob("**/*.wav")))
    
    if not all_audio_paths:
        print_success("No audio files found to check")
        return True
    
    print_info(f"Found {len(all_audio_paths)} audio files to check")
    
    try:
        # Use shared utility for scanning and conversion
        results = scan_and_convert_audio_files(
            audio_files=all_audio_paths,
            target_sr=target_sr,
            max_workers=None,  # Auto-detect optimal workers
            remove_corrupted=remove_corrupted,
            show_progress=True
        )
        
        # Report results
        print_success(f"Processed {results['total']} files:")
        print_info(f"  - Already correct: {results['already_correct']}")
        print_info(f"  - Converted: {results['converted']}")
        if results['corrupted'] > 0:
            status = "removed" if remove_corrupted else "found"
            print_warning(f"  - Corrupted ({status}): {results['corrupted']}")
        
        return True
    except KeyboardInterrupt:
        print_warning("\n\nAudio conversion interrupted by user")
        print_info("Training cannot continue without proper audio file conversion")
        print_info("Please run the script again when ready")
        sys.exit(0)

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

def check_onnx_models():
    """Verify ONNX models exist (should be downloaded by setup script)"""
    models_dir = Path("openwakeword/openwakeword/resources/models")
    melspec_path = models_dir / "melspectrogram.onnx"
    embedding_path = models_dir / "embedding_model.onnx"
    
    # Check if both models exist
    if not melspec_path.exists() or not embedding_path.exists():
        print_error("ONNX models not found!")
        print_error("Please run SETUP_COMPLETE.ps1 first to download the models.")
        print_error(f"Expected location: {models_dir.absolute()}")
        sys.exit(1)
    
    # Models exist, show info
    mel_size = melspec_path.stat().st_size / (1024*1024)
    emb_size = embedding_path.stat().st_size / (1024*1024)
    print_info(f"ONNX models verified:")
    print_info(f"  melspectrogram.onnx ({mel_size:.2f} MB)")
    print_info(f"  embedding_model.onnx ({emb_size:.2f} MB)")

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
    
    # Step 4.5: Get training parameters BEFORE downloads
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
    print_info("  - Quick test: 10,000 steps (usually works well)")
    print_info("  - Standard: 30,000 steps (recommended)")
    print_info("  - Best quality: 50,000+ steps (training longer usually helps)")
    print_info("  Time: ~1-2 sec per 100 steps on GPU")
    training_steps = get_user_input("Training steps", default="30000", input_type=int)
    
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
            
            download_script = Path("download_audioset.py")
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
                print_warning("download_audioset.py not found - skipping AudioSet")
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
    if not generate_samples(wake_word, pronunciations, n_samples, base_dir):
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
