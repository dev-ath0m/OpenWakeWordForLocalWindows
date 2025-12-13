#!/usr/bin/env python3
"""
Download Background Datasets for Wake Word Training
Consolidates all background noise/music/RIR dataset downloads

Available datasets:
- MIT RIRs: Room impulse responses (~50MB, 271 files)
- MIT Environmental: Environmental impulse responses (~300MB, HuggingFace)
- FMA: Free Music Archive (7.2GB small or 22GB medium)
- AudioSet: Environmental sounds from YouTube (requires yt-dlp and ffmpeg)
"""

import sys
import urllib.request
import zipfile
import shutil
from pathlib import Path
from typing import Optional

# Import console logging functions
from scripts.console_logger import (
    Colors,
    print_header,
    print_success,
    print_error,
    print_warning,
    print_info
)


# ============================================================================
# MIT Room Impulse Responses
# ============================================================================

def download_mit_rirs(output_path: Path) -> bool:
    """Download MIT Room Impulse Responses (~50MB, 271 files)
    
    Args:
        output_path: Directory to save WAV files
    
    Returns:
        True if successful, False otherwise
    """
    try:
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
        
        file_count = len(list(output_path.glob("*.wav")))
        print_success(f"Downloaded {file_count} MIT RIR files")
        return True
        
    except Exception as e:
        print_error(f"Failed to download MIT RIRs: {e}")
        print_info("You can download manually from: https://mcdermottlab.mit.edu/Reverb/IR_Survey.html")
        return False


# ============================================================================
# MIT Environmental Impulse Responses (HuggingFace)
# ============================================================================

def download_mit_environmental(output_path: Path) -> bool:
    """Download MIT Environmental Impulse Responses from HuggingFace (~300MB)
    
    This dataset is used in the original OpenWakeWord Colab training notebook.
    
    Args:
        output_path: Directory to save WAV files
    
    Returns:
        True if successful, False otherwise
    """
    try:
        print_info("Loading HuggingFace datasets library...")
        import datasets
        import numpy as np
        from scipy.io import wavfile
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


# ============================================================================
# FMA (Free Music Archive)
# ============================================================================

def download_fma(output_path: Path, size: str = '1') -> bool:
    """Download FMA dataset (7.2GB for small, 22GB for medium)
    
    Args:
        output_path: Directory to extract files
        size: '1' for small (7.2GB), '2' for medium (22GB)
    
    Returns:
        True if successful, False otherwise
    """
    try:
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
        
        file_count = len(list(output_path.rglob("*.mp3")))
        print_success(f"Extracted {file_count} FMA tracks")
        return True
        
    except Exception as e:
        print_error(f"Failed to download FMA: {e}")
        print_info("You can download manually from: https://github.com/mdeff/fma")
        return False


# ============================================================================
# Standalone Script Entry Point
# ============================================================================

def main():
    """Standalone dataset download script"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Download background datasets for wake word training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download MIT RIRs
  python scripts/download_background_datasets.py --mit-rirs --output-dir ./mit_rirs
  
  # Download MIT Environmental
  python scripts/download_background_datasets.py --mit-environmental --output-dir ./MIT_environmental_impulse_responses
  
  # Download FMA small
  python scripts/download_background_datasets.py --fma-small --output-dir ./fma
  
  # Download FMA medium
  python scripts/download_background_datasets.py --fma-medium --output-dir ./fma
  
  # Download all
  python scripts/download_background_datasets.py --all --base-dir .
        """
    )
    
    parser.add_argument('--mit-rirs', action='store_true',
                       help='Download MIT Room Impulse Responses')
    parser.add_argument('--mit-environmental', action='store_true',
                       help='Download MIT Environmental Impulse Responses')
    parser.add_argument('--fma-small', action='store_true',
                       help='Download FMA small dataset (7.2GB)')
    parser.add_argument('--fma-medium', action='store_true',
                       help='Download FMA medium dataset (22GB)')
    parser.add_argument('--all', action='store_true',
                       help='Download all datasets')
    parser.add_argument('--output-dir', type=Path,
                       help='Output directory for single dataset')
    parser.add_argument('--base-dir', type=Path, default=Path.cwd(),
                       help='Base directory for all datasets (used with --all)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.mit_rirs, args.mit_environmental, args.fma_small, args.fma_medium, args.all]):
        parser.print_help()
        print_error("\nError: Must specify at least one dataset to download")
        sys.exit(1)
    
    if args.all:
        # Download all datasets to base_dir
        print_header("Downloading All Background Datasets")
        base_dir = args.base_dir
        
        success_count = 0
        total_count = 0
        
        # MIT RIRs
        total_count += 1
        mit_rirs_path = base_dir / "mit_rirs"
        mit_rirs_path.mkdir(parents=True, exist_ok=True)
        if download_mit_rirs(mit_rirs_path):
            success_count += 1
        
        # MIT Environmental
        total_count += 1
        mit_env_path = base_dir / "MIT_environmental_impulse_responses"
        mit_env_path.mkdir(parents=True, exist_ok=True)
        if download_mit_environmental(mit_env_path):
            success_count += 1
        
        # FMA Small
        total_count += 1
        fma_path = base_dir / "fma"
        fma_path.mkdir(parents=True, exist_ok=True)
        if download_fma(fma_path, size='1'):
            success_count += 1
        
        print_header("Download Summary")
        print_info(f"Successfully downloaded {success_count}/{total_count} datasets")
        sys.exit(0 if success_count == total_count else 1)
    
    else:
        # Download individual dataset(s)
        if not args.output_dir:
            print_error("Error: --output-dir required when downloading individual datasets")
            sys.exit(1)
        
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        success = True
        
        if args.mit_rirs:
            success = download_mit_rirs(args.output_dir) and success
        
        if args.mit_environmental:
            success = download_mit_environmental(args.output_dir) and success
        
        if args.fma_small:
            success = download_fma(args.output_dir, size='1') and success
        
        if args.fma_medium:
            success = download_fma(args.output_dir, size='2') and success
        
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
