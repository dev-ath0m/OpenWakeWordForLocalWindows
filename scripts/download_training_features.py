#!/usr/bin/env python3
"""
Download Training Features from HuggingFace
Downloads ACAV100M and validation features required for wake word training
"""

import sys
import urllib.request
from pathlib import Path

# Import console logging functions
from scripts.console_logger import (
    Colors,
    print_header,
    print_success,
    print_error,
    print_info
)


def download_training_features(base_dir: Path) -> bool:
    """Download ACAV100M and validation features from Hugging Face
    
    Downloads two essential feature files for OpenWakeWord training:
    1. ACAV100M features (~4.7GB) - Used for adversarial negative sampling
    2. Validation features (~56MB) - Used for false positive evaluation
    
    Args:
        base_dir: Base directory where features will be saved
    
    Returns:
        True if all features downloaded successfully, False otherwise
    """
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


def main():
    """Standalone training features download script"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Download OpenWakeWord training features from HuggingFace',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download to current directory
  python scripts/download_training_features.py
  
  # Download to specific directory
  python scripts/download_training_features.py --output-dir ./features
  
Features downloaded:
  - ACAV100M features (~4.7GB): Used for adversarial negative sampling
  - Validation features (~56MB): Used for false positive evaluation
        """
    )
    
    parser.add_argument('--output-dir', type=Path, default=Path.cwd(),
                       help='Directory to save training features (default: current directory)')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download features
    success = download_training_features(args.output_dir)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
