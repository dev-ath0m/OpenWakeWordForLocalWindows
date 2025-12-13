#!/usr/bin/env python3
"""
Convert all audio files (WAV/MP3) to 16kHz sample rate
Used during setup to ensure all background noise and RIR files are compatible
"""

import sys
from pathlib import Path
from scripts.audio_utils import scan_and_convert_audio_files


def convert_directory(directory, file_extensions=['.wav', '.mp3'], target_sr=16000, max_workers=4, remove_corrupted=True):
    """Convert all audio files in a directory to target sample rate
    
    Args:
        directory: Directory to scan
        file_extensions: File extensions to process
        target_sr: Target sample rate
        max_workers: Parallel workers
        remove_corrupted: Remove corrupted files if True
    
    Returns:
        Tuple of (total_files, converted_files, corrupted_files)
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"Directory not found: {directory}")
        return 0, 0, 0
    
    # Find all audio files
    audio_files = []
    for ext in file_extensions:
        audio_files.extend(dir_path.glob(f"**/*{ext}"))
    
    if not audio_files:
        print(f"No audio files found in {directory}")
        return 0, 0, 0
    
    print(f"\nProcessing {len(audio_files)} files in {directory}...")
    
    # Use shared utility for scanning and conversion
    results = scan_and_convert_audio_files(
        audio_files=audio_files,
        target_sr=target_sr,
        max_workers=max_workers,
        remove_corrupted=remove_corrupted,
        show_progress=True
    )
    
    return results['total'], results['converted'], results['corrupted']

def main():
    """Main conversion function"""
    print("="*70)
    print("Converting Audio Files to 16kHz".center(70))
    print("="*70)
    print("\nCorrupted files will be automatically removed\n")
    
    # Directories to process
    directories = [
        "mit_rirs",
        "MIT_environmental_impulse_responses",
        "fma",
        "audioset_16k"
    ]
    
    total_files = 0
    total_converted = 0
    total_errors = 0
    
    for directory in directories:
        files, converted, errors = convert_directory(directory, file_extensions=['.wav', '.mp3'], remove_corrupted=True)
        total_files += files
        total_converted += converted
        total_errors += errors
    
    print("\n" + "="*70)
    print(f"Conversion Complete!")
    print(f"  Total files found: {total_files}")
    print(f"  Files converted: {total_converted}")
    print(f"  Files already correct: {total_files - total_converted - total_errors}")
    print(f"  Corrupted files removed: {total_errors}")
    print("="*70)
    
    return 0 if total_files > 0 else 1

if __name__ == "__main__":
    sys.exit(main())
