#!/usr/bin/env python3
"""
Convert all audio files (WAV/MP3) to 16kHz sample rate
Used during setup to ensure all background noise and RIR files are compatible
"""

import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import torchaudio
import torchaudio.transforms as T

def resample_audio_file(file_path, target_sr=16000):
    """Resample a single audio file to target sample rate"""
    try:
        # Load audio
        waveform, sr = torchaudio.load(str(file_path))
        
        # Skip if already correct sample rate
        if sr == target_sr:
            return (file_path, sr, 'skipped')
        
        # Resample
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform_resampled = resampler(waveform)
        
        # Save back to same file
        torchaudio.save(str(file_path), waveform_resampled, target_sr)
        
        return (file_path, sr, 'converted')
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return (file_path, None, f'error: {str(e)[:100]}')

def convert_directory(directory, file_extensions=['.wav', '.mp3'], target_sr=16000, max_workers=4, remove_corrupted=True):
    """Convert all audio files in a directory to target sample rate
    
    Args:
        directory: Directory to scan
        file_extensions: File extensions to process
        target_sr: Target sample rate
        max_workers: Parallel workers
        remove_corrupted: Remove corrupted files if True
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
    
    # First pass: check which files need conversion and identify corrupted
    files_to_convert = []
    corrupted_files = []
    
    for audio_file in tqdm(audio_files, desc="Scanning", unit="files"):
        try:
            _, sr = torchaudio.load(str(audio_file))
            if sr != target_sr:
                files_to_convert.append(audio_file)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            corrupted_files.append((audio_file, str(e)[:100]))
    
    # Handle corrupted files
    if corrupted_files:
        print(f"  Found {len(corrupted_files)} corrupted files")
        if remove_corrupted:
            print(f"  Removing corrupted files...")
            removed = 0
            for file_path, error in corrupted_files:
                try:
                    file_path.unlink()
                    removed += 1
                except Exception:
                    pass
            print(f"  Removed {removed} corrupted files")
    
    if not files_to_convert:
        valid_count = len(audio_files) - len(corrupted_files)
        print(f"  All {valid_count} valid files already at {target_sr} Hz")
        return len(audio_files), 0, len(corrupted_files)
    
    print(f"  Converting {len(files_to_convert)} files to {target_sr} Hz...")
    
    # Convert files in parallel
    converted = 0
    conversion_errors = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(resample_audio_file, f, target_sr): f for f in files_to_convert}
        
        with tqdm(total=len(files_to_convert), desc="Converting", unit="files") as pbar:
            for future in as_completed(futures):
                file_path, original_sr, status = future.result()
                
                if status == 'converted':
                    converted += 1
                elif status.startswith('error'):
                    conversion_errors.append((file_path, status))
                
                pbar.update(1)
    
    # Handle conversion errors
    total_errors = len(corrupted_files)
    if conversion_errors:
        print(f"  {len(conversion_errors)} files failed conversion")
        if remove_corrupted:
            print(f"  Removing failed files...")
            removed = 0
            for file_path, error in conversion_errors:
                try:
                    file_path.unlink()
                    removed += 1
                except Exception:
                    pass
            print(f"  Removed {removed} failed files")
        total_errors += len(conversion_errors)
    
    print(f"  Converted: {converted}, Total errors: {total_errors}")
    return len(audio_files), converted, total_errors

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
