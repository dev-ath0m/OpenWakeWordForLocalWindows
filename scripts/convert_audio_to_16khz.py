#!/usr/bin/env python3
"""
Convert all audio files (WAV/MP3) to 16kHz sample rate
Used during setup to ensure all background noise and RIR files are compatible

This module contains all audio sample rate checking and conversion functionality:
- scan_and_convert_audio_files: Core parallel conversion engine
- check_audio_file: Validate single file sample rate
- resample_audio_file: Convert single file to target sample rate
- check_and_fix_audio_sample_rates: Batch process from config dict
- convert_directory: Process entire directory structures
"""

import os
import sys
import contextlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import torchaudio
import torchaudio.transforms as T


# ============================================================================
# Low-level Audio Processing Functions
# ============================================================================

@contextlib.contextmanager
def suppress_stderr():
    """Temporarily suppress stderr to hide low-level library errors (mpg123, etc.)
    
    Uses OS-level file descriptor redirection to catch C library output on Windows
    """
    # Flush Python's stderr buffer first
    sys.stderr.flush()
    
    # Save original stderr file descriptor
    stderr_fd = sys.stderr.fileno()
    old_stderr_fd = os.dup(stderr_fd)
    
    # Open null device
    null_file = open(os.devnull, 'w')
    
    try:
        # Redirect stderr file descriptor to null
        os.dup2(null_file.fileno(), stderr_fd)
        yield
    finally:
        # Restore original stderr
        sys.stderr.flush()
        os.dup2(old_stderr_fd, stderr_fd)
        os.close(old_stderr_fd)
        null_file.close()


def check_audio_file(audio_file):
    """Check a single audio file's sample rate
    
    Args:
        audio_file: Path to audio file
    
    Returns:
        Tuple of (file_path, sample_rate, error_message)
        - If successful: (path, sample_rate, None)
        - If error: (path, None, error_string)
    """
    try:
        with suppress_stderr():
            _, sr = torchaudio.load(str(audio_file))
        return (audio_file, sr, None)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return (audio_file, None, str(e)[:100])


def resample_audio_file(args):
    """Resample a single audio file to target sample rate
    
    Args:
        args: Tuple of (file_path, target_sr)
    
    Returns:
        Tuple of (file_path, original_sr, status)
        - If skipped: (path, sr, 'skipped')
        - If converted: (path, sr, 'converted')
        - If error: (path, None, 'error: message')
    """
    file_path, target_sr = args
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


# ============================================================================
# Core Parallel Conversion Engine
# ============================================================================

def scan_and_convert_audio_files(
    audio_files,
    target_sr=16000,
    max_workers=None,
    remove_corrupted=True,
    show_progress=True
):
    """Scan and convert audio files to target sample rate with parallel processing
    
    Args:
        audio_files: List of Path objects to process
        target_sr: Target sample rate (default: 16000)
        max_workers: Max parallel workers (default: min(cpu_count, 8))
        remove_corrupted: Remove corrupted files if True (default: True)
        show_progress: Show progress bars if True (default: True)
    
    Returns:
        Dict with:
            - total: Total files processed
            - converted: Files successfully converted
            - corrupted: Files that were corrupted
            - already_correct: Files already at target sample rate
    """
    if not audio_files:
        return {'total': 0, 'converted': 0, 'corrupted': 0, 'already_correct': 0}
    
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 8)
    
    # Phase 1: Parallel scanning
    files_to_convert = []
    corrupted_files = []
    scanned = 0
    pbar = None
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(check_audio_file, f): f for f in audio_files}
            
            if show_progress:
                # Use sys.stdout explicitly and disable mininterval for responsive updates
                pbar = tqdm(total=len(audio_files), desc="Scanning", unit="files", 
                           file=sys.stdout, mininterval=0.5, dynamic_ncols=True)
            
            try:
                for future in as_completed(futures):
                    audio_file, sr, error = future.result()
                    
                    if error:
                        corrupted_files.append((audio_file, error))
                    elif sr != target_sr:
                        files_to_convert.append((audio_file, sr))
                    
                    scanned += 1
                    
                    if show_progress:
                        # Update failed percentage every 100 files
                        if scanned % 100 == 0:
                            failed_pct = (len(corrupted_files) / scanned * 100) if scanned > 0 else 0
                            pbar.set_postfix({'Failed': f'{len(corrupted_files)} ({failed_pct:.1f}%)'})
                        pbar.update(1)
            except KeyboardInterrupt:
                if show_progress and pbar:
                    pbar.close()
                print("\n[INTERRUPTED] Cancelling audio file scanning...")
                # Cancel remaining futures and shutdown executor quickly
                for future in futures:
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise
    except KeyboardInterrupt:
        if show_progress and pbar and not pbar.disable:
            pbar.close()
        raise
    finally:
        if show_progress and pbar and not pbar.disable:
            pbar.close()
    
    # Report scan results
    if show_progress:
        failed_pct = (len(corrupted_files) / len(audio_files) * 100) if audio_files else 0
        print(f"Scan complete: {len(corrupted_files)} corrupted files ({failed_pct:.1f}%)")
    
    # Handle corrupted files
    removed_corrupted = 0
    if corrupted_files and remove_corrupted:
        if show_progress:
            print(f"Removing {len(corrupted_files)} corrupted files...")
        for file_path, error in corrupted_files:
            try:
                file_path.unlink()
                removed_corrupted += 1
            except Exception:
                pass
        if show_progress:
            print(f"Removed {removed_corrupted} corrupted files")
    
    # Phase 2: Parallel conversion
    converted = 0
    conversion_errors = []
    pbar2 = None
    
    if files_to_convert:
        if show_progress:
            print(f"Converting {len(files_to_convert)} files to {target_sr} Hz...")
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(resample_audio_file, (f, target_sr)): f for f, _ in files_to_convert}
                
                if show_progress:
                    pbar2 = tqdm(total=len(files_to_convert), desc="Converting", unit="files",
                               file=sys.stdout, mininterval=0.5, dynamic_ncols=True)
                
                try:
                    for future in as_completed(futures):
                        file_path, original_sr, status = future.result()
                        
                        if status == 'converted':
                            converted += 1
                        elif status.startswith('error'):
                            conversion_errors.append((file_path, status))
                        
                        if show_progress:
                            pbar2.update(1)
                except KeyboardInterrupt:
                    if show_progress and pbar2:
                        pbar2.close()
                    print("\n[INTERRUPTED] Cancelling audio file conversion...")
                    # Cancel remaining futures and shutdown quickly
                    for future in futures:
                        future.cancel()
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
        except KeyboardInterrupt:
            if show_progress and pbar2 and not pbar2.disable:
                pbar2.close()
            raise
        finally:
            if show_progress and pbar2 and not pbar2.disable:
                pbar2.close()
    
    # Handle conversion errors
    removed_failed = 0
    if conversion_errors and remove_corrupted:
        if show_progress:
            print(f"Removing {len(conversion_errors)} files that failed conversion...")
        for file_path, error in conversion_errors:
            try:
                file_path.unlink()
                removed_failed += 1
            except Exception:
                pass
        if show_progress:
            print(f"Removed {removed_failed} failed files")
    
    # Calculate statistics
    total_corrupted = len(corrupted_files) + len(conversion_errors)
    already_correct = len(audio_files) - len(files_to_convert) - len(corrupted_files)
    
    return {
        'total': len(audio_files),
        'converted': converted,
        'corrupted': total_corrupted,
        'already_correct': already_correct
    }


# ============================================================================
# High-level Batch Processing Functions
# ============================================================================

def check_and_fix_audio_sample_rates(config: dict, remove_corrupted: bool = True) -> bool:
    """Check and convert all background/RIR files to 16kHz using parallel processing
    
    This function processes audio files from a training configuration dict,
    checking and converting all background noise and room impulse response files.
    
    Args:
        config: Configuration dict with 'background_paths' and 'rir_paths'
        remove_corrupted: If True, delete files that cannot be loaded or converted
    
    Returns:
        True if all files were processed successfully, False if errors occurred
    """
    from scripts.console_logger import print_header, print_success, print_info, print_warning
    
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



# ============================================================================
# Directory Processing Functions
# ============================================================================

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


# ============================================================================
# Standalone Script Entry Point
# ============================================================================

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
