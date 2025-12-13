#!/usr/bin/env python3
"""
Shared audio processing utilities for sample rate conversion
Used by both training script and standalone conversion tool
"""

import os
import sys
import contextlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import torchaudio
import torchaudio.transforms as T


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
