#!/usr/bin/env python3
"""
Augmentation Module
Augments wake word samples with background noise and room impulse responses
"""
import sys
import subprocess
from pathlib import Path


def run_subprocess_with_logging(
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
        print(f"Running {description}...")
        print(f"Output will be logged to: {log_file_path}")
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
            print(f"{description} completed successfully")
            return True
        else:
            print(f"ERROR: {description} failed with exit code {process.returncode}")
            print(f"Check log file for details: {log_file_path}")
            return False
            
    except Exception as e:
        print(f"ERROR: {description} failed: {e}")
        print(f"Check log file for details: {log_file_path}")
        
        # Log the exception to file
        try:
            with open(log_file_path, 'a') as f:
                f.write(f"\n\n=== Exception ===\n")
                f.write(f"Error: {str(e)}\n")
                f.write(traceback.format_exc())
        except:
            pass
        
        return False


def patch_openwakeword_train_py(base_dir: Path) -> bool:
    """Apply runtime patches to OpenWakeWord's train.py for compatibility fixes.
    
    Wrapper function that delegates to scripts.patch_openwakeword_for_windows module.
    
    Patches:
    1. Fix warmup_steps division by zero for low step counts
    2. Fix TFLite conversion for TensorFlow 2.16+ (use onnx2tf instead of onnx-tf)
    3. Fix Windows file locking in trim_mmap function
    4. Filter out directories from RIR paths (only include .wav files)
    5. Filter out directories from background paths (only include audio files)
    
    Returns:
        True if patching succeeded, False otherwise
    """
    from scripts.patch_openwakeword_for_windows import apply_patches
    
    # Delegate to the patch module with verbose=False
    return apply_patches(base_dir, verbose=False)


def augment_samples(config_file: Path, base_dir: Path) -> bool:
    """Augment samples with background noise and room impulse responses"""
    print("=" * 70)
    print("Augmenting Samples")
    print("=" * 70)
    
    # Apply compatibility patches to OpenWakeWord
    if not patch_openwakeword_train_py(base_dir):
        print("ERROR: Failed to apply OpenWakeWord patches")
        return False
    
    print("Applying audio augmentation (noise, music, reverb)")
    print("This may take several minutes...")
    
    openwakeword_dir = base_dir / "openwakeword"
    train_script = openwakeword_dir / "openwakeword" / "train.py"
    
    if not train_script.exists():
        print(f"ERROR: Training script not found: {train_script}")
        return False
    
    aug_log_path = base_dir / "augmentation_output.log"
    
    return run_subprocess_with_logging(
        command=[sys.executable, str(train_script),
                '--training_config', str(config_file),
                '--augment_clips'],
        log_file_path=aug_log_path,
        cwd=openwakeword_dir,
        description="Augmentation"
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Augment wake word samples")
    parser.add_argument("--config", type=Path, required=True, help="Training configuration YAML file")
    parser.add_argument("--base-dir", type=Path, default=Path.cwd(), help="Base directory")
    
    args = parser.parse_args()
    
    success = augment_samples(
        config_file=args.config,
        base_dir=args.base_dir
    )
    
    sys.exit(0 if success else 1)
