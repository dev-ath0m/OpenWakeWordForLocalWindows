#!/usr/bin/env python3
"""
OpenWakeWord Compatibility Patches for Windows and TensorFlow 2.16+

Applies runtime patches to OpenWakeWord library files for:
1. Fix warmup_steps division by zero for low step counts
2. Fix TFLite conversion for TensorFlow 2.16+ (use onnx2tf instead of onnx-tf)
3. Fix Windows file locking in trim_mmap function
4. Filter out directories from RIR paths (only include .wav files)
5. Filter out directories from background paths (only include audio files)
"""

from pathlib import Path
from typing import Optional


def apply_patches(base_dir: Path, verbose: bool = True) -> bool:
    """Apply runtime patches to OpenWakeWord's train.py and data.py for compatibility fixes.
    
    Args:
        base_dir: Base directory containing the openwakeword folder
        verbose: If True, print status messages
    
    Returns:
        True if patching succeeded, False otherwise
    """
    train_py_path = base_dir / "openwakeword" / "openwakeword" / "train.py"
    data_py_path = base_dir / "openwakeword" / "openwakeword" / "data.py"
    
    if not train_py_path.exists():
        if verbose:
            print(f"[ERROR] Cannot patch: {train_py_path} not found")
        return False
    
    if not data_py_path.exists():
        if verbose:
            print(f"[ERROR] Cannot patch: {data_py_path} not found")
        return False
    
    try:
        # Read train.py
        with open(train_py_path, 'r', encoding='utf-8') as f:
            train_content = f.read()
        
        # Read data.py
        with open(data_py_path, 'r', encoding='utf-8') as f:
            data_content = f.read()
        
        # Check if patches already applied (look for minimal API version without fallback)
        minimal_api_signature = 'onnx2tf.convert(\n                input_onnx_file_path=onnx_model_path,\n                output_folder_path=tmp_dir\n            )'
        no_fallback_signature = 'raise RuntimeError(f"TFLite conversion failed: {e}") from e'
        warmup_fix_signature = 'if warmup_steps > 0:'
        rir_filter_signature = 'i.is_file() and i.path.endswith'
        
        if (minimal_api_signature in train_content and no_fallback_signature in train_content and 
            warmup_fix_signature in train_content and rir_filter_signature in train_content and 
            'explicit close + gc' in data_content):
            if verbose:
                print("[INFO] OpenWakeWord patches already applied")
            return True  # Already patched with correct version
        
        # Check if old patch with deprecated API or onnx-tf fallback is present
        needs_update = (
            ('output_tfjs=False' in train_content and 'onnx2tf.convert(' in train_content) or
            ('from onnx_tf.backend import prepare' in train_content)
        )
        
        if needs_update:
            if verbose:
                print("[WARNING] Old onnx2tf patch detected - updating to current API...")
        elif 'onnx2tf' in train_content:
            if verbose:
                print("[INFO] Updating OpenWakeWord patches to remove onnx-tf fallback...")
        else:
            if verbose:
                print("[INFO] Applying compatibility patches to OpenWakeWord...")
        
        # Patch 1: Fix warmup_steps division by zero for low step counts
        old_warmup_code = '''    def lr_warmup_cosine_decay(self, global_step, warmup_steps, hold_steps, total_steps,
                               start_lr=0, target_lr=0.001):
        # Linear warm up
        warmup_lr = target_lr * (global_step / warmup_steps)'''
        
        new_warmup_code = '''    def lr_warmup_cosine_decay(self, global_step, warmup_steps, hold_steps, total_steps,
                               start_lr=0, target_lr=0.001):
        # Linear warm up (avoid division by zero for very small warmup_steps)
        if warmup_steps > 0:
            warmup_lr = target_lr * (global_step / warmup_steps)
        else:
            warmup_lr = target_lr'''
        
        if old_warmup_code in train_content:
            train_content = train_content.replace(old_warmup_code, new_warmup_code)
            if verbose:
                print("  ✓ Patched warmup_steps division by zero fix")
        else:
            if verbose:
                print("  ! Warmup code not found (may already be patched)")
        
        # Patch 2: Fix TFLite conversion in train.py
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
        logging.error(f"onnx2tf conversion failed: {e}")
        logging.error(f"TFLite conversion requires onnx2tf with TensorFlow 2.16+")
        logging.error(f"Please ensure onnx2tf is installed: pip install onnx2tf==1.20.0")
        import traceback
        logging.error(f"Full error:\n{traceback.format_exc()}")
        raise RuntimeError(f"TFLite conversion failed: {e}") from e

    return None'''
        
        if old_convert_func in train_content:
            train_content = train_content.replace(old_convert_func, new_convert_func)
            if verbose:
                print("  ✓ Patched TFLite conversion for TensorFlow 2.16 compatibility")
        else:
            if verbose:
                print("  ! TFLite conversion function not found (may already be patched)")
        
        # Patch 3: Filter out directories from RIR paths (only include audio files)
        old_rir_line = '    rir_paths = [i.path for j in config["rir_paths"] for i in os.scandir(j)]'
        new_rir_line = '    rir_paths = [i.path for j in config["rir_paths"] for i in os.scandir(j) if i.is_file() and i.path.endswith((\'.wav\', \'.WAV\'))]'
        
        if old_rir_line in train_content:
            train_content = train_content.replace(old_rir_line, new_rir_line)
            if verbose:
                print("  ✓ Patched RIR path collection to exclude directories")
        else:
            if verbose:
                print("  ! RIR path collection code not found (may already be patched)")
        
        # Patch 4: Filter background_paths to exclude directories
        old_bg_line = '        background_paths.extend([i.path for i in os.scandir(background_path)]*duplication_rate)'
        new_bg_line = '        background_paths.extend([i.path for i in os.scandir(background_path) if i.is_file() and i.path.endswith((\'.wav\', \'.mp3\', \'.WAV\', \'.MP3\'))]*duplication_rate)'
        
        if old_bg_line in train_content:
            train_content = train_content.replace(old_bg_line, new_bg_line)
            if verbose:
                print("  ✓ Patched background path collection to exclude directories")
        else:
            if verbose:
                print("  ! Background path collection code not found (may already be patched)")
        
        # Patch 5: Fix Windows file locking in data.py
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
            if verbose:
                print("  ✓ Patched Windows file locking in trim_mmap")
        else:
            if verbose:
                print("  ! File locking code not found (may already be patched)")
        
        # Write patched files
        with open(train_py_path, 'w', encoding='utf-8') as f:
            f.write(train_content)
        
        with open(data_py_path, 'w', encoding='utf-8') as f:
            f.write(data_content)
        
        if verbose:
            print("[SUCCESS] OpenWakeWord compatibility patches applied successfully")
        return True
        
    except Exception as e:
        if verbose:
            print(f"[ERROR] Failed to patch OpenWakeWord: {e}")
            import traceback
            traceback.print_exc()
        return False


def patch_train_script(openwakeword_dir: Path, verbose: bool = True) -> bool:
    """Patch OpenWakeWord's train.py to make Piper dependency optional
    
    Args:
        openwakeword_dir: Path to openwakeword directory
        verbose: If True, print status messages
    
    Returns:
        True if patching succeeded, False otherwise
    """
    train_script = openwakeword_dir / "openwakeword" / "train.py"
    
    if not train_script.exists():
        if verbose:
            print(f"[ERROR] train.py not found at {train_script}")
        return False
    
    if verbose:
        print("[INFO] Patching train.py to make Piper optional...")
    
    try:
        import re
        
        # Read the file
        content = train_script.read_text(encoding='utf-8')
        original_content = content
        
        # Patch 1: Make Piper import conditional
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
        pattern2 = r'(if n_current_samples <= 0\.95\*config\["n_samples"\]:\s+)(generate_samples\(\s+text=config\["target_phrase"\])'
        replacement2 = r'''\1if generate_samples is None:
                logging.error("Piper sample generator not available and positive samples not pre-generated!")
                logging.error("Please generate samples manually or provide piper_sample_generator_path in config")
                raise RuntimeError("Cannot generate clips without Piper or pre-generated samples")
            \2'''
        
        content = re.sub(pattern2, replacement2, content, count=1)
        
        # Check if any changes were made
        if content == original_content:
            if verbose:
                print("[WARNING] No changes made - train.py may already be patched or format has changed")
            return True  # Don't fail, just warn
        
        # Write the patched content back
        train_script.write_text(content, encoding='utf-8')
        if verbose:
            print("[SUCCESS] train.py patched successfully - Piper is now optional")
        return True
        
    except Exception as e:
        if verbose:
            print(f"[ERROR] Patching failed: {e}")
        return False


def patch_utils(openwakeword_dir: Path, verbose: bool = True) -> bool:
    """Patch OpenWakeWord's utils.py to fix Windows file permission errors
    
    Args:
        openwakeword_dir: Path to openwakeword directory
        verbose: If True, print status messages
    
    Returns:
        True if patching succeeded, False otherwise
    """
    utils_script = openwakeword_dir / "openwakeword" / "utils.py"
    
    if not utils_script.exists():
        if verbose:
            print(f"[ERROR] utils.py not found at {utils_script}")
        return False
    
    if verbose:
        print("[INFO] Patching utils.py for Windows compatibility...")
    
    try:
        content = utils_script.read_text(encoding='utf-8')
        original_content = content
        
        # Patch: Close memory-mapped file before trimming (Windows requires this)
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
            if verbose:
                print("[WARNING] No changes made - utils.py may already be patched or format has changed")
            return True
        
        utils_script.write_text(content, encoding='utf-8')
        if verbose:
            print("[SUCCESS] utils.py patched successfully - Windows file handling fixed")
        return True
        
    except Exception as e:
        if verbose:
            print(f"[ERROR] Patching utils.py failed: {e}")
        return False


def patch_data(openwakeword_dir: Path, verbose: bool = True) -> bool:
    """Patch OpenWakeWord's data.py to fix Windows file permission errors (legacy, handled by apply_patches)
    
    Args:
        openwakeword_dir: Path to openwakeword directory
        verbose: If True, print status messages
    
    Returns:
        True (this is now handled by apply_patches)
    """
    if verbose:
        print("[INFO] data.py patching is handled by main apply_patches function")
    return True


def patch_train_multiprocessing(openwakeword_dir: Path, verbose: bool = True) -> bool:
    """Patch OpenWakeWord's train.py to fix Windows multiprocessing issues
    
    Args:
        openwakeword_dir: Path to openwakeword directory
        verbose: If True, print status messages
    
    Returns:
        True if patching succeeded, False otherwise
    """
    train_script = openwakeword_dir / "openwakeword" / "train.py"
    
    if not train_script.exists():
        if verbose:
            print(f"[ERROR] train.py not found at {train_script}")
        return False
    
    if verbose:
        print("[INFO] Patching train.py for Windows multiprocessing compatibility...")
    
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
            if verbose:
                print("[WARNING] No changes made - train.py may already be patched or format has changed")
            return True
        
        train_script.write_text(content, encoding='utf-8')
        if verbose:
            print("[SUCCESS] train.py patched successfully - Windows multiprocessing fixed")
        return True
        
    except Exception as e:
        if verbose:
            print(f"[ERROR] Patching train.py multiprocessing failed: {e}")
        return False


def apply_all_patches(openwakeword_dir: Path, verbose: bool = True) -> bool:
    """Apply all OpenWakeWord patches for Windows compatibility
    
    Args:
        openwakeword_dir: Path to openwakeword directory
        verbose: If True, print status messages
    
    Returns:
        True if all patches succeeded, False if any failed
    """
    success = True
    
    # Apply TF 2.16+ and Windows file locking patches
    base_dir = openwakeword_dir.parent
    if not apply_patches(base_dir, verbose=verbose):
        success = False
    
    # Apply Piper optional patch
    if not patch_train_script(openwakeword_dir, verbose=verbose):
        if verbose:
            print("[WARNING] Failed to patch train.py - Piper will be required")
        # Don't set success=False, this is optional
    
    # Apply utils.py patch
    if not patch_utils(openwakeword_dir, verbose=verbose):
        if verbose:
            print("[WARNING] Failed to patch utils.py - may have file permission errors on Windows")
        # Don't set success=False, this is optional
    
    # Apply multiprocessing patch
    if not patch_train_multiprocessing(openwakeword_dir, verbose=verbose):
        if verbose:
            print("[WARNING] Failed to patch train.py multiprocessing - training may fail on Windows")
        # Don't set success=False, this is optional
    
    return success


def main():
    """Standalone script to patch OpenWakeWord files"""
    import sys
    
    # Determine base directory
    if len(sys.argv) > 1:
        base_dir = Path(sys.argv[1])
    else:
        base_dir = Path(__file__).parent.parent  # Parent of scripts folder
    
    print("="*70)
    print("OpenWakeWord Compatibility Patcher for Windows & TensorFlow 2.16+")
    print("="*70)
    print(f"\nBase directory: {base_dir}")
    print("\nApplying all patches:")
    print("  1. Warmup steps division by zero fix")
    print("  2. TFLite conversion (onnx2tf for TF 2.16+)")
    print("  3. Windows file locking fix")
    print("  4. RIR directory filtering")
    print("  5. Background directory filtering")
    print("  6. Piper optional dependency")
    print("  7. Utils.py Windows file handling")
    print("  8. Multiprocessing Windows compatibility")
    print()
    
    openwakeword_dir = base_dir / "openwakeword"
    success = apply_all_patches(openwakeword_dir, verbose=True)
    
    print("\n" + "="*70)
    if success:
        print("Patching completed successfully!")
        return 0
    else:
        print("Patching failed - see errors above")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
