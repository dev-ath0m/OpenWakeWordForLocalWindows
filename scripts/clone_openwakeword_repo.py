#!/usr/bin/env python3
"""
Clone and Setup OpenWakeWord Repository
Handles cloning, installation, patching, and ONNX model verification
"""

import sys
import subprocess
from pathlib import Path

# Import console logging functions
from scripts.console_logger import (
    print_header,
    print_success,
    print_error,
    print_warning,
    print_info
)


def check_onnx_models() -> bool:
    """Verify ONNX models exist (should be downloaded by setup script)
    
    Returns:
        True if models exist, exits with error if not found
    """
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
    
    return True


def clone_openwakeword(base_dir: Path) -> bool:
    """Clone OpenWakeWord repository if not present and install it
    
    Steps:
    1. Clone repository from GitHub (if not already present)
    2. Install as editable package (pip install -e)
    3. Apply Windows compatibility patches
    
    Args:
        base_dir: Base directory where openwakeword folder will be created
    
    Returns:
        True if successful, False otherwise
    """
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
    
    # Apply all OpenWakeWord patches for Windows compatibility
    from scripts.patch_openwakeword_for_windows import apply_all_patches
    if not apply_all_patches(openwakeword_dir, verbose=False):
        print_warning("Some patches failed - training may encounter issues")
    
    return True


def main():
    """Standalone OpenWakeWord setup script"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Clone and setup OpenWakeWord repository',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clone to current directory
  python scripts/clone_openwakeword_repo.py
  
  # Clone to specific directory
  python scripts/clone_openwakeword_repo.py --base-dir ./my_project
  
  # Only check ONNX models (no cloning)
  python scripts/clone_openwakeword_repo.py --check-only
        """
    )
    
    parser.add_argument('--base-dir', type=Path, default=Path.cwd(),
                       help='Base directory for OpenWakeWord repository (default: current directory)')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check if ONNX models exist, do not clone')
    
    args = parser.parse_args()
    
    if args.check_only:
        # Only verify ONNX models
        success = check_onnx_models()
    else:
        # Clone and setup repository
        success = clone_openwakeword(args.base_dir)
        
        if success:
            # Also check ONNX models after cloning
            try:
                check_onnx_models()
            except SystemExit:
                print_warning("Repository cloned but ONNX models not found")
                print_info("Run SETUP_COMPLETE.ps1 to download the models")
                success = False
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
