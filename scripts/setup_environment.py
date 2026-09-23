#!/usr/bin/env python3
"""
Environment Setup and Validation
Checks Python version, GPU, virtual environment, and dependencies
"""

import sys
import subprocess
from pathlib import Path
from typing import Optional, Tuple

# Import console logging functions
from scripts.console_logger import (
    print_header,
    print_success,
    print_error,
    print_warning,
    print_info
)


def check_python_version() -> bool:
    """Check if Python version is 3.11.x
    
    Returns:
        True if Python 3.11.x is detected, False otherwise
    """
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
    """Check for NVIDIA GPU and CUDA availability
    
    Returns:
        Tuple of (has_gpu: bool, gpu_name: Optional[str])
    """
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
    """Check if running in virtual environment
    
    Returns:
        True if in virtual environment, False otherwise
    """
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
    """Check if required packages are installed
    
    Returns:
        Dict mapping package names to installation status (True/False)
    """
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


def install_dependencies(requirements_file: Path = None) -> bool:
    """Install missing dependencies from requirements.txt
    
    Args:
        requirements_file: Path to requirements.txt (auto-detected if None)
    
    Returns:
        True if installation succeeded, False otherwise
    """
    print_header("Installing Dependencies")
    
    if requirements_file is None:
        # Auto-detect requirements.txt in current directory or parent
        requirements_file = Path.cwd() / "requirements.txt"
        if not requirements_file.exists():
            requirements_file = Path(__file__).parent.parent / "requirements.txt"
    
    if not requirements_file.exists():
        print_error(f"requirements.txt not found: {requirements_file}")
        return False
    
    print_info(f"Installing packages from: {requirements_file}")
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


def main():
    """Standalone environment setup script"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Check and setup Python environment for wake word training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check all environment requirements
  python scripts/setup_environment.py
  
  # Check and install missing dependencies
  python scripts/setup_environment.py --install
  
  # Use custom requirements file
  python scripts/setup_environment.py --install --requirements ./my_requirements.txt
        """
    )
    
    parser.add_argument('--install', action='store_true',
                       help='Install missing dependencies automatically')
    parser.add_argument('--requirements', type=Path,
                       help='Path to requirements.txt file')
    parser.add_argument('--skip-gpu-check', action='store_true',
                       help='Skip GPU availability check')
    
    args = parser.parse_args()
    
    # Run all checks
    all_ok = True
    
    # 1. Python version
    if not check_python_version():
        all_ok = False
        print_error("Python version check failed")
        sys.exit(1)
    
    # 2. Virtual environment
    if not check_virtual_env():
        all_ok = False
        print_error("Virtual environment check failed")
        sys.exit(1)
    
    # 3. GPU (optional)
    if not args.skip_gpu_check:
        has_gpu, gpu_name = check_gpu()
        if has_gpu:
            print_info(f"GPU acceleration available: {gpu_name}")
        else:
            print_warning("No GPU detected - training will use CPU (slower)")
    
    # 4. Dependencies
    deps_status = check_dependencies()
    all_deps_ok = all(deps_status.values())
    
    if not all_deps_ok:
        if args.install:
            # Install missing dependencies
            if install_dependencies(args.requirements):
                print_success("\nEnvironment setup complete!")
                sys.exit(0)
            else:
                print_error("\nDependency installation failed")
                sys.exit(1)
        else:
            print_warning("\nSome dependencies are missing")
            print_info("Run with --install flag to install them automatically")
            sys.exit(1)
    else:
        print_success("\nAll environment checks passed!")
        sys.exit(0)


if __name__ == '__main__':
    main()
