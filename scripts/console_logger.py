#!/usr/bin/env python3
"""
Console Logger for OpenWakeWord Generator

Provides colored console output functions for different message types.
All functions automatically flush stdout to ensure messages appear immediately.
"""

import sys
from pathlib import Path
from typing import Optional


# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(70)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")
    sys.stdout.flush()  # Ensure header is displayed before any input prompts


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}[OK] {text}{Colors.ENDC}")
    sys.stdout.flush()


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.FAIL}[ERROR] {text}{Colors.ENDC}")
    sys.stdout.flush()


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}[!] {text}{Colors.ENDC}")
    sys.stdout.flush()


def print_info(text: str):
    """Print info message"""
    print(f"{Colors.OKCYAN}[INFO] {text}{Colors.ENDC}")
    sys.stdout.flush()


def print_summary(wake_word: str, model_dir: Path, onnx_file: Optional[Path], 
                  tflite_file: Optional[Path], n_samples: int, training_steps: int):
    """Print training summary
    
    Args:
        wake_word: Name of the wake word
        model_dir: Directory containing the trained models
        onnx_file: Path to ONNX model file (or None if not found)
        tflite_file: Path to TFLite model file (or None if not created)
        n_samples: Number of training samples used
        training_steps: Number of training steps performed
    """
    print_header("Training Complete!")
    
    print(f"{Colors.OKGREEN}{Colors.BOLD}Wake Word:{Colors.ENDC} {wake_word}")
    print(f"{Colors.OKGREEN}{Colors.BOLD}Samples:{Colors.ENDC} {n_samples}")
    print(f"{Colors.OKGREEN}{Colors.BOLD}Training Steps:{Colors.ENDC} {training_steps}")
    print(f"{Colors.OKGREEN}{Colors.BOLD}Output Directory:{Colors.ENDC} {model_dir}")
    
    print(f"\n{Colors.BOLD}Generated Models:{Colors.ENDC}")
    if onnx_file:
        size_kb = onnx_file.stat().st_size / 1024
        print(f"  {Colors.OKGREEN}[OK]{Colors.ENDC} ONNX: {onnx_file} ({size_kb:.2f} KB)")
    else:
        print(f"  {Colors.FAIL}[X]{Colors.ENDC} ONNX: Not found")
    
    if tflite_file:
        size_kb = tflite_file.stat().st_size / 1024
        print(f"  {Colors.OKGREEN}[OK]{Colors.ENDC} TFLite: {tflite_file} ({size_kb:.2f} KB)")
    else:
        print(f"  {Colors.WARNING}[-]{Colors.ENDC} TFLite: Not created (ONNX is sufficient)")
    
    print(f"\n{Colors.BOLD}Next Steps:{Colors.ENDC}")
    print(f"  1. Test the model with OpenWakeWord")
    print(f"  2. Copy to Home Assistant: {onnx_file if onnx_file else 'model.onnx'}")
    print(f"  3. Configure Wyoming OpenWakeWord service")
    print(f"  4. Test wake word detection in Home Assistant")
    
    if onnx_file:
        print(f"\n{Colors.OKCYAN}Copy command for Home Assistant:{Colors.ENDC}")
        print(f"  Copy-Item \"{onnx_file}\" \"\\\\homeassistant\\config\\custom_wakewords\\{wake_word}.onnx\"")
