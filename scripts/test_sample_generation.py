#!/usr/bin/env python3
"""
Test Sample Generation and Pronunciation Verification
Generates test samples for each pronunciation and gets user feedback
"""

import sys
import platform
import subprocess
import shutil
from pathlib import Path
from typing import List

# Import console logging functions
from scripts.console_logger import (
    Colors,
    print_header,
    print_success,
    print_error,
    print_warning,
    print_info
)


def get_yes_no(prompt: str, default: bool = True) -> bool:
    """Get yes/no input from user"""
    default_str = "Y/n" if default else "y/N"
    full_prompt = f"{Colors.OKCYAN}{prompt} [{default_str}]: {Colors.ENDC}"
    
    while True:
        print(full_prompt, end='', flush=True)
        response = input().strip().lower()
        
        if not response:
            return default
        elif response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print_warning("Please answer 'y' or 'n'")


def test_sample_generation(wake_word: str, pronunciations: List[str], base_dir: Path) -> List[str]:
    """Generate test samples for each pronunciation and get user feedback
    
    Args:
        wake_word: The wake word being trained
        pronunciations: List of pronunciation variations to test
        base_dir: Base directory for the project
    
    Returns:
        List of approved pronunciations that the user wants to keep
    """
    print_header("Testing Sample Generation")
    
    print_info(f"Generating test samples for {len(pronunciations)} pronunciation(s)")
    print_info("This will use the default TTS model to create one sample for each")
    
    # Create temporary test directory
    test_dir = base_dir / "test_sample"
    test_dir.mkdir(exist_ok=True)
    
    approved_pronunciations = []
    test_files = []
    
    try:
        # Set TTS cache to use workspace tts folder
        import os
        os.environ['TTS_HOME'] = str(base_dir / 'tts')
        
        # Import TTS
        from TTS.api import TTS
        
        # Initialize TTS (using fast model for quick test)
        print_info("\nLoading TTS model...")
        tts = TTS("tts_models/en/ljspeech/fast_pitch")
        
        # Generate samples for each pronunciation
        for i, pronunciation in enumerate(pronunciations, 1):
            print_info(f"\n[{i}/{len(pronunciations)}] Generating sample for: '{pronunciation}'")
            
            safe_name = pronunciation.replace(' ', '_').replace('/', '_')
            test_file = test_dir / f"{safe_name}_test.wav"
            
            try:
                tts.tts_to_file(text=pronunciation, file_path=str(test_file))
                test_files.append((pronunciation, test_file))
                print_success(f"Generated: {test_file}")
            except Exception as e:
                print_error(f"Failed to generate sample: {e}")
                continue
        
        # Play and get user feedback for each sample
        print_header("Review Generated Samples")
        print_info("Please listen to each sample and decide if the pronunciation sounds good")
        
        for pronunciation, test_file in test_files:
            print(f"\n{Colors.BOLD}Pronunciation: {pronunciation}{Colors.ENDC}")
            print_info(f"File: {test_file}")
            
            # Auto-play on Windows
            if platform.system() == 'Windows':
                print_info("Playing sample...")
                try:
                    subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "{test_file}").PlaySync()'], 
                                 timeout=10)
                except:
                    print_warning("Auto-play failed, please play manually")
            
            # Get user decision
            keep = get_yes_no(f"Keep this pronunciation '{pronunciation}' for training?", default=True)
            
            if keep:
                approved_pronunciations.append(pronunciation)
                print_success(f"✓ Kept: {pronunciation}")
            else:
                print_warning(f"✗ Skipped: {pronunciation}")
        
        # Summary
        print_header("Pronunciation Selection Summary")
        if approved_pronunciations:
            print_success(f"Selected {len(approved_pronunciations)} pronunciation(s) for training:")
            for i, p in enumerate(approved_pronunciations, 1):
                print_info(f"  {i}. {p}")
        else:
            print_warning("No pronunciations selected!")
            
        return approved_pronunciations
            
    except Exception as e:
        print_error(f"Sample generation failed: {e}")
        print_info("You can still continue, but verify samples manually later")
        if get_yes_no("Continue anyway?", default=True):
            return pronunciations  # Return all pronunciations if user wants to continue
        else:
            return []
    finally:
        # Cleanup test directory
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)


def main():
    """Standalone test sample generation"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate test samples for pronunciation verification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test pronunciations for a wake word
  python scripts/test_sample_generation.py --wake-word "homie" --pronunciations homie "ho mee" "houw mee"
  
  # Use custom base directory
  python scripts/test_sample_generation.py --wake-word "alexa" --pronunciations alexa "a lex a" --base-dir /path/to/project
        """
    )
    
    parser.add_argument('--wake-word', type=str, required=True,
                       help='Wake word to train')
    parser.add_argument('--pronunciations', type=str, nargs='+', required=True,
                       help='List of pronunciation variations to test')
    parser.add_argument('--base-dir', type=Path, default=Path.cwd(),
                       help='Base directory for the project (default: current directory)')
    
    args = parser.parse_args()
    
    # Run test sample generation
    approved = test_sample_generation(args.wake_word, args.pronunciations, args.base_dir)
    
    if approved:
        print_success(f"\n✓ {len(approved)} pronunciations approved for training")
        sys.exit(0)
    else:
        print_error("\n✗ No pronunciations approved")
        sys.exit(1)


if __name__ == '__main__':
    main()
