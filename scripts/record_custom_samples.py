#!/usr/bin/env python3
"""
Custom Voice Sample Recording Module
Records user's own voice pronouncing the wake word for training data
"""

import sys
import time
from pathlib import Path
from typing import List, Tuple
import sounddevice as sd
import soundfile as sf
import numpy as np
from pynput import keyboard

# Import console logging functions
from scripts.console_logger import (
    Colors,
    print_header,
    print_success,
    print_error,
    print_warning,
    print_info
)


class SampleRecorder:
    """Records audio samples from microphone with keyboard controls"""
    
    def __init__(self, sample_rate: int = 16000, duration: float = 2.0):
        """Initialize recorder
        
        Args:
            sample_rate: Audio sample rate in Hz (16000 for wake word training)
            duration: Recording duration in seconds
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.is_recording = False
        self.should_stop = False
        self.space_pressed = False
        self.enter_pressed = False
        self.backspace_pressed = False
        self.current_action = None
        
    def on_press(self, key):
        """Handle keyboard press events"""
        try:
            if key == keyboard.Key.space:
                self.space_pressed = True
                self.current_action = 'space'
            elif key == keyboard.Key.enter:
                self.enter_pressed = True
                self.current_action = 'enter'
            elif key == keyboard.Key.backspace:
                self.backspace_pressed = True
                self.current_action = 'backspace'
        except AttributeError:
            pass
    
    def reset_keys(self):
        """Reset all key press flags"""
        self.space_pressed = False
        self.enter_pressed = False
        self.backspace_pressed = False
        self.current_action = None
    
    def wait_for_key(self, allowed_keys: List[str]) -> str:
        """Wait for specific key press
        
        Args:
            allowed_keys: List of allowed keys ('space', 'enter', 'backspace')
            
        Returns:
            The key that was pressed
        """
        self.reset_keys()
        while True:
            if 'space' in allowed_keys and self.space_pressed:
                return 'space'
            if 'enter' in allowed_keys and self.enter_pressed:
                return 'enter'
            if 'backspace' in allowed_keys and self.backspace_pressed:
                return 'backspace'
            time.sleep(0.05)
    
    def record_sample(self) -> np.ndarray:
        """Record audio sample with SPACE to start/stop
        
        Returns:
            Audio data as numpy array
        """
        print_info("🎤 Press SPACE to start recording...")
        self.wait_for_key(['space'])
        
        print_success("🔴 Recording... Press SPACE to stop")
        
        # Start recording in a buffer
        recording_chunks = []
        chunk_duration = 0.1  # Record in 100ms chunks
        chunk_samples = int(chunk_duration * self.sample_rate)
        
        self.reset_keys()
        while not self.space_pressed:
            chunk = sd.rec(chunk_samples, samplerate=self.sample_rate, channels=1, dtype='float32')
            sd.wait()
            recording_chunks.append(chunk.flatten())
            time.sleep(0.01)  # Small delay to check for key press
        
        # Combine all chunks
        if recording_chunks:
            audio_data = np.concatenate(recording_chunks)
            duration = len(audio_data) / self.sample_rate
            print_success(f"✓ Recorded {duration:.1f} seconds")
            return audio_data
        else:
            return np.array([], dtype='float32')
    
    def preview_sample(self, audio_data: np.ndarray) -> bool:
        """Play back recorded sample and ask for user approval
        
        Args:
            audio_data: Recorded audio data
            
        Returns:
            True if user approves, False otherwise
        """
        print_info("🔊 Playing back your recording...")
        sd.play(audio_data, self.sample_rate)
        sd.wait()
        
        print_info(f"\n{Colors.OKGREEN}ENTER{Colors.ENDC} = Keep sample  |  {Colors.WARNING}BACKSPACE{Colors.ENDC} = Discard")
        key = self.wait_for_key(['enter', 'backspace'])
        
        return key == 'enter'


def record_custom_samples(wake_word: str, base_dir: Path, target_count: int = 10) -> Tuple[bool, int]:
    """Record custom voice samples from user
    
    Args:
        wake_word: The wake word to record
        base_dir: Base directory for the project
        target_count: Suggested minimum number of samples (informational only)
        
    Returns:
        Tuple of (success: bool, samples_recorded: int)
    """
    print_header("Custom Voice Sample Recording")
    
    print_info(f"Record your own voice pronouncing '{wake_word}'")
    print_info("This adds your unique voice to the training data for better accuracy\n")
    
    print_info("Recording settings:")
    print_info(f"  • Duration: 2 seconds per sample")
    print_info(f"  • Sample rate: 16kHz (required for training)")
    print_info(f"  • Format: WAV mono\n")
    
    print_info(f"{Colors.BOLD}How it works:{Colors.ENDC}")
    print_info(f"  1. Press {Colors.OKGREEN}SPACE{Colors.ENDC} to start recording")
    print_info(f"  2. Say '{wake_word}' clearly while recording")
    print_info(f"  3. Press {Colors.OKGREEN}SPACE{Colors.ENDC} again to stop recording")
    print_info(f"  4. Repeat steps 1-3 as many times as you want")
    print_info(f"  5. Press {Colors.WARNING}ENTER{Colors.ENDC} (instead of SPACE) when done recording")
    print_info(f"  6. Review each sample: {Colors.OKGREEN}ENTER{Colors.ENDC} to keep, {Colors.WARNING}BACKSPACE{Colors.ENDC} to discard\n")
    
    # Check microphone availability
    try:
        devices = sd.query_devices()
        default_input = sd.query_devices(kind='input')
        print_success(f"Using microphone: {default_input['name']}")
        print_info(f"Sample rate: {default_input['default_samplerate']} Hz")
    except Exception as e:
        print_error(f"Microphone error: {e}")
        print_error("Please check that a microphone is connected")
        return False, 0
    
    # Create output directories for train/test split
    model_name = wake_word.lower().replace(' ', '_')
    train_dir = base_dir / "trained_models" / model_name / "positive_train"
    test_dir = base_dir / "trained_models" / model_name / "positive_test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print_info(f"\nSamples will be saved directly to training directories:")
    print_info(f"  Training: {train_dir}")
    print_info(f"  Testing:  {test_dir}")
    print_info(f"  Split: 90% train / 10% test\n")
    
    # Confirm to start
    sys.stdout.write(f"{Colors.OKCYAN}Ready to start recording? [Y/n]: {Colors.ENDC}")
    sys.stdout.flush()
    response = input().strip().lower()
    if response and response not in ['y', 'yes']:
        print_info("Recording cancelled")
        return False, 0
    
    # Initialize recorder
    recorder = SampleRecorder(sample_rate=16000, duration=2.0)
    recorded_samples = []  # Store all recordings for review phase
    
    print_info(f"\n{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print_info(f"{Colors.BOLD}Recording Phase{Colors.ENDC}")
    print_info(f"{Colors.BOLD}{'='*60}{Colors.ENDC}\n")
    
    # Start keyboard listener
    listener = keyboard.Listener(on_press=recorder.on_press)
    listener.start()
    
    try:
        print_info(f"Press {Colors.OKGREEN}SPACE{Colors.ENDC} to start first recording")
        print_info(f"Press {Colors.WARNING}ENTER{Colors.ENDC} when you're done recording all samples\n")
        
        sample_number = 1
        
        while True:
            # Wait for SPACE (record) or ENTER (finish)
            key = recorder.wait_for_key(['space', 'enter'])
            
            if key == 'enter':
                break
            
            # Record sample
            print_info(f"\n{Colors.BOLD}[Recording #{sample_number}]{Colors.ENDC}")
            
            try:
                audio_data = recorder.record_sample()
                
                if len(audio_data) > 0:
                    recorded_samples.append(audio_data)
                    print_info(f"Sample #{sample_number} captured. Press {Colors.OKGREEN}SPACE{Colors.ENDC} for next or {Colors.WARNING}ENTER{Colors.ENDC} to finish\n")
                    sample_number += 1
                else:
                    print_warning("Recording too short, discarded\n")
                    
            except Exception as e:
                print_error(f"Recording failed: {e}")
                print_warning("Try again...\n")
                continue
        
        if not recorded_samples:
            print_warning("\nNo samples recorded")
            return False, 0
        
        print_info(f"\n{Colors.BOLD}{'='*60}{Colors.ENDC}")
        print_info(f"{Colors.BOLD}Review Phase - {len(recorded_samples)} samples to review{Colors.ENDC}")
        print_info(f"{Colors.BOLD}{'='*60}{Colors.ENDC}\n")
        
        # Review phase - play each sample and get approval
        approved_samples = []
        
        for idx, audio_data in enumerate(recorded_samples, 1):
            print_info(f"\n{Colors.BOLD}[Sample {idx}/{len(recorded_samples)}]{Colors.ENDC}")
            
            try:
                approved = recorder.preview_sample(audio_data)
                
                if approved:
                    approved_samples.append(audio_data)
                    print_success(f"✓ Sample {idx} kept ({len(approved_samples)} total)")
                else:
                    print_warning(f"✗ Sample {idx} discarded")
                    
            except Exception as e:
                print_error(f"Playback failed: {e}")
                print_warning("Sample discarded")
                continue
        
        if not approved_samples:
            print_warning("\nNo samples were approved")
            return False, 0
        
        print_info(f"\n{Colors.BOLD}{'='*60}{Colors.ENDC}")
        print_info(f"{Colors.BOLD}Saving Phase - {len(approved_samples)} samples to save{Colors.ENDC}")
        print_info(f"{Colors.BOLD}{'='*60}{Colors.ENDC}\n")
        
        # Save approved samples with train/test split
        samples_saved = 0
        train_count = 0
        test_count = 0
        
        for idx, audio_data in enumerate(approved_samples):
            # Every 10th approved sample goes to test set
            is_test = (idx % 10 == 9)
            
            if is_test:
                sample_file = test_dir / f"custom_voice_test_{test_count:03d}.wav"
                test_count += 1
                set_name = "test"
            else:
                sample_file = train_dir / f"custom_voice_{train_count:03d}.wav"
                train_count += 1
                set_name = "train"
            
            try:
                sf.write(str(sample_file), audio_data, recorder.sample_rate)
                samples_saved += 1
                print_success(f"✓ Saved to {set_name}: {sample_file.name}")
            except Exception as e:
                print_error(f"Failed to save sample: {e}")
    
    except KeyboardInterrupt:
        print_warning("\n\nRecording interrupted by user")
        return False, 0
    
    finally:
        listener.stop()
    
    print_info(f"\n{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print_info(f"{Colors.BOLD}Recording Session Complete{Colors.ENDC}")
    print_info(f"{Colors.BOLD}{'='*60}{Colors.ENDC}\n")
    
    if samples_saved > 0:
        print_success(f"✓ Successfully saved {samples_saved} custom voice samples!")
        print_info(f"  Training samples: {train_count}")
        print_info(f"  Testing samples:  {test_count}")
        print_info(f"  Saved to: {train_dir.parent}")
        
        # Quality feedback based on count
        if samples_saved < 5:
            print_warning("\n⚠ Only a few samples saved - model may not recognize your voice well")
            print_info("Consider re-running to add more samples later")
        elif samples_saved < target_count:
            print_info(f"\n✓ {samples_saved} samples is a good start!")
            print_info(f"Recording {target_count - samples_saved} more would improve accuracy further")
        else:
            print_success(f"\n🎉 Excellent! {samples_saved} samples provides robust training data!")
        
        return True, samples_saved
    else:
        print_warning("No samples were saved")
        return False, 0


def main():
    """Standalone CLI for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Record custom voice samples for wake word training')
    parser.add_argument('wake_word', help='Wake word to record')
    parser.add_argument('--output-dir', type=Path, default=Path.cwd(), 
                       help='Base directory for output (default: current directory)')
    parser.add_argument('--target', type=int, default=10,
                       help='Target number of samples to record (default: 10)')
    
    args = parser.parse_args()
    
    success, count = record_custom_samples(args.wake_word, args.output_dir, args.target)
    
    if success:
        print_success(f"\n✓ Recorded {count} samples")
        sys.exit(0)
    else:
        print_error("\n✗ Recording failed or cancelled")
        sys.exit(1)


if __name__ == "__main__":
    main()
