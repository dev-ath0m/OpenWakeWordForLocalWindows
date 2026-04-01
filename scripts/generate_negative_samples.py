#!/usr/bin/env python3
"""
Negative Sample Generation Module
Generates phonetically similar non-wake-word samples using TTS
"""
import sys
import os
from pathlib import Path
from typing import List
import warnings


def generate_negative_samples(wake_word: str, n_samples: int, n_samples_val: int, base_dir: Path) -> bool:
    """Generate negative samples (phonetically similar non-wake-words) using TTS"""
    print("=" * 70)
    print("Generating Negative Samples")
    print("=" * 70)
    
    model_name = wake_word.lower().replace(' ', '_')
    output_dir = base_dir / "trained_models" / model_name
    negative_train_dir = output_dir / "negative_train"
    negative_test_dir = output_dir / "negative_test"
    negative_train_dir.mkdir(parents=True, exist_ok=True)
    negative_test_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating phonetically similar adversarial phrases")
    print("These teach the model what NOT to trigger on")
    print(f"Target: {n_samples} training samples + {n_samples_val} test samples")
    
    try:
        # Import required modules
        from scripts.generate_positive_samples import (
            get_tts_models_config, 
            setup_tts_environment, 
            SuppressOutput,
            get_process_usage
        )
        
        # Import OpenWakeWord's adversarial text generator
        openwakeword_path = base_dir / "openwakeword"
        if str(openwakeword_path) not in sys.path:
            sys.path.insert(0, str(openwakeword_path))
        
        from openwakeword.data import generate_adversarial_texts
        
        # Patch torch.load for PyTorch 2.6 compatibility
        print(f"Analyzing phonemes for '{wake_word}'...")
        print("Note: Temporarily allowing unsafe pickle loading for deep-phonemizer (PyTorch 2.6 compatibility)")
        
        import torch
        original_torch_load = torch.load
        
        def patched_torch_load(*args, **kwargs):
            """Temporary patch for PyTorch 2.6 compatibility with deep-phonemizer"""
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        
        try:
            # Apply patch
            torch.load = patched_torch_load
            
            # Generate adversarial texts
            print("Generating adversarial text phrases...")
            print("This may take 1-2 minutes on first run (downloading phonemizer models)")
            print("Please wait...")
            
            adversarial_texts = generate_adversarial_texts(
                input_text=wake_word,
                N=n_samples,
                include_partial_phrase=1.0,
                include_input_words=0.2
            )
            
            print("Adversarial text generation complete!")
        finally:
            # Restore original torch.load
            torch.load = original_torch_load
        
        print(f"Generated {len(adversarial_texts)} phonetically similar phrases")
        print(f"Examples: {', '.join(adversarial_texts[:5])}")
        
        # Now generate audio samples
        from TTS.api import TTS
        import librosa
        from scipy.io import wavfile
        import numpy as np
        
        # Setup TTS environment
        device = setup_tts_environment(base_dir)
        
        # Setup TTS log file
        tts_log_path = base_dir / "tts_output.log"
        
        # Redirect TTS library logging to file
        import logging
        tts_logger = logging.getLogger('TTS')
        tts_logger.setLevel(logging.WARNING)
        file_handler = logging.FileHandler(tts_log_path, mode='a')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        tts_logger.addHandler(file_handler)
        
        # Use the same TTS models as positive sample generation
        tts_models_config = get_tts_models_config()
        
        # Separate fast and slow models
        fast_models = [m for m in tts_models_config if m.get("max_samples") is None]
        slow_models = [m for m in tts_models_config if m.get("max_samples") is not None]
        
        # Same distribution strategy as positive samples
        slow_total_target = int(n_samples * 0.02)
        slow_samples_each = slow_total_target // len(slow_models) if slow_models else 0
        
        fast_total_target = n_samples - (slow_samples_each * len(slow_models))
        fast_samples_each = fast_total_target // len(fast_models) if fast_models else 0
        
        # Test samples: only use slow models
        slow_test_each = n_samples_val // len(slow_models) if slow_models else 0
        
        train_count = 0
        test_count = 0
        failed_count = 0
        
        # Combine models with their targets
        models_with_targets = []
        for model in fast_models:
            models_with_targets.append((model, fast_samples_each, 0))
        for model in slow_models:
            train_target = min(slow_samples_each, model.get("max_samples", slow_samples_each))
            models_with_targets.append((model, train_target, slow_test_each))
        
        # Generate samples using each model configuration
        import threading
        import time as time_module
        
        for model_config, model_train_target, model_test_target in models_with_targets:
            
            model_name = model_config["model"]
            model_short = model_name.split('/')[-1]
            
            try:
                # Load model with spinner feedback
                init_done = threading.Event()
                def init_spinner():
                    spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
                    idx = 0
                    start_time = time_module.time()
                    while not init_done.is_set():
                        elapsed = time_module.time() - start_time
                        print(f"\r  {spinner_chars[idx]} Loading {model_short}... (elapsed: {int(elapsed)}s)", 
                              end='', flush=True)
                        idx = (idx + 1) % len(spinner_chars)
                        time_module.sleep(0.1)
                    print("\r" + " " * 80 + "\r", end='', flush=True)
                
                init_thread = threading.Thread(target=init_spinner)
                init_thread.daemon = True
                init_thread.start()
                
                # Load model
                warnings.filterwarnings('ignore', message='.*gpu.*will be deprecated.*')
                
                with SuppressOutput(log_file_path=str(tts_log_path)):
                    tts = TTS(model_name=model_name)
                
                init_done.set()
                init_thread.join(timeout=0.5)
                
                # GPU transfer
                gpu_done = threading.Event()
                def gpu_spinner():
                    spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
                    idx = 0
                    start_time = time_module.time()
                    while not gpu_done.is_set():
                        elapsed = time_module.time() - start_time
                        print(f"\r  {spinner_chars[idx]} Model ready on {device.upper()}... (elapsed: {int(elapsed)}s)", 
                              end='', flush=True)
                        idx = (idx + 1) % len(spinner_chars)
                        time_module.sleep(0.1)
                    print("\r" + " " * 80 + "\r", end='', flush=True)
                
                gpu_thread = threading.Thread(target=gpu_spinner)
                gpu_thread.daemon = True
                gpu_thread.start()
                
                # Force synthesizer and vocoder to GPU
                if device == "cuda:0":
                    if hasattr(tts, 'synthesizer') and tts.synthesizer is not None:
                        if hasattr(tts.synthesizer, 'tts_model'):
                            tts.synthesizer.tts_model = tts.synthesizer.tts_model.to(device)
                    if hasattr(tts, 'vocoder') and tts.vocoder is not None:
                        if hasattr(tts.vocoder, 'model'):
                            tts.vocoder.model = tts.vocoder.model.to(device)
                
                gpu_done.set()
                gpu_thread.join(timeout=0.5)
                
                print(f"Model {model_short} loaded successfully on {device.upper()}")
                
                # Get native sample rate
                try:
                    if hasattr(tts, 'synthesizer') and hasattr(tts.synthesizer, 'output_sample_rate'):
                        native_sr = tts.synthesizer.output_sample_rate
                    else:
                        native_sr = 22050
                except:
                    native_sr = 22050
                
                # Initialize progress bar
                from tqdm import tqdm
                import soundfile as sf
                
                gender = model_config.get("gender", "voice")
                speaker = model_config.get("speaker", "")
                speaker_suffix = f"_{speaker}_{gender}" if speaker else f"_{gender}"
                model_desc = f"{model_short}{speaker_suffix}"
                total_target = model_train_target + model_test_target
                pbar = tqdm(total=total_target, desc=f"Negative ({model_desc})", 
                           unit="samples", initial=0,
                           file=sys.stdout, mininterval=0.5, dynamic_ncols=True,
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
                
                try:
                    # Generate training samples
                    text_idx = 0
                    model_train_count = 0
                    while train_count < n_samples and text_idx < len(adversarial_texts) and model_train_count < model_train_target:
                        text = adversarial_texts[text_idx % len(adversarial_texts)]
                        text_idx += 1
                        
                        output_file = negative_train_dir / f"neg_{train_count}.wav"
                        temp_file = output_file.parent / f"temp_{output_file.name}"
                        
                        try:
                            # Generate with speed variation
                            speed = 1.0 + np.random.uniform(-0.1, 0.1)
                            
                            kwargs = {
                                'text': text,
                                'file_path': str(temp_file),
                                'speed': speed
                            }
                            
                            if 'speaker' in model_config:
                                kwargs['speaker'] = model_config['speaker']
                            if 'language' in model_config:
                                kwargs['language'] = model_config['language']
                            
                            # Generate sample
                            with SuppressOutput():
                                tts.tts_to_file(**kwargs)
                            
                            # Load, resample, trim, and save
                            audio, sr = librosa.load(str(temp_file), sr=native_sr, mono=True)
                            audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)
                            
                            if len(audio_trimmed) / native_sr > 4.0:
                                temp_file.unlink()
                                failed_count += 1
                                continue
                            
                            if native_sr != 16000:
                                audio_16k = librosa.resample(audio_trimmed, orig_sr=native_sr, target_sr=16000)
                            else:
                                audio_16k = audio_trimmed
                            
                            if len(audio_16k) > 0:
                                max_val = np.abs(audio_16k).max()
                                if max_val > 0:
                                    audio_16k = audio_16k / max_val * 0.95
                            
                            sf.write(str(output_file), audio_16k, 16000, subtype='PCM_16')
                            temp_file.unlink()
                            
                            train_count += 1
                            model_train_count += 1
                            
                            # Update progress
                            if model_train_count == 1 or model_train_count % 10 == 0:
                                cpu_usage, gpu_usage = get_process_usage()
                                text_display = text if len(text) <= 30 else text[:27] + '...'
                                pbar.n = model_train_count + model_test_count
                                pbar.set_postfix({
                                    'Train': model_train_count,
                                    'Test': model_test_count,
                                    'Failed': failed_count,
                                    'CPU': f'{cpu_usage:.0f}%',
                                    'GPU': gpu_usage,
                                    'Text': text_display
                                })
                                pbar.refresh()
                            
                        except Exception as e:
                            # Log error
                            with open(tts_log_path, 'a') as f:
                                import traceback
                                f.write(f"\n=== Error generating negative sample {train_count} ===\n")
                                f.write(f"Model: {model_short}, Text: {text}\n")
                                f.write(f"Error: {str(e)}\n")
                                f.write(traceback.format_exc())
                                f.write("\n")
                            failed_count += 1
                            if temp_file.exists():
                                temp_file.unlink()
                            if output_file.exists():
                                output_file.unlink()
                            continue
                    
                    # Generate test samples
                    text_idx = 0
                    model_test_count = 0
                    while test_count < n_samples_val and text_idx < len(adversarial_texts) * 2 and model_test_count < model_test_target:
                        text = adversarial_texts[text_idx % len(adversarial_texts)]
                        text_idx += 1
                        
                        output_file = negative_test_dir / f"neg_test_{test_count}.wav"
                        temp_file = output_file.parent / f"temp_{output_file.name}"
                        
                        try:
                            speed = 1.0 + np.random.uniform(-0.1, 0.1)
                            
                            kwargs = {
                                'text': text,
                                'file_path': str(temp_file),
                                'speed': speed
                            }
                            
                            if 'speaker' in model_config:
                                kwargs['speaker'] = model_config['speaker']
                            if 'language' in model_config:
                                kwargs['language'] = model_config['language']
                            
                            with SuppressOutput():
                                tts.tts_to_file(**kwargs)
                            
                            audio, sr = librosa.load(str(temp_file), sr=native_sr, mono=True)
                            audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)
                            
                            if len(audio_trimmed) / native_sr > 4.0:
                                temp_file.unlink()
                                failed_count += 1
                                continue
                            
                            if native_sr != 16000:
                                audio_16k = librosa.resample(audio_trimmed, orig_sr=native_sr, target_sr=16000)
                            else:
                                audio_16k = audio_trimmed
                            
                            if len(audio_16k) > 0:
                                max_val = np.abs(audio_16k).max()
                                if max_val > 0:
                                    audio_16k = audio_16k / max_val * 0.95
                            
                            sf.write(str(output_file), audio_16k, 16000, subtype='PCM_16')
                            temp_file.unlink()
                            
                            test_count += 1
                            model_test_count += 1
                            
                            # Update progress bar
                            cpu_usage, gpu_usage = get_process_usage()
                            text_display = text if len(text) <= 30 else text[:27] + '...'
                            pbar.n = model_train_count + model_test_count
                            pbar.set_postfix({
                                'Train': model_train_count,
                                'Test': model_test_count,
                                'Failed': failed_count,
                                'CPU': f'{cpu_usage:.0f}%',
                                'GPU': gpu_usage,
                                'Text': text_display
                            })
                            pbar.refresh()
                            
                        except Exception as e:
                            with open(tts_log_path, 'a') as f:
                                import traceback
                                f.write(f"\n=== Error generating negative test sample {test_count} ===\n")
                                f.write(f"Model: {model_short}, Text: {text}\n")
                                f.write(f"Error: {str(e)}\n")
                                f.write(traceback.format_exc())
                                f.write("\n")
                            failed_count += 1
                            if temp_file.exists():
                                temp_file.unlink()
                            if output_file.exists():
                                output_file.unlink()
                            continue
                    
                    if train_count >= n_samples and test_count >= n_samples_val:
                        break
                    
                    # Close progress bar
                    pbar.close()
                        
                except Exception as model_error:
                    if 'pbar' in locals():
                        pbar.close()
                    with open(tts_log_path, 'a') as f:
                        import traceback
                        f.write(f"\n=== Negative samples: Model {model_short} failed ===\n")
                        f.write(f"Error: {str(model_error)}\n")
                        f.write(traceback.format_exc())
                        f.write("\n")
                    print(f"WARNING: Model {model_short} failed: {model_error}")
                    print(f"Full error details logged to: {tts_log_path}")
                    continue
                
            except Exception as e:
                print(f"WARNING: Model {model_short} failed to load: {e}")
                continue
            
            if train_count >= n_samples and test_count >= n_samples_val:
                break
        
        # Final validation
        total_valid = train_count + test_count
        total_target = n_samples + n_samples_val
        if total_valid < total_target * 0.8:
            print(f"\nERROR: Insufficient negative samples generated: {train_count} train + {test_count} test / {total_target} target")
            print("Training requires at least 80% of target samples")
            return False
        
        print(f"\nGenerated {train_count} training samples and {test_count} test samples")
        if failed_count > 0:
            total_attempts = total_valid + failed_count
            print(f"Failed: {failed_count} samples ({failed_count/total_attempts*100:.1f}%)")
        
        if train_count < n_samples or test_count < n_samples_val:
            print(f"WARNING: Generated {train_count}/{n_samples} train + {test_count}/{n_samples_val} test (some models may have failed)")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Negative sample generation failed: {e}")
        print("This is a critical error - training cannot proceed without negative samples")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate negative wake word samples")
    parser.add_argument("--wake-word", required=True, help="Wake word to generate negative samples for")
    parser.add_argument("--n-samples", type=int, default=3000, help="Number of training samples")
    parser.add_argument("--n-samples-val", type=int, default=300, help="Number of validation samples")
    parser.add_argument("--base-dir", type=Path, default=Path.cwd(), help="Base directory")
    
    args = parser.parse_args()
    
    success = generate_negative_samples(
        wake_word=args.wake_word,
        n_samples=args.n_samples,
        n_samples_val=args.n_samples_val,
        base_dir=args.base_dir
    )
    
    sys.exit(0 if success else 1)
