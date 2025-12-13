#!/usr/bin/env python3
"""
Positive Sample Generation Module
Generates wake word samples using multiple TTS models
"""
import sys
import os
from pathlib import Path
from typing import List, Tuple
import warnings


def get_tts_models_config():
    """Get the unified TTS model configuration for both positive and negative samples
    
    Models are categorized by speed:
    - Fast models (1-2 sec/sample): Used for bulk generation
    - Slow models (8-12 sec/sample): Limited to 20 samples for voice variety
    """
    return [
        # Fast models (1-2 seconds per sample on GPU) - no limit
        {"model": "tts_models/en/ljspeech/vits", "gender": "female", "max_samples": None},
        {"model": "tts_models/en/ljspeech/fast_pitch", "gender": "female", "max_samples": None},
        {"model": "tts_models/en/ljspeech/glow-tts", "gender": "female", "max_samples": None},
        {"model": "tts_models/en/vctk/vits", "speaker": "p225", "gender": "female", "max_samples": None},
        {"model": "tts_models/en/vctk/vits", "speaker": "p226", "gender": "male", "max_samples": None},
        {"model": "tts_models/en/jenny/jenny", "gender": "female", "max_samples": None},
        {"model": "tts_models/multilingual/multi-dataset/your_tts", "speaker": "male-en-2", "language": "en", "max_samples": None},
        {"model": "tts_models/multilingual/multi-dataset/your_tts", "speaker": "female-en-5", "language": "en", "max_samples": None},
        
        # Slow but high-quality models (8-12 seconds per sample) - limited to 20 samples for variety
        {"model": "tts_models/en/ljspeech/tacotron2-DDC", "gender": "female", "max_samples": 20},
        {"model": "tts_models/en/ljspeech/tacotron2-DCA", "gender": "female", "max_samples": 20},
        {"model": "tts_models/en/ljspeech/neural_hmm", "gender": "female", "max_samples": 20},
        {"model": "tts_models/en/sam/tacotron-DDC", "gender": "male", "max_samples": 20},
    ]


def setup_tts_environment(base_dir: Path):
    """Setup TTS environment with PyTorch 2.6 compatibility fixes"""
    os.environ['TTS_HOME'] = str(base_dir / 'tts')
    
    # Fix PyTorch 2.6 weights_only issue for TTS models
    import torch
    from TTS.utils.radam import RAdam
    from collections import defaultdict
    torch.serialization.add_safe_globals([RAdam, defaultdict, dict])
    
    # Force CUDA device selection
    if torch.cuda.is_available():
        # Set default CUDA device
        torch.cuda.set_device(0)  # Use first NVIDIA GPU
        # Force PyTorch to use CUDA
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        device = "cuda:0"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        device = "cpu"
        print("CUDA not available, using CPU")
    
    # Suppress verbose TTS logging
    import logging
    logging.getLogger('TTS').setLevel(logging.CRITICAL)
    
    return device


class SuppressOutput:
    """Context manager to suppress TTS library output (Windows-compatible)"""
    def __init__(self, log_file_path=None):
        self.log_file_path = log_file_path
        
    def __enter__(self):
        # Save original streams
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        
        # Redirect to log file or null device using Python-level redirection only
        # (OS-level redirection with os.dup2 causes issues with tqdm on Windows)
        if self.log_file_path:
            self.redirect_file = open(self.log_file_path, 'a', encoding='utf-8', buffering=1)
            sys.stdout = self.redirect_file
            sys.stderr = self.redirect_file
        else:
            import io
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original streams
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        
        # Close redirect file if it was opened
        if hasattr(self, 'redirect_file'):
            try:
                self.redirect_file.close()
            except:
                pass


def get_process_usage():
    """Get current process CPU and GPU usage for progress monitoring"""
    try:
        import psutil
        # Get CPU usage for current process
        process = psutil.Process()
        cpu_percent = process.cpu_percent(interval=0.1)
        
        # Get GPU usage if available
        try:
            import pynvml
            gpu_util = "N/A"
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = f"{util.gpu}%"
            except:
                pass
        except ImportError:
            # pynvml not available, try nvidia-smi
            try:
                import torch
                import subprocess
                if torch.cuda.is_available():
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits', '--id=0'],
                        capture_output=True, text=True, timeout=0.5
                    )
                    if result.returncode == 0:
                        gpu_util = f"{result.stdout.strip()}%"
                    else:
                        gpu_util = "N/A"
                else:
                    gpu_util = "N/A"
            except:
                gpu_util = "N/A"
        
        return cpu_percent, gpu_util
    except:
        return 0.0, "N/A"


def generate_tts_samples_with_model(
    tts,
    model_config: dict,
    texts: List[str],
    output_train_dir: Path,
    output_test_dir: Path,
    model_train_target: int,
    model_test_target: int,
    train_count: int,
    test_count: int,
    failed_count: int,
    base_dir: Path,
    sample_type: str = "Positive"
) -> Tuple[int, int, int]:
    """
    Generate TTS samples for a single model configuration.
    
    Args:
        tts: Loaded TTS model instance
        model_config: Model configuration dict
        texts: List of texts to synthesize
        output_train_dir: Directory for training samples
        output_test_dir: Directory for test samples
        model_train_target: Number of training samples this model should generate
        model_test_target: Number of test samples this model should generate
        train_count: Current global training sample count
        test_count: Current global test sample count
        failed_count: Current failed sample count
        base_dir: Base directory for project
        sample_type: "Positive" or "Negative" for progress bar label
    
    Returns:
        Tuple of (train_count, test_count, failed_count)
    """
    import numpy as np
    import librosa
    import soundfile as sf
    from tqdm import tqdm
    
    tts_log_path = base_dir / "tts_output.log"
    model_path = model_config["model"]
    model_short = model_path.split('/')[-1]
    
    # Get native sample rate
    try:
        if hasattr(tts, 'synthesizer') and hasattr(tts.synthesizer, 'output_sample_rate'):
            native_sr = tts.synthesizer.output_sample_rate
        else:
            native_sr = 22050
    except:
        native_sr = 22050
    
    print(f"Starting generation with {len(texts)} text variant(s): {', '.join(texts)}")
    print(f"Native TTS sample rate: {native_sr}Hz → Resampling to 16kHz")
    
    # Setup progress bar (show only this model's target, not global total)
    need_resample = (native_sr != 16000)
    gender = model_config.get("gender", "voice")
    speaker = model_config.get("speaker", "")
    speaker_suffix = f"_{speaker}_{gender}" if speaker else f"_{gender}"
    model_desc = f"{model_short}{speaker_suffix}"
    
    # Calculate this model's contribution to the total (not the global total)
    model_total_target = model_train_target + model_test_target
    pbar = tqdm(total=model_total_target, desc=f"{sample_type} ({model_desc})", 
               unit="samples", initial=0,  # Start at 0 for this model
               file=sys.stdout, mininterval=0.5, dynamic_ncols=True,
               bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    # Force initial display
    pbar.refresh()
    
    progress_update_interval = 10
    text_idx = 0
    model_train_count = 0
    model_test_count = 0
    
    # Generate training samples - distribute evenly across text variants
    samples_per_text = max(1, model_train_target // len(texts))
    for text in texts:
        combo_count = 0
        while combo_count < samples_per_text and model_train_count < model_train_target:
            output_file = output_train_dir / f"{model_short}_{train_count}.wav"
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
                
                # Generate sample with suppressed output
                with SuppressOutput(log_file_path=str(tts_log_path)):
                    tts.tts_to_file(**kwargs)
                
                # Load, resample if needed, trim silence, and save
                audio, sr = librosa.load(str(temp_file), sr=native_sr, mono=True)
                
                # Trim silence from start and end
                audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)
                
                # Reject if too long (> 4 seconds)
                if len(audio_trimmed) / native_sr > 4.0:
                    temp_file.unlink()
                    failed_count += 1
                    continue
                
                # Resample to 16kHz if needed
                if native_sr != 16000:
                    audio_16k = librosa.resample(audio_trimmed, orig_sr=native_sr, target_sr=16000)
                else:
                    audio_16k = audio_trimmed
                
                # Normalize audio
                if len(audio_16k) > 0:
                    max_val = np.abs(audio_16k).max()
                    if max_val > 0:
                        audio_16k = audio_16k / max_val * 0.95
                
                # Save as 16-bit PCM WAV
                sf.write(str(output_file), audio_16k, 16000, subtype='PCM_16')
                
                # Clean up temp file
                temp_file.unlink()
                
                train_count += 1
                model_train_count += 1
                combo_count += 1
                
                # Update progress every 10 samples to reduce overhead
                if model_train_count % progress_update_interval == 0 or model_train_count == 1:
                    cpu_usage, gpu_usage = get_process_usage()
                    text_display = text if len(text) <= 20 else text[:17] + '...'
                    pbar.n = model_train_count + model_test_count
                    pbar.set_postfix({
                        'Variant': f'"{text_display}"',
                        'CPU': f'{cpu_usage:.0f}%',
                        'GPU': gpu_usage
                    })
                    pbar.refresh()
                
            except Exception as e:
                # Log detailed error to file for debugging
                with open(tts_log_path, 'a') as f:
                    import traceback
                    f.write(f"\n=== Error generating sample {train_count} ===\n")
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
    test_samples_per_text = max(1, model_test_target // len(texts))
    
    while model_test_count < model_test_target and text_idx < len(texts) * 2:
        text = texts[text_idx % len(texts)]
        text_idx += 1
        
        output_file = output_test_dir / f"{model_short}_test_{test_count}.wav"
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
            
            # Generate sample with suppressed output
            with SuppressOutput(log_file_path=str(tts_log_path)):
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
            
            test_count += 1
            model_test_count += 1
            
            # Update progress bar
            cpu_usage, gpu_usage = get_process_usage()
            text_display = text if len(text) <= 20 else text[:17] + '...'
            pbar.n = model_train_count + model_test_count
            pbar.set_postfix({
                'Variant': f'"{text_display}"',
                'CPU': f'{cpu_usage:.0f}%',
                'GPU': gpu_usage
            })
            pbar.refresh()
            
        except Exception as e:
            # Log detailed error to file
            with open(tts_log_path, 'a') as f:
                import traceback
                f.write(f"\n=== Error generating test sample {test_count} ===\n")
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
    
    # Close progress bar
    pbar.close()
    
    # Report completion using model-specific counts
    if failed_count > 0:
        total_attempts = model_train_count + model_test_count + failed_count
        fail_pct = (failed_count / total_attempts * 100) if total_attempts > 0 else 0
        print(f"Completed model '{model_short}': {model_train_count} train + {model_test_count} test, {failed_count} failed ({fail_pct:.1f}%)")
    else:
        print(f"Completed model '{model_short}': {model_train_count} train + {model_test_count} test")
    
    return train_count, test_count, failed_count


def generate_positive_samples(wake_word: str, pronunciations: List[str], n_samples: int, n_samples_val: int, base_dir: Path) -> bool:
    """Generate positive samples (wake word pronunciations) using TTS"""
    print("=" * 70)
    print("Generating Positive Samples")
    print("=" * 70)
    
    model_name = wake_word.lower().replace(' ', '_')
    output_dir = base_dir / "trained_models" / model_name
    positive_train_dir = output_dir / "positive_train"
    positive_test_dir = output_dir / "positive_test"
    positive_train_dir.mkdir(parents=True, exist_ok=True)
    positive_test_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from TTS.api import TTS
        
        # Setup environment
        device = setup_tts_environment(base_dir)
        tts_models = get_tts_models_config()
        
        # Separate fast and slow models
        fast_models = [m for m in tts_models if m.get("max_samples") is None]
        slow_models = [m for m in tts_models if m.get("max_samples") is not None]
        
        # Calculate distribution
        slow_total_target = int(n_samples * 0.02)  # 2% for slow models
        slow_samples_each = slow_total_target // len(slow_models) if slow_models else 0
        
        fast_total_target = n_samples - (slow_samples_each * len(slow_models))
        fast_samples_each = fast_total_target // len(fast_models) if fast_models else 0
        
        # Test samples: only use slow models for higher quality validation
        slow_test_each = n_samples_val // len(slow_models) if slow_models else 0
        
        print(f"Using {len(tts_models)} TTS models ({len(fast_models)} fast, {len(slow_models)} slow)")
        print(f"Target: {n_samples} training samples + {n_samples_val} test samples")
        print(f"Training distribution:")
        print(f"  - Fast models: ~{fast_samples_each} samples each ({len(fast_models)} models)")
        print(f"  - Slow models: ~{slow_samples_each} samples each ({len(slow_models)} models, 2% total)")
        print(f"Test distribution:")
        print(f"  - Slow models only: ~{slow_test_each} samples each (high quality for validation)")
        print(f"Device: {device.upper()}")
        print("\n[Model Loading Strategy]")
        print("Each model is loaded ONCE and generates all its samples (all text variations)")
        print("before unloading and moving to the next model.")
        print("This ensures efficient GPU memory usage and faster generation.\n")
        print("Starting sample generation...\n")
        
        train_count = 0
        test_count = 0
        failed_count = 0
        
        # Combine models with their targets
        models_with_targets = []
        for model in fast_models:
            # Fast models: training only, no test samples
            models_with_targets.append((model, fast_samples_each, 0))
        for model in slow_models:
            # Slow models: both training and test samples
            train_target = min(slow_samples_each, model.get("max_samples", slow_samples_each))
            models_with_targets.append((model, train_target, slow_test_each))
        
        for model_config, model_train_target, model_test_target in models_with_targets:
            model_path = model_config["model"]
            model_short = model_path.split('/')[-1]
            
            # Clear any previous progress line
            print("\r" + " " * 120 + "\r", end='', flush=True)
            print(f"Loading model: {model_short}...")
            
            try:
                # Setup TTS log file
                tts_log_path = base_dir / "tts_output.log"
                
                # Redirect TTS library logging to file
                import logging
                tts_logger = logging.getLogger('TTS')
                tts_logger.setLevel(logging.WARNING)
                file_handler = logging.FileHandler(tts_log_path, mode='a')
                file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                tts_logger.addHandler(file_handler)
                
                # Load TTS model (suppress verbose output)
                warnings.filterwarnings('ignore', message='.*gpu.*will be deprecated.*')
                
                with SuppressOutput(log_file_path=str(tts_log_path)):
                    tts = TTS(model_name=model_path)
                
                # Manually force synthesizer and vocoder to GPU
                if device == "cuda:0":
                    if hasattr(tts, 'synthesizer') and tts.synthesizer is not None:
                        if hasattr(tts.synthesizer, 'tts_model'):
                            tts.synthesizer.tts_model = tts.synthesizer.tts_model.to(device)
                    if hasattr(tts, 'vocoder') and tts.vocoder is not None:
                        if hasattr(tts.vocoder, 'model'):
                            tts.vocoder.model = tts.vocoder.model.to(device)
                
                print(f"Model {model_short} loaded successfully on {device.upper()}")
                
                # Generate samples
                train_count, test_count, failed_count = generate_tts_samples_with_model(
                    tts=tts,
                    model_config=model_config,
                    texts=pronunciations,
                    output_train_dir=positive_train_dir,
                    output_test_dir=positive_test_dir,
                    model_train_target=model_train_target,
                    model_test_target=model_test_target,
                    train_count=train_count,
                    test_count=test_count,
                    failed_count=failed_count,
                    base_dir=base_dir,
                    sample_type="Positive"
                )
                        
            except Exception as e:
                # Log full error details to file
                tts_log_path = base_dir / "tts_output.log"
                with open(tts_log_path, 'a') as f:
                    import traceback
                    f.write(f"\n=== Model {model_short} failed ===\n")
                    f.write(f"Error: {str(e)}\n")
                    f.write(traceback.format_exc())
                    f.write("\n")
                print(f"WARNING: Model {model_short} failed: {e}")
                print(f"Full error details logged to: {tts_log_path}")
                continue
            
            if train_count >= n_samples and test_count >= n_samples_val:
                break
        
        print(f"\nGenerated {train_count} training samples and {test_count} test samples")
        total_valid = train_count + test_count
        total_attempts = total_valid + failed_count
        if failed_count > 0:
            fail_pct = (failed_count / total_attempts * 100)
            print(f"Total attempts: {total_attempts} (Success rate: {100-fail_pct:.1f}%)")
        print(f"TTS models cached in: {base_dir / 'tts'}")
        return train_count > 0 and test_count > 0
        
    except Exception as e:
        print(f"ERROR: Positive sample generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate positive wake word samples")
    parser.add_argument("--wake-word", required=True, help="Wake word to generate samples for")
    parser.add_argument("--pronunciations", nargs='+', required=True, help="Pronunciation variations")
    parser.add_argument("--n-samples", type=int, default=3000, help="Number of training samples")
    parser.add_argument("--n-samples-val", type=int, default=300, help="Number of validation samples")
    parser.add_argument("--base-dir", type=Path, default=Path.cwd(), help="Base directory")
    
    args = parser.parse_args()
    
    success = generate_positive_samples(
        wake_word=args.wake_word,
        pronunciations=args.pronunciations,
        n_samples=args.n_samples,
        n_samples_val=args.n_samples_val,
        base_dir=args.base_dir
    )
    
    sys.exit(0 if success else 1)
