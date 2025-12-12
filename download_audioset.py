"""
AudioSet Balanced + Eval Subset Downloader
Downloads ~40,000 audio clips from YouTube for training wake word models
"""

import os
import sys
import csv
import subprocess
import urllib.request
from pathlib import Path
import shutil
import logging
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('audioset_download.log'),
        logging.StreamHandler()
    ]
)

# URLs for AudioSet metadata CSV files
METADATA_URLS = {
    'balanced_train': 'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv',
    'eval': 'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv',
    'class_labels': 'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv'
}

# Target dataset size (~40GB) and balanced sampling strategy
TARGET_DATASET_SIZE_GB = 40  # Target total size in GB
ESTIMATED_BYTES_PER_CLIP = 1.2 * 1024 * 1024  # ~1.2MB per 10-second 16kHz mono WAV
MAX_CLIPS_TARGET = int((TARGET_DATASET_SIZE_GB * 1024 * 1024 * 1024) / ESTIMATED_BYTES_PER_CLIP)  # ~35,000 clips
MIN_SAMPLES_PER_CATEGORY = 5  # Minimum samples per category before moving to next round
MAX_SAMPLES_PER_CATEGORY = 100  # Maximum samples per category to prevent over-representation
MAX_WORKERS = 8  # Number of parallel download threads

def check_dependencies():
    """Check if yt-dlp and ffmpeg are installed"""
    dependencies = {
        'yt-dlp': False,
        'ffmpeg': False
    }
    
    # Check yt-dlp
    try:
        result = subprocess.run(['yt-dlp', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            dependencies['yt-dlp'] = True
            logging.info(f"yt-dlp version: {result.stdout.strip()}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Check ffmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            dependencies['ffmpeg'] = True
            logging.info(f"ffmpeg found")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    return dependencies

def install_ytdlp():
    """Install yt-dlp if not already installed"""
    logging.info("Installing yt-dlp...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-U', 'yt-dlp'],
                      check=True, capture_output=True)
        logging.info("yt-dlp installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to install yt-dlp: {e}")
        return False

def download_metadata(output_dir):
    """Download AudioSet CSV metadata files"""
    metadata_dir = os.path.join(output_dir, 'metadata')
    os.makedirs(metadata_dir, exist_ok=True)
    
    logging.info("Downloading AudioSet metadata...")
    for name, url in METADATA_URLS.items():
        output_path = os.path.join(metadata_dir, f'{name}.csv')
        if os.path.exists(output_path):
            logging.info(f"  Metadata already exists: {name}.csv")
            continue
        
        try:
            logging.info(f"  Downloading {name}.csv...")
            urllib.request.urlretrieve(url, output_path)
            logging.info(f"  [OK] {name}.csv")
        except Exception as e:
            logging.error(f"  [ERROR] Failed to download {name}.csv: {e}")
            return False
    
    return metadata_dir

def load_class_labels(metadata_dir):
    """Load AudioSet class labels and return mapping of ID to name"""
    class_labels = {}
    labels_path = os.path.join(metadata_dir, 'class_labels.csv')
    
    if not os.path.exists(labels_path):
        logging.warning("Class labels file not found, category tracking disabled")
        return class_labels
    
    with open(labels_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        # Skip header rows (first 3 lines)
        for _ in range(3):
            next(reader)
        
        for row in reader:
            if len(row) >= 3:
                label_id = row[1].strip()  # mid (e.g., /m/0dgw9r)
                label_name = row[2].strip()  # display_name (e.g., "Bass drum")
                class_labels[label_id] = label_name
    
    logging.info(f"Loaded {len(class_labels)} class labels")
    return class_labels

def parse_labels(label_string):
    """Parse quoted label string into list of label IDs"""
    # Labels are in format: "\"/m/09x0r\",\"/t/dd00088\""
    labels = []
    for part in label_string.split(','):
        label = part.strip().strip('"').strip('\\"')
        if label:
            labels.append(label)
    return labels

def count_existing_files(audio_dir):
    """Count how many audio files have already been downloaded"""
    if not os.path.exists(audio_dir):
        return 0
    return len([f for f in os.listdir(audio_dir) if f.endswith('.wav')])

def download_audio_segment(youtube_id, start_time, end_time, output_path, max_retries=2):
    """Download a single audio segment from YouTube
    
    Args:
        youtube_id: YouTube video ID
        start_time: Start time in seconds
        end_time: End time in seconds
        output_path: Where to save the audio file
        max_retries: Number of download attempts
    
    Returns:
        bool: True if successful, False otherwise
    """
    if os.path.exists(output_path):
        return True  # Already downloaded
    
    duration = end_time - start_time
    temp_output = output_path.replace('.wav', '.temp')
    
    # Build yt-dlp command
    # Download audio, extract segment, convert to 16kHz mono WAV
    cmd = [
        'yt-dlp',
        '--quiet',
        '--no-warnings',
        '--extract-audio',
        '--audio-format', 'wav',
        '--audio-quality', '0',
        '--postprocessor-args', f'ffmpeg:-ss {start_time} -t {duration} -ar 16000 -ac 1',
        '-o', temp_output,
        f'https://www.youtube.com/watch?v={youtube_id}'
    ]
    
    for attempt in range(max_retries):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            # Check if file was created (yt-dlp adds .wav extension)
            possible_files = [
                temp_output,
                temp_output + '.wav',
                temp_output.replace('.temp', '.temp.wav')
            ]
            
            downloaded_file = None
            for f in possible_files:
                if os.path.exists(f):
                    downloaded_file = f
                    break
            
            if downloaded_file:
                # Rename to final output path
                shutil.move(downloaded_file, output_path)
                return True
            
        except subprocess.TimeoutExpired:
            logging.warning(f"  Timeout downloading {youtube_id} (attempt {attempt + 1}/{max_retries})")
        except Exception as e:
            logging.warning(f"  Error downloading {youtube_id}: {e}")
        
        # Clean up any temp files
        for f in [temp_output, temp_output + '.wav']:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass
    
    return False

def download_from_csv(csv_path, output_dir, subset_name, class_labels=None, limit=None, ensure_min_per_category=True):
    """Download audio files listed in a CSV file with category balancing
    
    Args:
        csv_path: Path to CSV file with video IDs and timestamps
        output_dir: Directory to save audio files
        subset_name: Name of subset (for logging)
        class_labels: Dictionary mapping label IDs to names
        limit: Optional limit on number of files to download (for testing)
        ensure_min_per_category: If True, prioritize downloading at least MIN_SAMPLES_PER_CATEGORY from each category
    
    Returns:
        dict: Statistics about the download including category coverage
    """
    audio_dir = os.path.join(output_dir, subset_name)
    os.makedirs(audio_dir, exist_ok=True)
    
    # Read CSV file with labels
    segments = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        # Skip header rows (first 3 lines)
        for _ in range(3):
            next(reader)
        
        for row in reader:
            if len(row) >= 4:  # Need at least 4 columns to get labels
                youtube_id = row[0].strip()
                start_time = float(row[1])
                end_time = float(row[2])
                labels = parse_labels(row[3]) if len(row) > 3 else []
                segments.append((youtube_id, start_time, end_time, labels))
            elif len(row) >= 3:
                # Fallback for rows without labels
                youtube_id = row[0].strip()
                start_time = float(row[1])
                end_time = float(row[2])
                segments.append((youtube_id, start_time, end_time, []))
    
    # Build category index for balanced downloading
    category_samples = {}  # label_id -> list of segment indices
    category_downloaded = {}  # label_id -> count of downloaded samples
    
    for idx, segment in enumerate(segments):
        labels = segment[3] if len(segment) > 3 else []
        for label in labels:
            if label not in category_samples:
                category_samples[label] = []
                category_downloaded[label] = 0
            category_samples[label].append(idx)
    
    # Check which files already exist and update category counts
    existing_files = set()
    if os.path.exists(audio_dir):
        existing_files = {f for f in os.listdir(audio_dir) if f.endswith('.wav')}
        
        # Update category counts for existing files
        for idx, segment in enumerate(segments):
            youtube_id = segment[0]
            if f'Y{youtube_id}.wav' in existing_files:
                labels = segment[3] if len(segment) > 3 else []
                for label in labels:
                    if label in category_downloaded:
                        category_downloaded[label] += 1
    
    total_count = len(segments)
    existing_count = len(existing_files)
    
    # Determine download order - prioritize under-represented categories
    if ensure_min_per_category and not limit:
        download_order = []
        downloaded_indices = set()
        
        # Round-robin sampling: cycle through categories to build balanced dataset
        logging.info(f"Building balanced dataset up to ~{MAX_CLIPS_TARGET:,} clips (~{TARGET_DATASET_SIZE_GB}GB)...")
        logging.info(f"Strategy: {MIN_SAMPLES_PER_CATEGORY}-{MAX_SAMPLES_PER_CATEGORY} samples per category, round-robin across all categories")
        
        round_num = 0
        samples_per_round = MIN_SAMPLES_PER_CATEGORY
        
        while len(download_order) < len(segments) and len(download_order) < MAX_CLIPS_TARGET:
            round_num += 1
            added_this_round = 0
            
            # Sort categories by current download count (least to most)
            sorted_categories = sorted(category_samples.items(), 
                                      key=lambda x: category_downloaded.get(x[0], 0))
            
            # Try to add samples from each category
            for label, sample_indices in sorted_categories:
                current_count = category_downloaded.get(label, 0)
                
                # Skip if category already at max
                if current_count >= MAX_SAMPLES_PER_CATEGORY:
                    continue
                
                # Calculate how many samples this category should have after this round
                target_for_round = min(samples_per_round * round_num, MAX_SAMPLES_PER_CATEGORY)
                needed = target_for_round - current_count
                
                if needed <= 0:
                    continue
                
                # Add samples from this category
                added = 0
                for idx in sample_indices:
                    if idx not in downloaded_indices:
                        youtube_id = segments[idx][0]
                        if f'Y{youtube_id}.wav' not in existing_files:
                            download_order.append(idx)
                            downloaded_indices.add(idx)
                            category_downloaded[label] += 1
                            added += 1
                            added_this_round += 1
                            
                            if added >= needed:
                                break
                            if len(download_order) >= MAX_CLIPS_TARGET:
                                break
                
                if len(download_order) >= MAX_CLIPS_TARGET:
                    break
            
            # If no samples added this round, we're done
            if added_this_round == 0:
                logging.info(f"Completed after {round_num} rounds - all categories exhausted")
                break
        
        # Add any remaining high-value samples if under target
        if len(download_order) < MAX_CLIPS_TARGET:
            for idx in range(len(segments)):
                if idx not in downloaded_indices:
                    download_order.append(idx)
                    if len(download_order) >= MAX_CLIPS_TARGET:
                        break
        
        # Reorder segments based on priority
        segments_ordered = [segments[idx] for idx in download_order]
        
        logging.info(f"Planned downloads: {len(segments_ordered):,} clips (estimated {len(segments_ordered) * ESTIMATED_BYTES_PER_CLIP / 1024 / 1024 / 1024:.1f}GB)")
    else:
        segments_ordered = segments
    
    if limit:
        segments_ordered = segments_ordered[:limit]
        logging.info(f"Limiting download to {limit} files for testing")
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Downloading {subset_name} subset")
    logging.info(f"Total segments: {total_count}")
    logging.info(f"Already downloaded: {existing_count}")
    logging.info(f"To download: {len(segments_ordered) - existing_count}")
    logging.info(f"Total categories: {len(category_samples)}")
    logging.info(f"{'='*60}\n")
    
    stats = {
        'total': len(segments_ordered),
        'success': 0,
        'failed': 0,
        'skipped': existing_count,
        'unavailable': 0,
        'categories': len(category_samples),
        'category_coverage': {}  # label_id -> count
    }
    
    start_time_overall = time.time()
    
    # Thread-safe counters
    stats_lock = threading.Lock()
    progress_counter = 0
    
    def download_task(segment_data):
        """Download a single segment (for parallel execution)"""
        i, segment = segment_data
        youtube_id = segment[0]
        start_sec = segment[1]
        end_sec = segment[2]
        labels = segment[3] if len(segment) > 3 else []
        
        output_path = os.path.join(audio_dir, f'Y{youtube_id}.wav')
        
        # Skip if already exists
        if os.path.exists(output_path):
            with stats_lock:
                stats['success'] += 1
                # Update category coverage
                for label in labels:
                    stats['category_coverage'][label] = stats['category_coverage'].get(label, 0) + 1
            return True, labels
        
        # Download
        success = download_audio_segment(youtube_id, start_sec, end_sec, output_path)
        
        with stats_lock:
            if success:
                stats['success'] += 1
                # Update category coverage
                for label in labels:
                    stats['category_coverage'][label] = stats['category_coverage'].get(label, 0) + 1
            else:
                stats['failed'] += 1
                stats['unavailable'] += 1
        
        return success, labels
    
    # Parallel download with thread pool
    logging.info(f"Starting parallel download with {MAX_WORKERS} workers...")
    
    # Create progress bar
    pbar = tqdm(total=len(segments_ordered), 
                desc=f"Downloading {subset_name}",
                unit="clips",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_segment = {
            executor.submit(download_task, (i, segment)): i 
            for i, segment in enumerate(segments_ordered, 1)
        }
        
        # Process completed downloads
        for future in as_completed(future_to_segment):
            i = future_to_segment[future]
            progress_counter += 1
            
            try:
                future.result()
            except Exception as e:
                logging.warning(f"Error processing segment {i}: {e}")
                with stats_lock:
                    stats['failed'] += 1
            
            # Update progress bar
            with stats_lock:
                # Calculate current statistics
                current_size_gb = stats['success'] * ESTIMATED_BYTES_PER_CLIP / 1024 / 1024 / 1024
                categories_with_min = sum(1 for count in stats['category_coverage'].values() if count >= MIN_SAMPLES_PER_CATEGORY)
                
                # Update progress bar with postfix info
                pbar.set_postfix({
                    'OK': stats['success'],
                    'Failed': stats['failed'],
                    'Size': f"{current_size_gb:.1f}/{TARGET_DATASET_SIZE_GB}GB",
                    'Categories': f"{categories_with_min}/{stats['categories']} with {MIN_SAMPLES_PER_CATEGORY}+"
                })
                pbar.update(1)
    
    pbar.close()
    
    # Final stats with category coverage
    elapsed = time.time() - start_time_overall
    categories_with_min = sum(1 for count in stats['category_coverage'].values() if count >= MIN_SAMPLES_PER_CATEGORY)
    categories_with_max = sum(1 for count in stats['category_coverage'].values() if count >= MAX_SAMPLES_PER_CATEGORY)
    categories_under_min = [label for label, count in stats['category_coverage'].items() if count < MIN_SAMPLES_PER_CATEGORY]
    avg_per_category = sum(stats['category_coverage'].values()) / len(stats['category_coverage']) if stats['category_coverage'] else 0
    final_size_gb = stats['success'] * ESTIMATED_BYTES_PER_CLIP / 1024 / 1024 / 1024
    
    logging.info(f"\n{'='*60}")
    logging.info(f"{subset_name} download complete!")
    logging.info(f"Time: {timedelta(seconds=int(elapsed))}")
    logging.info(f"Success: {stats['success']}/{stats['total']}")
    logging.info(f"Failed/Unavailable: {stats['failed']}")
    logging.info(f"Estimated size: {final_size_gb:.1f}GB")
    logging.info(f"\nCategory Coverage:")
    logging.info(f"  Total categories: {stats['categories']}")
    logging.info(f"  Categories with {MIN_SAMPLES_PER_CATEGORY}+ samples: {categories_with_min}/{stats['categories']} ({categories_with_min/stats['categories']*100:.1f}%)")
    logging.info(f"  Categories at max ({MAX_SAMPLES_PER_CATEGORY}): {categories_with_max}")
    logging.info(f"  Categories with 1+ samples: {len(stats['category_coverage'])}")
    logging.info(f"  Average samples per category: {avg_per_category:.1f}")
    
    if categories_under_min and class_labels:
        logging.info(f"\nCategories with < {MIN_SAMPLES_PER_CATEGORY} samples ({len(categories_under_min)}):") 
        for label in sorted(categories_under_min[:20]):  # Show first 20
            label_name = class_labels.get(label, label)
            count = stats['category_coverage'].get(label, 0)
            logging.info(f"  {label_name}: {count} samples")
        if len(categories_under_min) > 20:
            logging.info(f"  ... and {len(categories_under_min) - 20} more")
    
    logging.info(f"{'='*60}\n")
    
    return stats

def main():
    """Main download function"""
    # Check arguments
    output_dir = 'audioset_16k'
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    
    test_mode = '--test' in sys.argv
    
    print("\n" + "="*70)
    print("AudioSet Balanced + Eval Subset Downloader")
    print("="*70)
    print("\nThis will download audio clips from YouTube using balanced sampling")
    print(f"Target: ~{TARGET_DATASET_SIZE_GB}GB (~{MAX_CLIPS_TARGET:,} clips)")
    print(f"Strategy: {MIN_SAMPLES_PER_CATEGORY}-{MAX_SAMPLES_PER_CATEGORY} samples per category across all 527 categories")
    print("Expected time: 6-24 hours (depends on internet speed)")
    print("\nNote: Many videos may be unavailable or region-locked")
    print("Typical success rate: 70-90%")
    print("="*70 + "\n")
    
    if test_mode:
        print("[TEST MODE] Will download only 10 samples from each subset\n")
        limit = 10
    else:
        limit = None
    
    # Check dependencies
    print("Checking dependencies...")
    deps = check_dependencies()
    
    if not deps['yt-dlp']:
        print("\n[!] yt-dlp not found. Installing...")
        if not install_ytdlp():
            print("\n[ERROR] Failed to install yt-dlp")
            print("Please install manually: pip install -U yt-dlp")
            return 1
        # Check again
        deps = check_dependencies()
        if not deps['yt-dlp']:
            print("\n[ERROR] yt-dlp still not available after installation")
            return 1
    
    if not deps['ffmpeg']:
        print("\n[WARNING] ffmpeg not found!")
        print("AudioSet download requires ffmpeg for audio extraction")
        print("\nInstall ffmpeg:")
        print("  Windows: Download from https://www.gyan.dev/ffmpeg/builds/")
        print("           Extract and add to PATH")
        print("  Linux: sudo apt install ffmpeg")
        print("  Mac: brew install ffmpeg")
        return 1
    
    print("[OK] All dependencies available\n")
    
    # Download metadata
    metadata_dir = download_metadata(output_dir)
    if not metadata_dir:
        print("\n[ERROR] Failed to download metadata")
        return 1
    
    print("[OK] Metadata downloaded\n")
    
    # Load class labels for category tracking
    class_labels = load_class_labels(metadata_dir)
    
    # Download audio files
    overall_stats = {}
    
    for subset_name in ['balanced_train', 'eval']:
        csv_path = os.path.join(metadata_dir, f'{subset_name}.csv')
        stats = download_from_csv(csv_path, output_dir, subset_name, class_labels=class_labels,
                                  limit=limit, ensure_min_per_category=(not test_mode))
        overall_stats[subset_name] = stats
    
    # Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    total_success = sum(s['success'] for s in overall_stats.values())
    total_attempted = sum(s['total'] for s in overall_stats.values())
    success_rate = (total_success / total_attempted * 100) if total_attempted > 0 else 0
    
    # Combine category coverage from all subsets
    all_categories = set()
    combined_coverage = {}
    for stats in overall_stats.values():
        all_categories.update(stats.get('category_coverage', {}).keys())
        for label, count in stats.get('category_coverage', {}).items():
            combined_coverage[label] = combined_coverage.get(label, 0) + count
    
    categories_with_min = sum(1 for count in combined_coverage.values() if count >= MIN_SAMPLES_PER_CATEGORY)
    categories_with_max = sum(1 for count in combined_coverage.values() if count >= MAX_SAMPLES_PER_CATEGORY)
    avg_per_category = sum(combined_coverage.values()) / len(combined_coverage) if combined_coverage else 0
    total_size_gb = total_success * ESTIMATED_BYTES_PER_CLIP / 1024 / 1024 / 1024
    
    for subset_name, stats in overall_stats.items():
        print(f"\n{subset_name}:")
        print(f"  Downloaded: {stats['success']}/{stats['total']} ({stats['success']/stats['total']*100:.1f}%)")
        print(f"  Failed: {stats['failed']}")
    
    print(f"\nOverall: {total_success}/{total_attempted} ({success_rate:.1f}%)")
    print(f"Total size: ~{total_size_gb:.1f}GB")
    print(f"\nCategory Coverage:")
    print(f"  Total categories in AudioSet: 527")
    print(f"  Categories with any samples: {len(all_categories)} ({len(all_categories)/527*100:.1f}%)")
    print(f"  Categories with {MIN_SAMPLES_PER_CATEGORY}+ samples: {categories_with_min} ({categories_with_min/527*100:.1f}%)")
    print(f"  Categories at max ({MAX_SAMPLES_PER_CATEGORY}): {categories_with_max}")
    print(f"  Average samples per category: {avg_per_category:.1f}")
    
    print(f"\nAudio files saved to: {os.path.abspath(output_dir)}")
    print(f"Log file: audioset_download.log")
    print("="*70 + "\n")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
