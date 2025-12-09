import numpy as np
from scapy.all import rdpcap
from scapy.layers.dot11 import Dot11
import os
import sys
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from scipy.signal import medfilt

FPS = 20

STD_MIN_DURATION_S = 2.0    # Minimum event duration to be considered valid
GAP_TOLERANCE_S = 1.0       # Max time allowed between spikes before event ends
IGNORE_START_S = 10.0       # Ignore the first 10 seconds to avoid startup noise
MIN_BYTES_FLOOR = 100       # Minimum byte threshold to prevent div/0 or noise floor errors

def pcap_to_frame_sizes(pcap_path):
    print(f"Loading: {pcap_path}...")
    try:
        packets = rdpcap(str(pcap_path))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
        
    start_time = 0
    first_idx = 0
    for idx, pkt in enumerate(packets):
        if pkt.haslayer(Dot11) and pkt[Dot11].type == 2 and len(pkt) > 1000:
            start_time = float(pkt.time)
            first_idx = idx
            break
            
    packets = packets[first_idx:]
    frame_dur = 1.0 / FPS
    sizes = []
    curr_frame_idx = 0
    curr_acc = 0
    
    for pkt in tqdm(packets, desc="Parsing Packets", leave=False):
        if pkt.haslayer(Dot11) and pkt[Dot11].type == 2:
            ts = float(pkt.time)
            f_idx = int((ts - start_time) / frame_dur)
            if f_idx > curr_frame_idx:
                for _ in range(f_idx - curr_frame_idx):
                    sizes.append(curr_acc)
                    curr_acc = 0
                curr_frame_idx = f_idx
                curr_acc += len(pkt)
            else:
                curr_acc += len(pkt)       
    sizes.append(curr_acc)
    return np.array(sizes)

def calibrate_noise_profile(signal, sigma_clip=3.0):
    """
    Calculates the noise floor statistics (mean, std, max).
    Iteratively removes outliers (signal spikes) to find the true baseline noise.
    """
    noise_samples = signal.copy()
    for _ in range(3):
        median = np.median(noise_samples)
        std = np.std(noise_samples)
        cutoff = median + (sigma_clip * std) 
        noise_samples = noise_samples[noise_samples < cutoff]
        
    if len(noise_samples) == 0: 
        return np.median(signal), 100, 100 
        
    return np.mean(noise_samples), np.std(noise_samples), np.max(noise_samples)

def analyze_motion(raw_sizes, use_high_sensitivity=False):
    """
    Detecting motion events based on bitrate variance with filtering 
    and dynamic thresholding. 
    """
    if len(raw_sizes) == 0:
        return [], [], raw_sizes, raw_sizes, [], [], 0

    # Median filter to suppress I-frame spikes
    clean_signal = medfilt(raw_sizes, kernel_size=5)

    # Select sensitivity
    if not use_high_sensitivity:
        # Standard: For cameras with distinct signal-to-noise ratios
        CLIP_SIGMA = 3.0      
        TRIGGER_BUFFER = 2.0  
        SUSTAIN_RATIO = 0.4   
        FORCE_WINDOW = None   
        MIN_DURATION_S = STD_MIN_DURATION_S 
    else:
        # High Sensitivity: For weak signals or high background noise
        # Uses tighter trigger buffer and longer duration requirement to filter false positives
        CLIP_SIGMA = 2.0      
        TRIGGER_BUFFER = 0.5  
        SUSTAIN_RATIO = 0.2   
        FORCE_WINDOW = 10     
        MIN_DURATION_S = 4.0

    # Determine noise floor excluding the startup period
    start_idx = int(IGNORE_START_S * FPS)
    calib_slice = clean_signal[start_idx:] if len(clean_signal) > start_idx else clean_signal
    
    noise_mean, noise_std, noise_max = calibrate_noise_profile(calib_slice, sigma_clip=CLIP_SIGMA)
    
    # Calculate window size based on signal variance 
    if FORCE_WINDOW:
        smooth_window = FORCE_WINDOW
    else:
        cv = noise_std / (noise_mean + 1e-5) 
        if cv > 0.5: smooth_window = 40 
        elif cv > 0.2: smooth_window = 20 
        else: smooth_window = 10          

    smoothed = np.convolve(clean_signal, np.ones(smooth_window)/smooth_window, mode='same')
    
    # Threshold Calculation
    calib_smooth = smoothed[start_idx:] if len(smoothed) > start_idx else smoothed
    s_mean, s_std, s_max = calibrate_noise_profile(calib_smooth, sigma_clip=CLIP_SIGMA)
    
    thresh_high = max(s_max + (TRIGGER_BUFFER * s_std), MIN_BYTES_FLOOR)
    thresh_low = s_mean + (thresh_high - s_mean) * SUSTAIN_RATIO
    thresh_low = max(thresh_low, MIN_BYTES_FLOOR)

    high_thresholds = [thresh_high] * len(smoothed)
    low_thresholds = [thresh_low] * len(smoothed)

    valid_events = []
    rejected_events = [] 
    
    is_active = False
    start_frame = 0
    gap_counter = 0
    
    gap_limit_frames = int(GAP_TOLERANCE_S * FPS)
    min_frames = int(MIN_DURATION_S * FPS)
    
    for i, val in enumerate(smoothed):
        if i < start_idx: continue

        if not is_active:
            # Trigger condition
            if val > thresh_high:
                is_active = True
                start_frame = i
                gap_counter = 0
        else:
            # Sustain condition
            if val > thresh_low:
                gap_counter = 0
            else:
                gap_counter += 1
                # End of event detected
                if gap_counter >= gap_limit_frames:
                    end_frame = i - gap_limit_frames
                    duration_frames = end_frame - start_frame
                    
                    # Check event average is above sustain floor
                    event_avg = np.mean(smoothed[start_frame:end_frame])
                    is_valid_quality = True
                    if use_high_sensitivity and event_avg < thresh_low:
                        is_valid_quality = False

                    if duration_frames >= min_frames and is_valid_quality:
                        valid_events.append((start_frame, end_frame))
                    else:
                        rejected_events.append((start_frame, end_frame, duration_frames/FPS))
                    
                    is_active = False
                    gap_counter = 0
            
    if is_active:
        end_frame = len(smoothed)
        duration_frames = end_frame - start_frame
        
        event_avg = np.mean(smoothed[start_frame:end_frame])
        is_valid_quality = True
        if use_high_sensitivity and event_avg < thresh_low:
            is_valid_quality = False
            
        if duration_frames >= min_frames and is_valid_quality:
             valid_events.append((start_frame, end_frame))
                
    return valid_events, rejected_events, clean_signal, smoothed, np.array(high_thresholds), np.array(low_thresholds), smooth_window

def save_events(events, rejected, clean_signal, smoothed, thresh_high, thresh_low, raw_sizes, out_dir, camera_id, window_used, is_high_sens):
    events_file = f"camera_{camera_id}_events.json"
    events_dir = Path(os.path.join(out_dir, "events"))
    events_dir.mkdir(parents=True, exist_ok=True)
    events_path = Path(os.path.join(events_dir, events_file))

    plot_file = f"camera_{camera_id}_plot.png"
    plot_dir = Path(os.path.join(out_dir, "plots"))
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = os.path.join(plot_dir, plot_file)

    mode_label = "HIGH SENSITIVITY" if is_high_sens else "STANDARD"
    
    print("\n" + "="*50)
    print(f"CAMERA {camera_id}: {len(events)} EVENTS DETECTED")
    print(f"Mode: {mode_label} | Smoothing Window: {window_used}")
    print("="*50)
    print(f"{'#':<5} {'START (s)':<12} {'END (s)':<12} {'DURATION':<10}")
    print("-" * 50)

    event_list = []
    for i, (start, end) in enumerate(events):
        s_time = (start / FPS) + 1.0 
        e_time = (end / FPS) + 1.0
        dur = e_time - s_time
        
        start_fmt = f"{s_time:.2f}"
        if s_time > 60:
            m = int(s_time // 60)
            s = s_time % 60
            start_fmt += f" ({m}:{s:04.1f})"
            
        print(f"{i+1:<5} {start_fmt:<12} {e_time:<12.2f} {dur:<10.2f}s")
        event_list.append({"event": i+1, "start_s": s_time, "end_s": e_time, "duration": dur})

    print("="*50)

    with open(events_path, "w") as f:
        json.dump(event_list, f, indent=2)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(raw_sizes, color='lightgray', label='Raw Frame Sizes')
    plt.title(f"Camera {camera_id} - Raw Data")
    
    plt.subplot(2, 1, 2)
    plt.plot(clean_signal, color='lightblue', alpha=0.5, label='Filtered')
    plt.plot(smoothed, color='darkblue', linewidth=2, label=f'Smoothed (W={window_used})')
    
    if len(thresh_high) == len(smoothed):
        color = 'magenta' if is_high_sens else 'red'
        plt.plot(thresh_high, color=color, linestyle='--', label='Trigger Threshold')
        plt.plot(thresh_low, color='orange', linestyle=':', label='Sustain Threshold')

    for (start, end) in events:
        plt.axvspan(start, end, color='green', alpha=0.3, label='Event')
    
    if not is_high_sens:
        for (start, end, _) in rejected:
            plt.axvspan(start, end, color='red', alpha=0.1) 

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.title(f"Cam {camera_id} Analysis | {mode_label}")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pcap_file", help="Path to pcap file or folder")
    parser.add_argument("-o", "--output_dir", type=str)
    args = parser.parse_args()
    
    path = Path(args.pcap_file)
    files = list(path.glob("*.pcap")) if path.is_dir() else [path]
    out_dir = Path(args.output_dir) if args.output_dir else (path if path.is_dir() else path.parent)

    for f in files:
        try: cam_id = f.stem.split("_")[1].split(".")[0]
        except: cam_id = f.stem
        
        raw_sizes = pcap_to_frame_sizes(str(f))
        
        # Standard Mode
        events, rejected, clean, smoothed, high, low, win = analyze_motion(raw_sizes, use_high_sensitivity=False)
        is_high_sens = False
        
        # Fallback: If no events found, use High Sensitivity Mode except it doesn't really work rn
        if len(events) == 0:
            events_r, rejected_r, clean_r, smoothed_r, high_r, low_r, win_r = analyze_motion(raw_sizes, use_high_sensitivity=True)
            
            if len(events_r) > 0:
                events, rejected, clean, smoothed, high, low, win = events_r, rejected_r, clean_r, smoothed_r, high_r, low_r, win_r
                is_high_sens = True

        save_events(events, rejected, clean, smoothed, high, low, raw_sizes, out_dir, cam_id, win, is_high_sens)

if __name__ == "__main__":
    main()