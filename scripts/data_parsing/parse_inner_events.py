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

STD_MIN_DURATION_S = 2.0    # Events shorter than this are treated as noise/glitches
GAP_TOLERANCE_S = 1.0       # How long to wait for signal to return before ending an event
IGNORE_START_S = 10.0       # Ignore the beginning 10 seconds
MIN_BYTES_FLOOR = 100       # Minimum byte count to consider a signal 

CAMERA_OVERRIDES = {
    "18": {
        "TRIGGER_BUFFER": 0.6,   # Multiplier for Std Dev to set trigger threshold
        "MIN_DURATION_S": 5.0,   # Longer duration
        "FORCE_WINDOW": 20,      # Smoothing window size
        "SUSTAIN_RATIO": 0.3,    # How far the signal can drop before event ends
        "CLIP_SIGMA": 2.5        # Outlier rejection strictness
    },
    
    # CASE B: 
    "19": {
        "TRIGGER_BUFFER": 2.0,   
        "MIN_DURATION_S": 4.0,   
        "FORCE_WINDOW": 20,      
        "SUSTAIN_RATIO": 0.4,    
        "CLIP_SIGMA": 3.0        
    },

    "20": {
        "TRIGGER_BUFFER": 1.0,   
        "MIN_DURATION_S": 4.0,   
        "FORCE_WINDOW": 20,      
        "SUSTAIN_RATIO": 0.2,    
        "CLIP_SIGMA": 2.0        
    },

    # Fails for all, might remove
    "9": {
        "TRIGGER_BUFFER": 1.8,   
        "MIN_DURATION_S": 4.0,   
        "FORCE_WINDOW": 20,      
        "SUSTAIN_RATIO": 0.2,    
        "CLIP_SIGMA": 2.0        
    }
}

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
    Calculates the background noise statistics (Mean, Std, Max).
    Iteratively removes outliers so they don't skew the noise calculation.
    """
    noise_samples = signal.copy()
    
    # Run 3 passes of clipping to remove high-value spikes 
    for _ in range(3):
        median = np.median(noise_samples)
        std = np.std(noise_samples)
        cutoff = median + (sigma_clip * std) 
        noise_samples = noise_samples[noise_samples < cutoff]
        
    # Safety check if signal was empty
    if len(noise_samples) == 0: 
        return np.median(signal), 100, 100 
        
    return np.mean(noise_samples), np.std(noise_samples), np.max(noise_samples)

def analyze_motion(raw_sizes, use_high_sensitivity=False, override_config=None):
    """
    Core detection algorithm. 
    1. Filters signal.
    2. Calculates dynamic thresholds based on noise.
    3. Iterates through signal to find events.
    """
    if len(raw_sizes) == 0:
        return [], [], raw_sizes, raw_sizes, [], [], 0, 0, 0

    # Median Filter to remove single-frame spikes (glitches)
    clean_signal = medfilt(raw_sizes, kernel_size=5)

    settings = {
        "CLIP_SIGMA": 3.0,
        "TRIGGER_BUFFER": 2.0,       # High threshold = Noise Max + (2.0 * Std)
        "SUSTAIN_RATIO": 0.4,        # Low threshold location (percentage between mean and high thresh)
        "FORCE_WINDOW": None,
        "MIN_DURATION_S": STD_MIN_DURATION_S
    }

    # High sensitivity for fallback
    if use_high_sensitivity:
        settings.update({
            "CLIP_SIGMA": 2.0,       # Clip tighter to find smaller noise floor
            "TRIGGER_BUFFER": 1.0,   # Lower trigger threshold
            "SUSTAIN_RATIO": 0.2,
            "FORCE_WINDOW": 20,      
            "MIN_DURATION_S": 4.0
        })

    # If a specific Camera Override exists, it overwrites everything else
    if override_config:
        settings.update(override_config)

    # Unpack settings for easier access
    CLIP_SIGMA = settings["CLIP_SIGMA"]
    TRIGGER_BUFFER = settings["TRIGGER_BUFFER"]
    SUSTAIN_RATIO = settings["SUSTAIN_RATIO"]
    FORCE_WINDOW = settings["FORCE_WINDOW"]
    MIN_DURATION_S = settings["MIN_DURATION_S"]

    #  Determine Smoothing Window
    #  Analyze the first few seconds (ignoring start) to check signal volatility 
    start_idx = int(IGNORE_START_S * FPS)
    calib_slice = clean_signal[start_idx:] if len(clean_signal) > start_idx else clean_signal
    
    noise_mean, noise_std, noise_max = calibrate_noise_profile(calib_slice, sigma_clip=CLIP_SIGMA)
    
    if FORCE_WINDOW:
        smooth_window = FORCE_WINDOW
    else:
        # Coefficient of Variation determines how messy the signal is.
        # High CV -> Needs more smoothing (Window=40). Low CV -> Less smoothing.
        cv = noise_std / (noise_mean + 1e-5) 
        if cv > 0.5: smooth_window = 40 
        elif cv > 0.2: smooth_window = 20 
        else: smooth_window = 10          

    # Apply smoothing convolution
    smoothed = np.convolve(clean_signal, np.ones(smooth_window)/smooth_window, mode='same')
    
    # Recalibrate noise on the SMOOTHED signal for accurate thresholds
    calib_smooth = smoothed[start_idx:] if len(smoothed) > start_idx else smoothed
    s_mean, s_std, s_max = calibrate_noise_profile(calib_smooth, sigma_clip=CLIP_SIGMA)
    
    # Calculate Dual Thresholds (Schmitt Trigger)
    thresh_high = max(s_max + (TRIGGER_BUFFER * s_std), MIN_BYTES_FLOOR)
    
    # thresh_low is calculated as a percentage distance between the mean and the high threshold
    thresh_low = s_mean + (thresh_high - s_mean) * SUSTAIN_RATIO
    thresh_low = max(thresh_low, MIN_BYTES_FLOOR)

    # Store thresholds for plotting later
    high_thresholds = [thresh_high] * len(smoothed)
    low_thresholds = [thresh_low] * len(smoothed)

    # vent Detection Loop
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
            # TSignal exceeds High Threshold
            if val > thresh_high:
                is_active = True
                start_frame = i
                gap_counter = 0
        else:
            # Signal stays above Low Threshold
            if val > thresh_low:
                gap_counter = 0
            else:
                gap_counter += 1
                
                # If gap persists too long, the event ends
                if gap_counter >= gap_limit_frames:
                    end_frame = i - gap_limit_frames
                    duration_frames = end_frame - start_frame
                    
                    # Average bytes during event must allow threshold
                    event_avg = np.mean(smoothed[start_frame:end_frame])
                    is_valid_quality = True
                    if (use_high_sensitivity or override_config) and event_avg < thresh_low:
                        is_valid_quality = False

                    # Must be longer than minimum duration
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
        if (use_high_sensitivity or override_config) and event_avg < thresh_low:
            is_valid_quality = False
            
        if duration_frames >= min_frames and is_valid_quality:
             valid_events.append((start_frame, end_frame))
                
    return valid_events, rejected_events, clean_signal, smoothed, np.array(high_thresholds), np.array(low_thresholds), smooth_window, noise_mean, noise_std

def save_events(events, rejected, clean_signal, smoothed, thresh_high, thresh_low, raw_sizes, out_dir, camera_id, window_used, mode_label):

    events_file = f"camera_{camera_id}_events.json"
    events_dir = Path(os.path.join(out_dir, "events"))
    events_dir.mkdir(parents=True, exist_ok=True)
    events_path = Path(os.path.join(events_dir, events_file))

    plot_file = f"camera_{camera_id}_plot.png"
    plot_dir = Path(os.path.join(out_dir, "plots"))
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = os.path.join(plot_dir, plot_file)

    print("\n" + "="*50)
    print(f"CAMERA {camera_id}: {len(events)} EVENTS DETECTED")
    print(f"Mode: {mode_label} | Smoothing Window: {window_used}")
    print("="*50)
    print(f"{'#':<5} {'START (s)':<12} {'END (s)':<12} {'DURATION':<10}")
    print("-" * 50)

    event_list = []
    for i, (start, end) in enumerate(events):
        s_time = (start / FPS) + 1.0 # +1.0 offset, depends on sync but haven't messed with it
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
        color = 'magenta' if "SENSITIVITY" in mode_label or "OVERRIDE" in mode_label else 'red'
        plt.plot(thresh_high, color=color, linestyle='--', label='Trigger Threshold')
        plt.plot(thresh_low, color='orange', linestyle=':', label='Sustain Threshold')

    # Highlight Detected Events in Green
    for (start, end) in events:
        plt.axvspan(start, end, color='green', alpha=0.3, label='Event')
    
    # Highlight Rejected Events in Red 
    if "STANDARD" in mode_label:
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
        if cam_id in CAMERA_OVERRIDES:
            print(f"\nDEBUG: Camera {cam_id} found in OVERRIDES. Applying custom tuning.")
            config = CAMERA_OVERRIDES[cam_id]
            events, rejected, clean, smoothed, high, low, win, _, _ = analyze_motion(
                raw_sizes, override_config=config
            )
            mode_used = f"OVERRIDE ({cam_id})"
            
        else:
            events, rejected, clean, smoothed, high, low, win, n_mean, n_std = analyze_motion(raw_sizes, use_high_sensitivity=False)
            mode_used = "STANDARD"
            
            should_use_high_sens = False
            
            # No events found -> Try High Sensitivity
            if len(events) == 0:
                should_use_high_sens = True
                print(f"DEBUG: Camera {cam_id} | 0 events found. Trying High Sensitivity.")
                
            # High Noise Floor -> Try High Sensitivity
            # If background noise > 2000 bytes, standard thresholds might be too high
            elif n_mean > 2000 and n_std > 500:
                should_use_high_sens = True
                print(f"DEBUG: Camera {cam_id} | High Noise Detected (Mean:{n_mean:.0f}, Std:{n_std:.0f}). Forcing High Sensitivity.")

            if should_use_high_sens:
                events_r, rejected_r, clean_r, smoothed_r, high_r, low_r, win_r, _, _ = analyze_motion(raw_sizes, use_high_sensitivity=True)
                
                # Only keep high sensitivity results if it actually found something
                if len(events_r) > 0:
                    events, rejected, clean, smoothed, high, low, win = events_r, rejected_r, clean_r, smoothed_r, high_r, low_r, win_r
                    mode_used = "HIGH SENSITIVITY"
                elif len(events) == 0:
                     pass
                else:
                     print(f"DEBUG: High Sensitivity yielded 0 events vs Standard's {len(events)}. Keeping Standard.")

        save_events(events, rejected, clean, smoothed, high, low, raw_sizes, out_dir, cam_id, win, mode_used)

if __name__ == "__main__":
    main()