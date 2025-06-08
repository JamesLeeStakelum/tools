r"""
# Voice Input Transcription with Vosk

This Python script provides a live, real-time speech-to-text transcription service using the Vosk speech recognition toolkit. It leverages a progressive enhancement approach, starting with a smaller, faster model for immediate feedback and seamlessly switching to a larger, more accurate model in the background once loaded.

## Features

  * Real-time Transcription: Converts spoken audio from your microphone into text as you speak.
  * Progressive Accuracy: Initiates with a fast, lightweight Vosk model for quick startup and transitions to a more accurate, larger model for improved transcription quality.
  * Offline Operation: Vosk models run locally on your machine, ensuring privacy and eliminating the need for an internet connection during transcription.
  * Hotkeys: Use Ctrl+Alt+Space to gracefully stop the transcription process.
  * Session Management: Transcripts are saved into timestamped session folders, with individual utterances and a complete session transcript.
  * Intelligent Display: Filters common filler words and intelligently updates partial results on the console for a cleaner user experience.

## Requirements

  * Python 3.x
  * sounddevice
  * vosk
  * pynput
  * colorama

You can install these dependencies using pip:
pip install sounddevice vosk pynput colorama

## Vosk Models

This script utilizes two Vosk models for its progressive enhancement feature:

1.  Small Model (Fast): vosk-model-small-en-us-0.15
2.  Large Model (High Accuracy): vosk-model-en-us-0.22

Downloading Vosk Models:

You must download these Vosk models and place them in the same directory as this Python script (voice\_input\_vosk.py) or specify their paths using command-line arguments or the voice\_config.txt file.

You can download the models from the official Vosk website:

  * Vosk Models Page: [https://alphacephei.com/vosk/models](https://alphacephei.com/vosk/models)

    Navigate to this page to find a list of available models. For this script, look for the English models:

      * vosk-model-small-en-us-0.15
      * vosk-model-en-us-0.22 (or a newer, similarly sized high-accuracy English model if available)

Extraction:

After downloading, the models will be in compressed archives (e.g., .zip or .tar.gz). You must unzip/extract these archives into their respective folders. For example, vosk-model-small-en-us-0.15.zip should be extracted to a folder named vosk-model-small-en-us-0.15.

Placement:

Ensure the extracted model directories (e.g., vosk-model-small-en-us-0.15/ and vosk-model-en-us-0.22/) are located in the same folder as your voice\_input\_vosk.py script.

## Configuration (Optional)

You can create a voice\_config.txt file in the same directory as the script to customize settings. Here's an example:

# voice\_config.txt

device\_index = 1 \# Your audio input device index
small\_model\_path = vosk-model-small-en-us-0.15
large\_model\_path = vosk-model-en-us-0.22
output\_dir = transcripts

## Usage

To run the script, simply execute it from your terminal:

python voice\_input\_vosk.py

### Command-line Arguments

You can override configuration settings directly from the command line:

  * \--config \<path\>: Specify a different configuration file.
  * \--device-index \<index\>: Manually set the audio input device index. Run the script without this argument first to see a list of available devices.
  * \--small-model-path \<path\>: Specify the path to the small Vosk model.
  * \--large-model-path \<path\>: Specify the path to the large Vosk model.
  * \--output-dir \<path\>: Specify the directory to save session transcripts.

Example with arguments:

python voice\_input\_vosk.py --device-index 0 --output-dir my\_transcripts

The script will guide you through selecting an audio input device if device\_index is not specified.
"""
import os
import sys
import time
import queue
import threading
import itertools
from datetime import datetime
import argparse
import json

import sounddevice as sd
import vosk
from pynput import keyboard

import colorama
colorama.init()

# --- Configuration ---
CONFIG_FILE = "voice_config.txt"
CONFIG_DEFAULTS = {
    'device_index': None,
    'small_model_path': 'vosk-model-small-en-us-0.15',
    'large_model_path': 'vosk-model-en-us-0.22',
    'output_dir': None
}

# --- Audio Settings ---
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = 'int16'
BLOCK_SIZE = 400
LEADING_FILLER_WORDS = {"um", "uh"}
PAUSE_THRESHOLD_SECONDS = 1.0 # Softened from 1.5

# --- Display Enhancement Settings ---
COMMON_FILL_DISPLAY_SUPPRESS = {
    "the", "a", "an", "i", "and", "oh", "um", "uh", "mhm", "hmm", "yeah", "okay",
    "you", "so", "well", "like", "just", "it's", "is", "of", "to", "in", "for",
    "that", "no", "yes", "on", "he", "she", "we", "they", "was", "are", "do", "don't"
}
SILENCE_THRESHOLD_SECONDS = 0.5 # Softened from 0.75
PARTIAL_UPDATE_THROTTLE_SECONDS = 0.08 # Softer throttle: approx 12.5 Hz
MIN_PARTIAL_LENGTH_FOR_DISPLAY = 2 # Allowing shorter meaningful partials


# --- Globals (Managed by the application, shared across functions) ---
_audio_queue = queue.Queue()
_finish_event = threading.Event()

_vosk_model_small = None
_vosk_recognizer_small = None

_vosk_model_large = None
_vosk_recognizer_large = None

_current_recognizer = None

_large_model_loaded_event = threading.Event()


# --- Hotkey Callbacks ---
def _on_finish_hotkey():
    print("\nFinish hotkey pressed. Finalizing transcript...", file=sys.stderr)
    _finish_event.set()


def _start_hotkey_listener():
    try:
        listener = keyboard.GlobalHotKeys({'<ctrl>+<alt>+<space>': _on_finish_hotkey})
        listener.daemon = True
        listener.start()
    except Exception as e:
        print(f"Warning: Hotkey disabled (requires admin/root or specific OS setup): {e}", file=sys.stderr)


# --- Text Cleaning Functions ---
def _clean_text(text: str) -> str:
    words = text.split()
    while words and words[0].lower() in LEADING_FILLER_WORDS:
        words.pop(0)
    return ' '.join(words)


def _clean_partial_for_display(partial_text: str) -> str:
    words = partial_text.strip().split()
    lower_words = [w.lower() for w in words]

    if len(lower_words) == 1 and lower_words[0] in COMMON_FILL_DISPLAY_SUPPRESS:
        return ""
    if 1 < len(lower_words) <= 3:
        if all(w in COMMON_FILL_DISPLAY_SUPPRESS for w in lower_words):
            return ""
    if len(lower_words) == 2:
        if (lower_words[0] == "the" and lower_words[1] in ["is", "it", "to", "in", "of", "and", "a", "i"]): return ""
        if (lower_words[0] == "i" and lower_words[1] in ["am", "think", "know"]): return ""
        if (lower_words[0] == "you" and lower_words[1] in ["are", "know"]): return ""
    if len(partial_text.strip()) <= 5 and all(w in COMMON_FILL_DISPLAY_SUPPRESS for w in lower_words):
        return ""
        
    return partial_text


# --- Audio Callback Function ---
def _audio_callback(indata, frames, time_info, status):
    if status:
        pass
    _audio_queue.put(bytes(indata))


# --- Utility Functions ---
def list_audio_devices():
    print("\nAvailable Audio Input Devices:")
    try:
        for idx, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] > 0:
                print(f"  [{idx}] {dev['name']}")
    except Exception as e:
        print(f"Error listing audio devices: {e}", file=sys.stderr)


def _load_large_model_in_background(model_path: str, sample_rate: int):
    global _vosk_model_large, _vosk_recognizer_large
    print("\nLoading high-accuracy model in background (this may take a moment)...", file=sys.stderr)
    try:
        _vosk_model_large = vosk.Model(model_path)
        _vosk_recognizer_large = vosk.KaldiRecognizer(_vosk_model_large, sample_rate)
        _vosk_recognizer_large.SetWords(True)
        _large_model_loaded_event.set()
        print("High-accuracy model loaded. Will switch at next natural break.", file=sys.stderr)
    except Exception as e:
        print(f"Error loading high-accuracy model in background: {e}", file=sys.stderr)


# --- Main Application Logic ---
# --- Main Application Logic ---
def main():
    parser = argparse.ArgumentParser(
        description="Live mic transcription using Vosk with progressive enhancement.\n"
                    "Starts fast, then switches to higher accuracy in the background.\n"
                    "Output wraps naturally. Ctrl+Alt+Space to finish.")
    parser.add_argument("--config", default=CONFIG_FILE, help="Path to the configuration file.")
    parser.add_argument("--device-index", type=int, help="Specify audio input device index.")
    parser.add_argument("--small-model-path", help="Path to the small Vosk model directory.")
    parser.add_argument("--large-model-path", help="Path to the large Vosk model directory (high accuracy).")
    parser.add_argument("--output-dir", help="Directory to save session transcripts.")
    args = parser.parse_args()

    # --- Load Configuration from file or create it ---
    cfg = CONFIG_DEFAULTS.copy()
    config_file_path = args.config

    if os.path.exists(config_file_path):
        # Config file exists, load settings from it
        with open(config_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    k, v = line.split('=', 1)
                    # Convert 'None' string to actual None type for relevant keys
                    if v.strip().lower() == 'none':
                        cfg[k.strip()] = None
                    else:
                        cfg[k.strip()] = v.strip()
        try:
            cfg['device_index'] = int(cfg.get('device_index'))
        except (ValueError, TypeError):
            cfg['device_index'] = None # Ensure it's None if parsing fails
        print(f"Configuration loaded from '{config_file_path}'.")
    else:
        # Config file does not exist, prompt for device and create file
        print(f"Configuration file '{config_file_path}' not found.", file=sys.stderr)
        list_audio_devices()
        dev_idx_input = None
        while dev_idx_input is None:
            try:
                dev_idx_input = int(input("Select audio input device index: "))
                # Validate input: basic check if device index is plausible
                # You might want to add more robust validation here, e.g., checking against sd.query_devices()
                if dev_idx_input < 0: # Basic validation
                    raise ValueError
            except ValueError:
                print("Invalid input. Please enter a number.", file=sys.stderr)
                dev_idx_input = None # Reset to None to re-prompt

        cfg['device_index'] = dev_idx_input
        print(f"Device index {dev_idx_input} selected and will be saved to config.")

        # Create the config file with the chosen device index and other defaults
        try:
            with open(config_file_path, 'w', encoding='utf-8') as f:
                f.write("# This file contains configuration settings for the voice transcription script.\n")
                f.write("# Lines starting with '#' are comments and are ignored.\n")
                f.write("# Set 'device_index' to your preferred audio input device index.\n")
                f.write("# Specify the paths to your Vosk model directories (must be extracted).\n")
                f.write("# Set 'output_dir' to the desired directory for saving transcripts.\n\n")
                
                for key, value in CONFIG_DEFAULTS.items():
                    if key == 'device_index': # Use the chosen device index
                        f.write(f"device_index = {cfg['device_index']}\n")
                    elif value is None: # Explicitly write 'None' for None values
                        f.write(f"{key} = None\n")
                    else: # For string paths
                        f.write(f"{key} = {value}\n")
            print(f"Default config file created at '{config_file_path}'.", file=sys.stderr)
        except IOError as e:
            print(f"Error creating config file '{config_file_path}': {e}", file=sys.stderr)
            sys.exit(1) # Exit if cannot create config file

    # Override with command-line arguments if provided
    dev_idx = args.device_index if args.device_index is not None else cfg['device_index']
    small_model_path = args.small_model_path or cfg['small_model_path']
    large_model_path = args.large_model_path or cfg['large_model_path']
    # Ensure output_dir is an actual None if not provided via args and 'None' in config
    output_dir = args.output_dir if args.output_dir is not None else cfg['output_dir']

    # --- Validate model paths ---
    if not os.path.isdir(small_model_path):
        print(f"Error: Small Vosk model not found at '{small_model_path}'. Please download and extract it.", file=sys.stderr)
        sys.exit(1)
    if large_model_path and not os.path.isdir(large_model_path):
        print(f"Warning: High-accuracy Vosk model not found at '{large_model_path}'. "
              "Please download and extract it. Application will only use the fast model.", file=sys.stderr)
        large_model_path = None

    # --- Step 1: Load SMALL model FIRST ---
    spinner_active = True
    def spinner_small_model_load():
        for c in itertools.cycle('|/-\\'):
            if not spinner_active: break
            sys.stderr.write(f"\rLoading fast model... {c}")
            sys.stderr.flush()
            time.sleep(0.1)
        sys.stderr.write("\r" + " "*20 + "\r")
        sys.stderr.flush()

    th_small_spinner = threading.Thread(target=spinner_small_model_load, daemon=True)
    th_small_spinner.start()

    global _vosk_model_small, _vosk_recognizer_small, _current_recognizer
    _vosk_model_small = vosk.Model(small_model_path)
    _vosk_recognizer_small = vosk.KaldiRecognizer(_vosk_model_small, SAMPLE_RATE)
    _vosk_recognizer_small.SetWords(True)
    _current_recognizer = _vosk_recognizer_small

    spinner_active = False
    th_small_spinner.join()
    print("Fast model loaded. Transcription starting...")

    # --- Step 2: Start background thread to load LARGE model ---
    if large_model_path:
        load_large_thread = threading.Thread(
            target=_load_large_model_in_background,
            args=(large_model_path, SAMPLE_RATE),
            daemon=True
        )
        load_large_thread.start()
    else:
        _large_model_loaded_event.set()

    # --- Final Audio Device Selection (after potential config load/creation) ---
    # This ensures that if device_index was set in config or chosen by user, it's used.
    # If a command-line --device-index was provided, it takes precedence.
    if dev_idx is None:
        list_audio_devices()
        try:
            dev_idx = int(input("Select device index: "))
        except ValueError:
            print("Invalid input. Exiting.", file=sys.stderr)
            sys.exit(1)
    
    # Check if selected/configured device is valid for input
    try:
        device_info = sd.query_devices(dev_idx, 'input')
    except Exception as e:
        print(f"Error: Selected audio device index {dev_idx} is not valid for input: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Session Management (Output Directory Setup) ---
    if output_dir:
        # If output_dir is provided (and is not None), sessions will be a subfolder of it.
        sessions_root = os.path.join(output_dir, 'sessions')
    else:
        # If no output_dir is provided (it's None), create the 'sessions' folder in the current directory.
        sessions_root = 'sessions' # This will resolve to ./sessions

    os.makedirs(sessions_root, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(sessions_root, f"session_{ts}")
    os.makedirs(base, exist_ok=True)
    utter_dir = os.path.join(base, 'utterances')
    os.makedirs(utter_dir, exist_ok=True)

    print(f"Recording from device {dev_idx}. Session folder: {base}")
    transcripts = []
    _start_hotkey_listener()

    # --- Variables for Console Display and Pause Detection ---
    last_displayed_console_text = ""
    last_partial_update_time = time.monotonic()
    last_audio_time = time.monotonic()
    last_final_text_printed = ""

    switched_to_large_model = False

    # --- Main Audio Processing Loop ---
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE,
                            device=dev_idx, # Use the determined device index
                            channels=CHANNELS,
                            dtype=DTYPE,
                            blocksize=BLOCK_SIZE,
                            callback=_audio_callback):
            _current_recognizer.Result()
            sys.stdout.write("Start speaking. (Using fast model initially)\n")
            sys.stdout.flush()

            while not _finish_event.is_set():
                current_time = time.monotonic()

                # --- Step 3: Check for and perform seamless switch to large model ---
                if not switched_to_large_model and _large_model_loaded_event.is_set():
                    if _vosk_recognizer_large:
                        if last_displayed_console_text:
                            sys.stdout.write('\r\033[K') # Use ANSI clear
                            sys.stdout.flush()
                            sys.stdout.write(last_displayed_console_text + '\n')
                            sys.stdout.flush()
                            transcripts.append(last_displayed_console_text)
                            last_final_text_printed = last_displayed_console_text

                        _current_recognizer = _vosk_recognizer_large
                        _current_recognizer.Result()
                        switched_to_large_model = True
                        last_displayed_console_text = ""
                        last_partial_update_time = current_time
                        last_audio_time = current_time

                        print("\nSwitched to high-accuracy model. Enjoy!", file=sys.stderr)
                        sys.stdout.write(">> High-accuracy recognition active.\n")
                        sys.stdout.flush()
                    else:
                        switched_to_large_model = True

                # Get audio data or handle empty queue
                try:
                    data = _audio_queue.get(timeout=0.05)
                    last_audio_time = current_time
                    
                    if _current_recognizer.AcceptWaveform(data):
                        # --- Final Utterance Recognized ---
                        res = json.loads(_current_recognizer.Result())
                        full_text = _clean_text(res.get('text','')).strip()

                        if full_text:
                            new_segment = full_text[len(last_final_text_printed):].strip()
                            if new_segment:
                                sys.stdout.write('\r\033[K') # Use ANSI clear
                                sys.stdout.write(new_segment + ' ')
                                sys.stdout.flush()
                                last_final_text_printed = full_text

                            if full_text and (not transcripts or full_text != transcripts[-1]):
                                transcripts.append(full_text)
                                fn = datetime.now().strftime("utterance_%Y%m%d_%H%M%S.txt")
                                with open(os.path.join(utter_dir, fn), 'w', encoding='utf-8') as f:
                                    f.write(full_text)
                        
                        sys.stdout.write('\n')
                        sys.stdout.flush()
                        last_displayed_console_text = ""
                        last_partial_update_time = current_time
                        
                    else:
                        # --- Interim (Partial) Result ---
                        if current_time - last_audio_time <= SILENCE_THRESHOLD_SECONDS:
                            partial_json = json.loads(_current_recognizer.PartialResult())
                            raw_partial = partial_json.get('partial', '').strip()
                            filtered_partial = _clean_partial_for_display(raw_partial)

                            # REVISED PARTIAL DISPLAY LOGIC - BALANCED
                            if filtered_partial: # Must be non-empty after filtering
                                should_update = False

                                if filtered_partial != last_displayed_console_text: # Content has actually changed
                                    if len(filtered_partial) > len(last_displayed_console_text):
                                        # It's longer, almost always a progressive update
                                        should_update = True
                                    elif not last_displayed_console_text.startswith(filtered_partial) and \
                                         not filtered_partial.startswith(last_displayed_console_text):
                                        # Neither is a prefix of the other, implies a correction or different path
                                        should_update = True
                                    # If length is the same or shorter, only update if it's a meaningful correction
                                    # (not just a minor variation, and not a regression)
                                    elif len(filtered_partial) >= MIN_PARTIAL_LENGTH_FOR_DISPLAY:
                                        should_update = True
                                    
                                    # Ensure we display the first meaningful partial, even if short
                                    if not last_displayed_console_text and len(filtered_partial) >= MIN_PARTIAL_LENGTH_FOR_DISPLAY:
                                        should_update = True
                                        
                                # Condition: Throttle check
                                if should_update and (current_time - last_partial_update_time >= PARTIAL_UPDATE_THROTTLE_SECONDS):
                                    sys.stdout.write('\r\033[K') # Use ANSI clear
                                    sys.stdout.write(filtered_partial)
                                    sys.stdout.flush()
                                    last_displayed_console_text = filtered_partial
                                    last_partial_update_time = current_time
                            # --- END REVISED PARTIAL DISPLAY LOGIC ---
                        else:
                            if last_displayed_console_text:
                                sys.stdout.write('\r\033[K') # Use ANSI clear
                                sys.stdout.flush()
                                last_displayed_console_text = ""
                                last_partial_update_time = current_time

                except queue.Empty:
                    if current_time - last_partial_update_time > PAUSE_THRESHOLD_SECONDS and last_displayed_console_text:
                        sys.stdout.write('\r\033[K') # Use ANSI clear
                        sys.stdout.write(last_displayed_console_text + '\n')
                        sys.stdout.flush()
                        
                        if last_displayed_console_text and (not transcripts or last_displayed_console_text != transcripts[-1]):
                            transcripts.append(last_displayed_console_text)
                            last_final_text_printed = last_displayed_console_text

                        last_displayed_console_text = ""
                        last_partial_update_time = current_time
                        _current_recognizer.Result()
                        
                    elif current_time - last_audio_time > SILENCE_THRESHOLD_SECONDS:
                        if last_displayed_console_text:
                            sys.stdout.write('\r\033[K') # Use ANSI clear
                            sys.stdout.flush()
                            last_displayed_console_text = ""
                            last_partial_update_time = current_time
                        _current_recognizer.Result()


    except KeyboardInterrupt:
        _finish_event.set()

    # --- Finalization Steps After Loop Exits ---
    if last_displayed_console_text:
        sys.stdout.write('\r\033[K') # Use ANSI clear
        sys.stdout.flush()
        sys.stdout.write(last_displayed_console_text + '\n')
        sys.stdout.flush()
        if not transcripts or last_displayed_console_text != transcripts[-1]:
            transcripts.append(last_displayed_console_text)
            last_final_text_printed = last_displayed_console_text


    final_trailing_res = json.loads(_current_recognizer.FinalResult()).get('text','').strip()
    if final_trailing_res:
        final_trailing_res = _clean_text(final_trailing_res)
        
        new_segment = final_trailing_res[len(last_final_text_printed):].strip()
        if new_segment:
            sys.stdout.write(new_segment + '\n')
            sys.stdout.flush()
        elif last_final_text_printed:
             sys.stdout.write('\n')

        if not transcripts or (transcripts and final_trailing_res and final_trailing_res != transcripts[-1]):
            transcripts.append(final_trailing_res)
            fn = datetime.now().strftime("utterance_final_%Y%m%d_%H%M%S.txt")
            with open(os.path.join(utter_dir, fn), 'w', encoding='utf-8') as f:
                f.write(final_trailing_res)

    if transcripts:
        master_transcript_file = os.path.join(base, 'transcript.txt')
        with open(master_transcript_file, 'w', encoding='utf-8') as mf:
            mf.write("\n\n".join(transcripts))
        print(f"\nFull transcript saved to: {master_transcript_file}")
    else:
        print("\nNo transcript recorded during this session.")

    print("\nExiting application.")

if __name__=='__main__':
    main()