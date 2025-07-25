
r"""
# Voice Input Transcription with Vosk

This Python script provides a live, real-time speech-to-text transcription service using the Vosk speech recognition toolkit. It employs a progressive enhancement approach, starting with a smaller, faster model for immediate feedback and seamlessly switching to a larger, more accurate model in the background once loaded.

## Features

  * **Real-time Transcription:** Converts spoken audio from your microphone into text as you speak.
  * **Progressive Accuracy:** Initiates with a fast, lightweight Vosk model for quick startup and transitions to a more accurate, larger model for improved transcription quality.
  * **Offline Operation:** Vosk models run locally on your machine, ensuring privacy and eliminating the need for an internet connection during transcription.
  * **Configurable Display Modes:**
      * **Default (Batch Display):** Provides clean, stable on-screen output by displaying full utterances only after a pause or periodic flush. This mode is ideal for clean transcription display.
      * **Immediate Display:** Can be enabled via a command-line argument for real-time partial results, offering immediate visual feedback, though it may exhibit visual artifacts like line repetitions in a standard console.
  * **Intelligent Cleaning:** Filters common filler words (like "um", "uh", "huh", "by") and intelligently updates/suppresses partial results on the console for a cleaner user experience, especially in batch mode.
  * **Hotkeys:** Use `Ctrl+Alt+Space` to gracefully stop the transcription process.
  * **Session Management:** Transcripts are saved into timestamped session folders, with individual utterances and a complete session transcript.
  * **Accent Support:** Configurable to load different English accent models (e.g., US, Indian, British) for improved accuracy with various speech patterns.

## Requirements

  * Python 3.x
  * `sounddevice`
  * `vosk`
  * `pynput`
  * `colorama`

You can install these dependencies using pip:
`pip install sounddevice vosk pynput colorama`

## Vosk Models

This script utilizes two Vosk models for its progressive enhancement feature, and can be configured to use different English accent models.

Downloading Vosk Models:

You must download these Vosk models and place them in the same directory as this Python script (`voice_input_vosk.py`) or specify their paths using command-line arguments or the `voice_config.txt` file.

You can download the models from the official Vosk website:

  * **Vosk Models Page:** [https://alphacephei.com/vosk/models](https://alphacephei.com/vosk/models)

    Navigate to this page to find a list of available models. For this script, look for the English models. Examples include:
      * `vosk-model-small-en-us-0.15` (Small, Fast US English)
      * `vosk-model-en-us-0.22` (Large, High Accuracy US English)
      * `vosk-model-small-en-in-0.4` (Lightweight Indian English)
      * `vosk-model-en-in-0.5` (Generic Indian English)
      * `vosk-model-en-gb-something` (Example for British English - verify exact name on site)

Extraction:

After downloading, the models will be in compressed archives (e.g., `.zip` or `.tar.gz`). You must unzip/extract these archives into their respective folders. For example, `vosk-model-small-en-us-0.15.zip` should be extracted to a folder named `vosk-model-small-en-us-0.15`.

Placement:

Ensure the extracted model directories (e.g., `vosk-model-small-en-us-0.15/` and `vosk-model-en-us-0.22/`) are located in the same folder as your `voice_input_vosk.py` script, or specify their paths in `voice_config.txt`.

## Configuration (Optional)

You can create a `voice_config.txt` file in the same directory as the script to customize settings. Here's an example:

```
# voice_config.txt
device_index = 1 # Your audio input device index (run without arg to list devices)
# Model paths can be specified directly, or derived from language_model setting
small_model_path = vosk-model-small-en-us-0.15 
large_model_path = vosk-model-en-us-0.22
output_dir = transcripts
language_model = en-us # Set to 'en-in', 'en-gb', etc. to load different accent models
```

## Usage

To run the script with the default (batch display) mode, simply execute it from your terminal:

`python voice_input_vosk.py`

### Command-line Arguments

You can override configuration settings directly from the command line:

  * `--config <path>`: Specify a different configuration file.
  * `--device-index <index>`: Manually set the audio input device index. Run the script without this argument first to see a list of available devices.
  * `--small-model-path <path>`: Specify the path to the small Vosk model. **Overrides `language_model` if specified.**
  * `--large-model-path <path>`: Specify the path to the large Vosk model. **Overrides `language_model` if specified.**
  * `--output-dir <path>`: Specify the directory to save session transcripts.
  * `--immediate-display`: Enable immediate display of partial results. By default, batch display is active for cleaner output.
  * `--language-model <code_str>`: Specify the accent-specific language model to use (e.g., `en-us`, `en-in`, `en-gb`). This will dynamically select `small_model_path` and `large_model_path` unless they are explicitly provided.

Example with arguments:

`python voice_input_vosk.py --device-index 0 --output-dir my_transcripts --language-model en-in`

The script will guide you through selecting an audio input device if `device_index` is not specified.
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

# --- Configuration Constants ---
CONFIG_FILE = "voice_config.txt"
CONFIG_DEFAULTS = {
    'device_index': None,
    'device_name': None, # NEW: Store device name for robust selection
    'small_model_path': 'vosk-model-small-en-us-0.15', # Default to US English
    'large_model_path': 'vosk-model-en-us-0.22',      # Default to US English
    'output_dir': 'transcripts', # Default output directory
    'language_model': 'en-us'    # Default language model
}

# Define available language models and their corresponding Vosk model directories
LANGUAGE_MODELS = {
    'en-us': {
        'small': 'vosk-model-small-en-us-0.15',
        'large': 'vosk-model-en-us-0.22'
    },
    'en-in': {
        'small': 'vosk-model-small-en-in-0.4',
        'large': 'vosk-model-en-in-0.5'
    }
}


# --- Audio & Vosk Settings ---
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = 'int16'
BLOCK_SIZE = 400
LEADING_FILLER_WORDS = {"um", "uh", "huh"}

# --- Display & Behavioral Thresholds ---
PAUSE_THRESHOLD_SECONDS = 1.0 # Duration of silence to consider an utterance potentially finished
SILENCE_THRESHOLD_SECONDS = 0.5 # Duration of no audio to trigger a recognizer state check
PARTIAL_UPDATE_THROTTLE_SECONDS = 0.3 # Max frequency for updating partial console output
MIN_PARTIAL_LENGTH_FOR_DISPLAY = 2 # Minimum length for a partial to be considered for display
FORCED_FLUSH_INTERVAL_SECONDS = 4.0 # Interval for forcing a recognizer flush in batch mode for updates

# --- Common Filler Words for Display Suppression ---
COMMON_FILL_DISPLAY_SUPPRESS = {
    "the", "a", "an", "i", "and", "oh", "um", "uh", "mhm", "hmm", "yeah", "okay",
    "you", "so", "well", "like", "just", "it's", "is", "of", "to", "in", "for",
    "that", "no", "yes", "on", "he", "she", "we", "they", "was", "are", "do", "don't",
    "huh", "by"
}

# --- Globals (Managed by the application, shared across functions via queue/event) ---
_audio_queue = queue.Queue()
_finish_event = threading.Event() # Set to signal the main loop to stop


# --- Hotkey Callbacks ---
def _on_finish_hotkey():
    """Callback function for the global hotkey to stop transcription."""
    print("\nFinish hotkey pressed. Finalizing transcript...", file=sys.stderr)
    _finish_event.set()


def _start_hotkey_listener():
    """Starts a global keyboard listener for the finish hotkey."""
    try:
        listener = keyboard.GlobalHotKeys({'<ctrl>+<alt>+<space>': _on_finish_hotkey})
        listener.daemon = True # Allows the program to exit even if the listener is still running
        listener.start()
    except Exception as e:
        print(f"Warning: Hotkey disabled (requires admin/root or specific OS setup): {e}", file=sys.stderr)


# --- Text Cleaning Functions ---
def _clean_text(text: str) -> str:
    """Removes leading filler words from a given text string."""
    words = text.split()
    
    # Remove leading filler words
    while words and words[0].lower() in LEADING_FILLER_WORDS:
        words.pop(0)
    
    return ' '.join(words)


def _clean_partial_for_display(partial_text: str) -> str:
    """Filters out common filler words and short, less meaningful phrases for partial display.
       Also used for filtering forced flush results in batch mode.
    """
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
    """Callback function for sounddevice to put audio data into the queue."""
    if status:
        pass # print(status, file=sys.stderr) # Uncomment for debug
    _audio_queue.put(bytes(indata))


# --- Utility Functions ---
def list_audio_devices():
    """Prints a list of available audio input devices to stderr."""
    print("\nAvailable Audio Input Devices:", file=sys.stderr)
    try:
        for idx, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] > 0:
                print(f"  [{idx}] {dev['name']}", file=sys.stderr)
    except Exception as e:
        print(f"Error listing audio devices: {e}", file=sys.stderr)


# --- Helper Functions for Main Logic ---

def _load_configuration_and_args(config_file_path: str, args: argparse.Namespace) -> dict:
    """
    Loads configuration from file and command-line arguments.
    If config file doesn't exist, prompts user for device index and creates it.
    Also prompts for language model if not specified or invalid.
    Handles fallback to US English models if selected Indian models are not found.
    Ensures 'transcripts' is the default output directory.
    Command-line arguments override config file settings.
    """
    cfg = CONFIG_DEFAULTS.copy()
    config_exists = os.path.exists(config_file_path)
    
    # Flag to control device index prompting
    prompt_for_device_selection = False

    if config_exists:
        print(f"Configuration loaded from '{config_file_path}'.")
        with open(config_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    k, v = line.split('=', 1)
                    if v.strip().lower() == 'none':
                        cfg[k.strip()] = None
                    else:
                        cfg[k.strip()] = v.strip()
        
        # Try to resolve device from config file (by name first, then index)
        # Only if not overridden by command line arg --device-index
        if args.device_index is None: 
            if cfg.get('device_name'):
                # Attempt to find device by stored name
                found_by_name = False
                for idx, dev in enumerate(sd.query_devices()):                    
                    if dev['max_input_channels'] > 0 and dev['name'].lower() == cfg['device_name'].lower():
                        cfg['device_index'] = idx # Use the current index for that name
                        print(f"Found saved device '{cfg['device_name']}' at index {idx}.", file=sys.stderr)
                        found_by_name = True
                        break
                if not found_by_name:
                    print(f"Warning: Saved device '{cfg['device_name']}' not found by name.", file=sys.stderr)
                    # Fallback to index if name lookup fails, but only if index is valid
                    if cfg.get('device_index') is not None:
                        try:
                            int(cfg['device_index']) # Validate if it's still a number
                            print(f"Attempting to use saved device index {cfg['device_index']}.", file=sys.stderr)
                            # Let the final sd.query_devices check in main handle if this index is ambiguous
                        except (ValueError, TypeError):
                            cfg['device_index'] = None # Invalid index in config
                            prompt_for_device_selection = True
                    else:
                        prompt_for_device_selection = True # No name, no valid index, so prompt
            elif cfg.get('device_index') is not None: # No device_name, but old device_index exists
                try:
                    int(cfg['device_index']) # Validate if it's still a number
                    # No prompt, assume it's valid for now, final sd.query_devices check will confirm
                except (ValueError, TypeError):
                    cfg['device_index'] = None
                    prompt_for_device_selection = True
            else: # Config exists but no device_name or device_index
                prompt_for_device_selection = True
        else: # args.device_index is provided, it takes precedence
            cfg['device_index'] = args.device_index
            # When device_index is explicitly given via command-line, device_name from config is ignored.
            # We will try to resolve the name for saving, but not rely on it for lookup this run.
            prompt_for_device_selection = False # No need to prompt, arg takes over

    else: # Config file did not exist
        print(f"Configuration file '{config_file_path}' not found.", file=sys.stderr)
        prompt_for_device_selection = True # Definitely prompt if no config file

    # --- Prompt for Device Index if needed ---
    if prompt_for_device_selection:
        print(f"Please select an audio input device. {'(Config file not found or device invalid)' if not config_exists else ''}", file=sys.stderr)
        list_audio_devices()
        dev_idx_input = None
        selected_dev_info = None
        while dev_idx_input is None:
            try:
                dev_idx_input = int(input("Select audio input device index: "))
                if dev_idx_input < 0:
                    raise ValueError
                # Validate the selected index against sounddevice.query_devices() to get full info and catch errors early
                selected_dev_info = sd.query_devices(dev_idx_input, 'input')
            except (ValueError, TypeError):
                print("Invalid input. Please enter a number.", file=sys.stderr)
                dev_idx_input = None
            except sd.PortAudioError as e: # Catch specific sounddevice errors for invalid/ambiguous devices
                print(f"Error selecting device: {e}. Please try another index.", file=sys.stderr)
                dev_idx_input = None

        cfg['device_index'] = dev_idx_input
        # Save the name of the selected device for more robust future lookups
        cfg['device_name'] = selected_dev_info['name'] if selected_dev_info else None 
        print(f"Device index {dev_idx_input} ('{cfg['device_name'] if cfg['device_name'] else 'Unknown'}') selected and will be saved to config.", file=sys.stderr)


    # --- Language Model Selection and Fallback Logic ---
    # Determine the initial language preference
    # Prioritize --language-model command-line arg. If not set, check config file.
    initial_language_preference = args.language_model or cfg.get('language_model')

    # If the language preference isn't a known model OR the config file didn't exist at all, prompt the user
    if initial_language_preference not in LANGUAGE_MODELS or not config_exists:
        # Only print the "Available language models" heading if the config file didn't exist,
        # or if an invalid language_model was found in an existing config.
        if not config_exists:
            print("\nAvailable language models for English:", file=sys.stderr)
            for lang_code in LANGUAGE_MODELS.keys():
                print(f"  - {lang_code}", file=sys.stderr)
        elif initial_language_preference not in LANGUAGE_MODELS: # Print if config exists but language_model is invalid
             print(f"Invalid language_model '{initial_language_preference}' found in config. Please select again.", file=sys.stderr)
             print("\nAvailable language models for English:", file=sys.stderr)
             for lang_code in LANGUAGE_MODELS.keys():
                print(f"  - {lang_code}", file=sys.stderr)

        lang_input = None
        while lang_input not in LANGUAGE_MODELS:
            lang_input = input(f"Select a language model ({'/'.join(LANGUAGE_MODELS.keys())}, default 'en-us'): ").strip().lower()
            if not lang_input:  # Handle empty input, default to 'en-us'
                lang_input = 'en-us'
            if lang_input not in LANGUAGE_MODELS:
                print("Invalid language model. Please choose from the list or press Enter for default.", file=sys.stderr)
        cfg['language_model'] = lang_input
        print(f"Language model '{lang_input}' selected and will be saved to config.", file=sys.stderr)
    else:
        cfg['language_model'] = initial_language_preference # Use the preference found (from arg or config)

    # Determine the model paths based on the selected language model
    # This 'chosen_models' will provide the default paths for the selected language
    chosen_models = LANGUAGE_MODELS.get(cfg['language_model'], LANGUAGE_MODELS['en-us'])

    # Initialize final model paths, prioritizing command-line arguments
    final_small_model_path = args.small_model_path
    final_large_model_path = args.large_model_path

    # If command-line arguments for model paths were NOT provided,
    # then use the paths from the loaded config. If config paths are invalid/missing,
    # then use the paths from the selected language model.
    if not final_small_model_path:
        if cfg.get('small_model_path') and os.path.isdir(cfg['small_model_path']):
            final_small_model_path = cfg['small_model_path']
        else:
            final_small_model_path = chosen_models['small']
            print(f"Using default small model path for '{cfg['language_model']}': {final_small_model_path}", file=sys.stderr)
            
    if not final_large_model_path:
        if cfg.get('large_model_path') and os.path.isdir(cfg['large_model_path']):
            final_large_model_path = cfg['large_model_path']
        else:
            final_large_model_path = chosen_models['large']
            # print(f"Using default large model path for '{cfg['language_model']}': {final_large_model_path}", file=sys.stderr) # Can be noisy


    # Final existence checks and fallbacks for models
    if not os.path.isdir(final_small_model_path):
        print(f"Error: Small Vosk model not found at '{final_small_model_path}'. Please download and extract it.", file=sys.stderr)
        sys.exit(1) # Small model is mandatory

    if final_large_model_path and not os.path.isdir(final_large_model_path):
        print(f"Warning: High-accuracy Vosk model not found at '{final_large_model_path}'. "
              "Application will proceed using only the fast model.", file=sys.stderr)
        final_large_model_path = None # Set to None for only small model usage

    cfg['small_model_path'] = final_small_model_path
    cfg['large_model_path'] = final_large_model_path
    
    # Output directory logic: Command line arg has highest priority
    # If not provided via command line, use value from config file, otherwise default to 'transcripts'
    cfg['output_dir'] = args.output_dir if args.output_dir is not None else cfg.get('output_dir', 'transcripts')


    # Write/rewrite the config file with selected settings
    try:
        with open(config_file_path, 'w', encoding='utf-8') as f:
            f.write("# This file contains configuration settings for the voice transcription script.\n")
            f.write("# Lines starting with '#' are comments and are ignored.\n")
            f.write("# Set 'device_index' to your preferred audio input device index.\n")
            f.write("# Set 'device_name' to your preferred audio input device name for robust selection.\n") # NEW comment
            f.write("# Specify the paths to your Vosk model directories (must be extracted).\n")
            f.write("# Set 'output_dir' to the desired directory for saving transcripts.\n")
            f.write("# Set 'language_model' to specify the accent/language (e.g., 'en-us', 'en-in').\n\n")
            
            f.write(f"device_index = {cfg['device_index']}\n")
            f.write(f"device_name = {cfg['device_name'] if cfg['device_name'] else 'None'}\n") # NEW: Write device_name
            f.write(f"small_model_path = {cfg['small_model_path']}\n")
            f.write(f"large_model_path = {cfg['large_model_path'] if cfg['large_model_path'] else 'None'}\n")
            f.write(f"output_dir = {cfg['output_dir']}\n")
            f.write(f"language_model = {cfg['language_model']}\n")
        if not config_exists:
            print(f"Default config file created at '{config_file_path}'.", file=sys.stderr)
    except IOError as e:
        print(f"Error creating config file '{config_file_path}': {e}", file=sys.stderr)
        sys.exit(1)

    return cfg

def _setup_session_directories(output_base_dir: str) -> tuple[str, str]:
    """
    Sets up the timestamped session and utterance directories directly under the output_base_dir.
    Returns a tuple: (base_session_dir, utterances_dir).
    """
    # Use output_base_dir directly as the root for session folders
    sessions_root = output_base_dir 

    os.makedirs(sessions_root, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # The base directory for the session will now be directly under output_base_dir
    base_dir = os.path.join(sessions_root, f"session_{ts}")
    os.makedirs(base_dir, exist_ok=True)
    
    utter_dir = os.path.join(base_dir, 'utterances')
    os.makedirs(utter_dir, exist_ok=True)
    return base_dir, utter_dir

def _initialize_vosk_recognizers(small_model_path: str, large_model_path: str | None, sample_rate: int) -> tuple[vosk.KaldiRecognizer, threading.Event, list]:
    """
    Loads Vosk models and initializes recognizers.
    Starts a background thread for loading the large model.
    Returns: (initial_recognizer, large_model_loaded_event, large_recognizer_instance_container)
    The large_recognizer_instance_container is a mutable list [None] that will hold the
    vosk.KaldiRecognizer instance for the large model once loaded in the background thread.
    """
    # Load SMALL model FIRST (blocking)
    spinner_active = True
    def spinner_small_model_load():
        for c in itertools.cycle('|/-\\'):
            if not spinner_active: break
            sys.stderr.write(f"\rLoading fast model... {c}")
            sys.stderr.flush()
            time.sleep(0.1)
        sys.stderr.write("\r" + " "*20 + "\r")
        sys.stderr.flush()

    # Create the small model instance
    _vosk_model_small = vosk.Model(small_model_path)
    vosk_recognizer_small = vosk.KaldiRecognizer(_vosk_model_small, sample_rate)
    vosk_recognizer_small.SetWords(True)

    th_small_spinner = threading.Thread(target=spinner_small_model_load, daemon=True)
    th_small_spinner.start()
    
    spinner_active = False # Signal spinner to stop as model is loaded
    th_small_spinner.join() # Wait for spinner thread to finish its cleanup

    print("Fast model loaded. Transcription starting...")

    # Set up container for large recognizer and event for signaling
    large_recognizer_instance_container = [None] # This will hold the Vosk.KaldiRecognizer instance
    large_model_loaded_event = threading.Event()

    def _load_large_model_in_background_thread(model_path: str, sr: int, container: list, event: threading.Event):
        """Internal function for the background thread to load the large model."""
        print("\nLoading high-accuracy model in background (this may take a moment)...", file=sys.stderr)
        try:
            vosk_model_large = vosk.Model(model_path)
            vosk_recognizer_large = vosk.KaldiRecognizer(vosk_model_large, sr)
            vosk_recognizer_large.SetWords(True)
            container[0] = vosk_recognizer_large # Store in shared container
            event.set()
            print("High-accuracy model loaded. Will switch at next natural break.", file=sys.stderr)
        except Exception as e:
            print(f"Error loading high-accuracy model in background: {e}", file=sys.stderr)
            event.set() # Signal completion even on error to unblock main thread
            container[0] = None # Ensure it's None if loading failed

    if large_model_path:
        load_large_thread = threading.Thread(
            target=_load_large_model_in_background_thread,
            args=(large_model_path, sample_rate, large_recognizer_instance_container, large_model_loaded_event),
            daemon=True
        )
        load_large_thread.start()
    else:
        # If no large model path, immediately set the event to skip waiting
        large_model_loaded_event.set()

    return vosk_recognizer_small, large_model_loaded_event, large_recognizer_instance_container


def _process_audio_stream(
    dev_idx: int,
    sample_rate: int,
    batch_display_mode: bool, # This controls the display mode
    transcripts: list[str],
    utter_dir: str,
    initial_recognizer: vosk.KaldiRecognizer,
    large_model_loaded_event: threading.Event,
    large_recognizer_container: list # Pass the mutable list to get the large recognizer
):
    """
    Handles the main audio processing loop, Vosk recognition, and console display.
    Manages model switching and forced flushing in batch mode.
    """
    _current_recognizer = initial_recognizer
    switched_to_large_model = False

    current_line_buffer = ""
    has_printed_to_current_line = False
    last_partial_update_time = time.monotonic()
    last_audio_time = time.monotonic()
    last_forced_flush_time = time.monotonic() # For periodic flushing in batch mode

    initial_message = "Start speaking."
    if batch_display_mode:
        initial_message += " (Batch display mode)"
    else:
        initial_message += " (Immediate display mode)"
    sys.stdout.write(initial_message + "\n")
    sys.stdout.flush()

    try:
        with sd.InputStream(samplerate=sample_rate,
                            device=dev_idx,
                            channels=CHANNELS,
                            dtype=DTYPE,
                            blocksize=BLOCK_SIZE,
                            callback=_audio_callback):
            _current_recognizer.Result() # Clear any initial state

            while not _finish_event.is_set():
                current_time = time.monotonic()

                # --- Model Switch Logic ---
                if not switched_to_large_model and large_model_loaded_event.is_set():
                    # Attempt to get the large recognizer instance from the container
                    vosk_recognizer_large = large_recognizer_container[0] 

                    if vosk_recognizer_large: # Check if it was successfully loaded
                        # Before switching, try to finalize any pending speech from the small model
                        # by forcing a flush (AcceptWaveform with empty bytes) or waiting for a pause.
                        if _current_recognizer.AcceptWaveform(b'') or \
                           (current_time - last_audio_time > PAUSE_THRESHOLD_SECONDS):
                            
                            # Finalize any pending text from the small model before switching
                            raw_res_before_switch = json.loads(_current_recognizer.Result()).get('text','').strip()
                            cleaned_res_before_switch = _clean_text(raw_res_before_switch)
                            # Apply _clean_partial_for_display to ensure non-meaningful words are suppressed
                            filtered_res_before_switch = _clean_partial_for_display(cleaned_res_before_switch) 

                            if filtered_res_before_switch: # Only print if meaningful after full cleaning
                                if has_printed_to_current_line:
                                    sys.stdout.write('\r\033[K')
                                sys.stdout.write(filtered_res_before_switch + '\n')
                                sys.stdout.flush()
                                if filtered_res_before_switch not in transcripts:
                                    transcripts.append(filtered_res_before_switch)
                                current_line_buffer = ""
                                has_printed_to_current_line = False

                            _current_recognizer = vosk_recognizer_large # PERFORM THE SWITCH
                            _current_recognizer.Result() # Clear new recognizer's initial state
                            switched_to_large_model = True
                            
                            print("\nSwitched to high-accuracy model. Enjoy!", file=sys.stderr)
                            sys.stdout.write(">> High-accuracy recognition active.\n")
                            sys.stdout.flush()
                    else:
                        # If large model failed to load but event is set, just mark as switched
                        # so we don't keep trying to switch.
                        switched_to_large_model = True 


                # --- Forced Flush for Batch Display (Periodic Updates) ---
                # This logic ONLY applies when batch_display_mode is TRUE
                # Only perform forced flush if enough time has passed AND there has been recent audio (not absolute silence).
                if batch_display_mode and \
                   (current_time - last_forced_flush_time >= FORCED_FLUSH_INTERVAL_SECONDS) and \
                   (current_time - last_audio_time < SILENCE_THRESHOLD_SECONDS):
                    
                    raw_forced_flush_res = json.loads(_current_recognizer.Result()).get('text','').strip()
                    cleaned_forced_flush_res = _clean_text(raw_forced_flush_res)
                    # Apply _clean_partial_for_display to filter out short, non-meaningful content (like "huh")
                    filtered_forced_flush_res = _clean_partial_for_display(cleaned_forced_flush_res) 

                    if filtered_forced_flush_res: # Only print if there's meaningful text after cleaning
                        if has_printed_to_current_line:
                            sys.stdout.write('\r\033[K')
                        sys.stdout.write(filtered_forced_flush_res + '\n')
                        sys.stdout.flush()
                        if filtered_forced_flush_res not in transcripts:
                            transcripts.append(filtered_forced_flush_res)
                        current_line_buffer = ""
                        has_printed_to_current_line = False
                        
                    _current_recognizer.Result() # Clear buffer after forced pull, even if empty
                    last_forced_flush_time = current_time # Reset timer


                # --- Get audio data or handle empty queue ---
                try:
                    data = _audio_queue.get(timeout=0.05)
                    last_audio_time = current_time
                    
                    if _current_recognizer.AcceptWaveform(data):
                        # --- Final Utterance Recognized ---
                        res = json.loads(_current_recognizer.Result())
                        full_text = _clean_text(res.get('text','')).strip()

                        if full_text:
                            if has_printed_to_current_line:
                                sys.stdout.write('\r\033[K')
                            sys.stdout.write(full_text + '\n')
                            sys.stdout.flush()
                            current_line_buffer = ""
                            has_printed_to_current_line = False

                            if full_text not in transcripts:
                                transcripts.append(full_text)
                                fn = datetime.now().strftime("utterance_%Y%m%d_%H%M%S.txt")
                                with open(os.path.join(utter_dir, fn), 'w', encoding='utf-8') as f:
                                    f.write(full_text)
                        else:
                            # If final result is empty, clear line and move cursor down
                            # BUT ONLY IF we had something on the line. Otherwise, avoid extra newlines.
                            if has_printed_to_current_line:
                                sys.stdout.write('\r\033[K')
                                sys.stdout.flush()
                                current_line_buffer = ""
                                has_printed_to_current_line = False
                                sys.stdout.write('\n') # Move cursor to next line
                                sys.stdout.flush()
                        
                        last_partial_update_time = current_time
                        last_forced_flush_time = current_time # Reset forced flush on final result
                        
                    else:
                        # --- Interim (Partial) Result ---
                        # This block is ONLY executed if batch_display_mode is FALSE (i.e., immediate mode)
                        if not batch_display_mode:
                            partial_json = json.loads(_current_recognizer.PartialResult())
                            raw_partial = partial_json.get('partial', '').strip()
                            filtered_partial = _clean_partial_for_display(raw_partial)

                            if filtered_partial:
                                is_meaningful_update = False
                                
                                if not current_line_buffer and len(filtered_partial) >= MIN_PARTIAL_LENGTH_FOR_DISPLAY:
                                    is_meaningful_update = True
                                elif len(filtered_partial) > len(current_line_buffer) + 3:
                                    is_meaningful_update = True
                                elif filtered_partial != current_line_buffer and \
                                     (len(filtered_partial) > len(current_line_buffer) or \
                                      abs(len(filtered_partial) - len(current_line_buffer)) > 5):
                                    is_meaningful_update = True
                                
                                if is_meaningful_update and (current_time - last_partial_update_time >= PARTIAL_UPDATE_THROTTLE_SECONDS):
                                    sys.stdout.write('\r\033[K')
                                    sys.stdout.write(filtered_partial)
                                    sys.stdout.flush()
                                    current_line_buffer = filtered_partial
                                    has_printed_to_current_line = True
                                    last_partial_update_time = current_time
                            elif has_printed_to_current_line and current_time - last_partial_update_time >= PARTIAL_UPDATE_THROTTLE_SECONDS:
                                # If partial became empty (e.g., user stopped talking, Vosk refined to nothing)
                                # and we previously printed something, clear it.
                                sys.stdout.write('\r\033[K')
                                sys.stdout.flush()
                                current_line_buffer = ""
                                has_printed_to_current_line = False
                                last_partial_update_time = current_time

                except queue.Empty:
                    # No new audio data, check for silence/pause
                    if current_time - last_audio_time > SILENCE_THRESHOLD_SECONDS:
                        raw_final_res_on_silence = json.loads(_current_recognizer.Result()).get('text','').strip()
                        cleaned_final_res_on_silence = _clean_text(raw_final_res_on_silence)
                        # Apply _clean_partial_for_display to filter out short, non-meaningful content (like "huh")
                        filtered_final_res_on_silence = _clean_partial_for_display(cleaned_final_res_on_silence) 

                        if filtered_final_res_on_silence: # Only print if meaningful after cleaning and filtering
                            if has_printed_to_current_line:
                                sys.stdout.write('\r\033[K')
                            sys.stdout.write(filtered_final_res_on_silence + '\n')
                            sys.stdout.flush()
                            if filtered_final_res_on_silence not in transcripts:
                                transcripts.append(filtered_final_res_on_silence)
                            current_line_buffer = ""
                            has_printed_to_current_line = False
                        elif has_printed_to_current_line:
                            # If we were showing a partial but got an empty/cleaned/filtered final result during silence,
                            # clear the line and move to the next.
                            sys.stdout.write('\r\033[K')
                            sys.stdout.flush()
                            current_line_buffer = ""
                            has_printed_to_current_line = False
                            sys.stdout.write('\n') # Move to next line
                            sys.stdout.flush()
                        
                        last_audio_time = current_time 
                        last_partial_update_time = current_time
                        last_forced_flush_time = current_time # Reset forced flush on silence
                        _current_recognizer.Result() # Flush internal state (important for new utterances)


    except KeyboardInterrupt:
        _finish_event.set()

    # Return final state of transcripts and display variables for finalization
    return transcripts, current_line_buffer, has_printed_to_current_line, _current_recognizer


def _finalize_session(
    transcripts: list[str],
    base_dir: str,
    utter_dir: str,
    current_line_buffer: str,
    has_printed_to_current_line: bool,
    final_recognizer: vosk.KaldiRecognizer
):
    """
    Performs final cleanup and saves the complete transcript to disk.
    Handles any remaining text in the recognizer's buffer.
    """
    # Process any remaining text in the buffer as a final utterance
    final_trailing_raw = json.loads(final_recognizer.FinalResult()).get('text','').strip()
    final_trailing_cleaned = _clean_text(final_trailing_raw)
    # Apply _clean_partial_for_display here as well for consistency with screen output
    final_trailing_filtered = _clean_partial_for_display(final_trailing_cleaned)
    
    if final_trailing_filtered:
        if has_printed_to_current_line:
            sys.stdout.write('\r\033[K')
        sys.stdout.write(final_trailing_filtered + '\n')
        sys.stdout.flush()

        if final_trailing_filtered not in transcripts:
            transcripts.append(final_trailing_filtered)
            fn = datetime.now().strftime("utterance_final_%Y%m%d_%H%M%S.txt")
            with open(os.path.join(utter_dir, fn), 'w', encoding='utf-8') as f:
                f.write(final_trailing_filtered)
    elif has_printed_to_current_line: # If there was a partial but no final text after filtering
        sys.stdout.write('\r\033[K')
        sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()


    if transcripts:
        master_transcript_file = os.path.join(base_dir, 'transcript.txt')
        with open(master_transcript_file, 'w', encoding='utf-8') as mf:
            mf.write("\n\n".join(transcripts))
        print(f"\nFull transcript saved to: {master_transcript_file}")
    else:
        print("\nNo transcript recorded during this session.")

    print("\nExiting application.")


# --- Main Function ---
def main():
    """
    Main entry point for the voice input transcription script.
    Orchestrates configuration, model loading, audio processing, and session finalization.
    """
    # 1. Parse Command-line Arguments
    parser = argparse.ArgumentParser(
        description="Live mic transcription using Vosk with progressive enhancement.\n"
                    "By default, displays full utterances for clean output.\n"
                    "Use --immediate-display for real-time partial results (may have visual artifacts).\n"
                    "Output wraps naturally. Ctrl+Alt+Space to finish.")
    parser.add_argument("--config", default=CONFIG_FILE, help="Path to the configuration file.")
    parser.add_argument("--device-index", type=int, help="Specify audio input device index (overrides config).")
    parser.add_argument("--small-model-path", help="Path to the small Vosk model directory.")
    parser.add_argument("--large-model-path", help="Path to the large Vosk model directory (high accuracy).")
    parser.add_argument("--output-dir", help="Directory to save session transcripts.")
    parser.add_argument("--immediate-display", action="store_true",
                        help="Enable immediate display of partial results. Batch display is default.")
    parser.add_argument("--language-model", help="Specify the accent-specific language model to use (e.g., 'en-us', 'en-in'). This will dynamically select model paths unless explicitly provided.")
    args = parser.parse_args()

    # 2. Load Configuration from file or create it interactively
    cfg = _load_configuration_and_args(args.config, args)

    # Retrieve final determined values from cfg
    dev_idx = cfg['device_index']
    small_model_path = cfg['small_model_path']
    large_model_path = cfg['large_model_path']
    output_dir = cfg['output_dir']
    
    # DETERMINE DISPLAY MODE: batch_display_mode is TRUE by default.
    # It becomes FALSE only if --immediate-display is present.
    batch_display_mode = not args.immediate_display

    # 3. Final Audio Device Selection Check (This check is primarily handled proactively in _load_configuration_and_args now)
    try:
        # A final check to ensure the chosen device index is still valid just before stream opening
        sd.query_devices(dev_idx, 'input')
    except Exception as e:
        print(f"Error: Selected audio device index {dev_idx} is not valid for input: {e}", file=sys.stderr)
        sys.exit(1)

    # 4. Initialize Vosk Recognizers (Small immediately, Large in background thread)
    initial_recognizer, large_model_loaded_event, large_recognizer_container = \
        _initialize_vosk_recognizers(small_model_path, large_model_path, SAMPLE_RATE)

    # 5. Setup Session Directories
    session_base_dir, utterance_dir = _setup_session_directories(output_dir)
    print(f"Recording from device {dev_idx}. Session folder: {session_base_dir}")

    # 6. Start Hotkey Listener
    _start_hotkey_listener()

    transcripts = [] # List to accumulate all transcribed utterances

    # 7. Process Audio Stream (Main Transcription Loop)
    transcripts, current_line_buffer, has_printed_to_current_line, final_recognizer_used = \
        _process_audio_stream(
            dev_idx=dev_idx,
            sample_rate=SAMPLE_RATE,
            batch_display_mode=batch_display_mode,
            transcripts=transcripts,
            utter_dir=utterance_dir,
            initial_recognizer=initial_recognizer,
            large_model_loaded_event=large_model_loaded_event,
            large_recognizer_container=large_recognizer_container
        )

    # 8. Finalize Session (Save Transcript to Disk and perform final console cleanup)
    _finalize_session(
        transcripts=transcripts,
        base_dir=session_base_dir,
        utter_dir=utterance_dir,
        current_line_buffer=current_line_buffer,
        has_printed_to_current_line=has_printed_to_current_line,
        final_recognizer=final_recognizer_used
    )


if __name__=='__main__':
    main()