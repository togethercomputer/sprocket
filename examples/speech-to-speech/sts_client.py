import base64
import json
import os
from queue import Queue
from typing import Any

import numpy as np
import requests
import sounddevice as sd

deployment_name = os.getenv("DEPLOYMENT_NAME", "speech-to-speech")
api_key = os.environ["TOGETHER_API_KEY"]
url = f"https://api.together.xyz/v1/deployment-request/{deployment_name}/generate"

# Recording parameters
SAMPLE_RATE = 16000  # Whisper works well with 16kHz
CHANNELS = 1  # Mono audio
BLOCK_SIZE = 4096  # Audio block size for streaming

# Global variables for recording
audio_queue: "Queue[np.ndarray]" = Queue()
recording = False


def audio_callback(
    indata: np.ndarray, frames: int, time: Any, status: sd.CallbackFlags
) -> None:
    """Callback function for audio input stream."""
    if status:
        print(f"Audio status: {status}")
    if recording:
        audio_queue.put(indata.copy())


def record_until_enter() -> None:
    """Record audio until user presses Enter."""
    global recording

    print("Starting recording... Speak now! Press Enter when finished.")

    # Start the input stream
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        callback=audio_callback,
        blocksize=BLOCK_SIZE,
        dtype=np.float32,
    ):
        recording = True
        # Wait for user to press Enter
        input()

    recording = False
    print("Recording finished. Processing...")


# Collect audio data
print("=== Speech-to-Speech Client ===")
record_until_enter()

# Collect all audio data from queue
audio_chunks = []
while not audio_queue.empty():
    audio_chunks.append(audio_queue.get())

# Concatenate all chunks
if audio_chunks:
    audio_data = np.concatenate(audio_chunks, axis=0)
    # Flatten audio data if needed
    if audio_data.ndim > 1:
        audio_data = audio_data.flatten()
else:
    print("!No audio data recorded!")
    exit(1)

print(f"Recorded {len(audio_data) / SAMPLE_RATE:.2f} seconds of audio")

# Prepare data for API
data = {
    "audio_data": base64.urlsafe_b64encode(audio_data).decode(),
    "sample_rate": SAMPLE_RATE,
}

response = requests.post(url, json=data, headers={"Authorization": f"Bearer {api_key}"})

print("Response status:", response.status_code)

try:
    # Parse the JSON response
    response_data = response.json()

    # Handle the audio data
    if "audio_data" in response_data and "sample_rate" in response_data:
        print("\nPlaying AI response...")

        # Decode the base64 audio data
        audio_bytes = base64.urlsafe_b64decode(response_data["audio_data"])

        # Convert bytes back to numpy array
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)

        # Get sample rate
        sample_rate = response_data["sample_rate"]

        print(f"Sample rate: {sample_rate} Hz")
        print(f" Duration: {len(audio_array) / sample_rate:.2f} seconds")

        # Play the audio response
        sd.play(audio_array, sample_rate)

        # Wait for playback to finish
        sd.wait()

        print("Playback finished!")
        print("\n=== Conversation Complete ===")
    else:
        print(f"No audio data found in response: {response_data}")

except json.JSONDecodeError:
    print("Failed to parse JSON response")
except Exception as e:
    print(f"Error processing response: {e}")
