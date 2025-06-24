import os
import base64
import traceback

import numpy as np
import torch
import sprocket
from chatterbox.tts import ChatterboxTTS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class StsSprocket(sprocket.Sprocket):
    def setup(self) -> None:
        """Initialize the Whisper and chat models using HuggingFace Transformers."""
        print("Initializing models...")

        # Check for GPU availability - fail if not available
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU is required but not available. "
                "Please ensure you have a CUDA-compatible GPU and PyTorch with CUDA support installed."
            )

        print("Initializing whisper model...")

        # Create Whisper ASR pipeline
        self.whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3-turbo",
            torch_dtype="auto",
            device_map="auto",
        )

        print("Initializing qwen model...")

        # Initialize Qwen model for text generation
        self.chat_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",  # Using a more standard Qwen model
            torch_dtype="auto",
            device_map="auto",
        )
        self.chat_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

        # Set generation parameters
        self.generation_config = {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_new_tokens": 1024,
            "do_sample": True,
        }

        print("Initializing chatterbox TTS model...")

        # Initialize Chatterbox TTS model for text-to-speech
        self.tts_model = ChatterboxTTS.from_pretrained(device="cuda")

        print("All models initialized successfully!")

    def stt(self, audio_data: np.ndarray, sample_rate: int = 16000) -> list[str]:
        """Transcribe audio using HuggingFace Transformers Whisper pipeline."""
        try:
            # Prepare the audio input
            audio_input = {"array": audio_data, "sampling_rate": sample_rate}

            # Generate transcription
            result = self.whisper_pipe(audio_input)

            # Extract the transcribed text
            transcription = (
                result.get("text", "") if isinstance(result, dict) else str(result)
            )

            return [transcription]

        except Exception as e:
            print(f"STT error: {e}")
            return [""]

    def chat(self, prompt: str) -> list[str]:
        """Generate chat response using HuggingFace Transformers."""
        try:
            # Prepare the input messages
            print(f"Received prompt: {prompt}")
            messages = [
                {
                    "role": "system",
                    "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                },
                {"role": "user", "content": prompt},
            ]
            # Apply chat template
            text = self.chat_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Tokenize input
            model_inputs = self.chat_tokenizer([text], return_tensors="pt").to(
                self.chat_model.device
            )

            # Generate response
            with torch.no_grad():
                generated_ids = self.chat_model.generate(
                    model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    **self.generation_config,
                )

            # Decode response
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = self.chat_tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            return [response]

        except Exception as e:
            print(f"Chat error: {e}")
            return [
                "I apologize, but I'm having trouble generating a response right now."
            ]

    def tts(self, text: str) -> np.ndarray:
        """Convert text to speech using Chatterbox TTS."""
        try:
            print(f"Converting text to speech: {text[:50]}...")

            # Generate audio using Chatterbox TTS
            wav = self.tts_model.generate(text)

            # Convert to numpy array and flatten if needed
            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()

            # Ensure it's a 1D array
            if wav.ndim > 1:
                wav = wav.flatten()

            return wav

        except Exception as e:
            print(f"TTS error: {e}")
            # Return silence as fallback
            return np.zeros(
                self.tts_model.sr * 2, dtype=np.float32
            )  # 2 seconds of silence

    def predict(self, args: dict) -> dict:
        try:
            # Extract audio data from input
            audio_data_b64 = args.get("audio_data")
            if not audio_data_b64:
                return {"error": "No audio_data provided"}

            sample_rate = args.get("sample_rate", 16000)

            # Decode base64 audio data
            audio_bytes = base64.urlsafe_b64decode(audio_data_b64)

            # Convert bytes to numpy array (assuming float32 format)
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32)

            transcriptions = self.stt(audio_array, sample_rate)

            chat_responses = self.chat("".join(transcriptions))

            # Convert the chat response to speech using Chatterbox TTS
            response_text = "".join(chat_responses)
            response_audio = self.tts(response_text)

            # Convert audio to base64 for transmission
            response_audio_bytes = response_audio.astype(np.float32).tobytes()
            response_audio_b64 = base64.urlsafe_b64encode(response_audio_bytes).decode(
                "utf-8"
            )

            return {
                "response": chat_responses,
                "audio_data": response_audio_b64,
                "sample_rate": self.tts_model.sr,
                "transcription": transcriptions,
            }

        except Exception as e:
            traceback.print_exc()
            return {"error": f"Transcription failed: {str(e)}"}


if __name__ == "__main__":
    queue_name = os.environ.get("TOGETHER_DEPLOYMENT_NAME", "speech-to-speech-test")
    sprocket.run(StsSprocket(), queue_name)
