import torch
import torchaudio
import numpy as np
from dotenv import load_dotenv
import os
import io
from pydub import AudioSegment
import openai
import time
from typing import Optional, List
import json
from pathlib import Path

class BreakcoreGenerator:
    def __init__(self, device="cpu"):
        self.device = device
        self.sample_rate = 44100
        self.training_data_dir = "training_data"
        
        # Create training data directory if it doesn't exist
        os.makedirs(self.training_data_dir, exist_ok=True)
        
        # Load environment variables
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
        
        # Initialize OpenAI client
        openai.api_key = self.api_key
    
    def prepare_training_data(self, audio_files: List[str], descriptions: List[str]) -> str:
        """Prepare training data from audio files and their descriptions"""
        try:
            if len(audio_files) != len(descriptions):
                raise ValueError("Number of audio files must match number of descriptions")
            
            training_data = []
            
            for audio_file, description in zip(audio_files, descriptions):
                # Load and process audio file
                audio_path = Path(audio_file)
                if not audio_path.exists():
                    raise FileNotFoundError(f"Audio file not found: {audio_file}")
                
                # Convert to required format if needed
                audio_segment = AudioSegment.from_file(audio_file)
                audio_segment = audio_segment.set_frame_rate(self.sample_rate)
                audio_segment = audio_segment.normalize()
                
                # Save processed audio
                processed_path = Path(self.training_data_dir) / f"processed_{audio_path.name}"
                audio_segment.export(processed_path, format="mp3", 
                                  parameters=["-q:a", "0", "-b:a", "320k"])
                
                # Create training example
                training_example = {
                    "audio": str(processed_path),
                    "prompt": f"Create breakcore music that sounds like this: {description}",
                    "metadata": {
                        "genre": "Breakcore",
                        "style": "Electronic",
                        "tempo": "Fast",
                        "description": description
                    }
                }
                training_data.append(training_example)
            
            # Save training data configuration
            config_path = Path(self.training_data_dir) / "training_config.json"
            with open(config_path, "w") as f:
                json.dump(training_data, f, indent=2)
            
            return str(config_path)
            
        except Exception as e:
            print(f"Error preparing training data: {str(e)}")
            raise
    
    def train_model(self, training_config_path: str, epochs: int = 1) -> str:
        """Fine-tune the model on breakcore music"""
        try:
            # Load training configuration
            with open(training_config_path, "r") as f:
                training_data = json.load(f)
            
            print(f"Starting fine-tuning with {len(training_data)} examples...")
            
            # Create fine-tuning job
            response = openai.fine_tuning.create(
                model="music-2",
                training_files=training_data,
                hyperparameters={
                    "n_epochs": epochs,
                    "batch_size": 1,  # Audio files are processed one at a time
                    "learning_rate_multiplier": 0.1  # Conservative learning rate
                }
            )
            
            job_id = response.id
            print(f"Fine-tuning job created: {job_id}")
            
            # Monitor training progress
            while True:
                status = openai.fine_tuning.retrieve(job_id)
                print(f"Status: {status.status}")
                
                if status.status == "succeeded":
                    print("Fine-tuning completed successfully!")
                    self.fine_tuned_model = status.fine_tuned_model
                    return self.fine_tuned_model
                elif status.status == "failed":
                    raise Exception(f"Fine-tuning failed: {status.error}")
                
                time.sleep(60)  # Check status every minute
                
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise
    
    def generate_from_prompt(self, prompt: str, duration: int = 180, use_fine_tuned: bool = True) -> bytes:
        """Generate breakcore music from a text prompt using OpenAI"""
        try:
            if not prompt:
                raise ValueError("Prompt cannot be empty")
            
            # Validate duration (max 5 minutes per generation)
            if duration <= 0 or duration > 300:
                raise ValueError("Duration must be between 1 and 300 seconds")
            
            # Split generation into chunks if duration is long
            chunk_duration = 60  # OpenAI's limit per request
            num_chunks = (duration + chunk_duration - 1) // chunk_duration
            audio_chunks = []
            
            # Select model
            model = self.fine_tuned_model if (use_fine_tuned and hasattr(self, 'fine_tuned_model')) else "music-2"
            
            for i in range(num_chunks):
                current_duration = min(chunk_duration, duration - i * chunk_duration)
                if current_duration <= 0:
                    break
                
                # Create detailed prompt for breakcore generation
                chunk_prompt = f"""Part {i+1}/{num_chunks} of: Create an intense breakcore track with the following elements:
                - Fast-paced drum breaks and aggressive rhythms
                - Glitch effects and digital distortion
                - Industrial and chaotic sound design
                - High energy and intense atmosphere
                - {prompt}
                Make it sound like a mix of ULTRAKILL's soundtrack and classic breakcore."""
                
                # Generate music using OpenAI
                response = openai.audio.generate(
                    model=model,
                    prompt=chunk_prompt,
                    duration=current_duration
                )
                
                # Convert response to AudioSegment
                audio_data = response.read()
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
                
                # Normalize audio
                audio_segment = audio_segment.normalize()
                audio_chunks.append(audio_segment)
            
            # Combine chunks if there are multiple
            if len(audio_chunks) > 1:
                combined_audio = audio_chunks[0]
                for chunk in audio_chunks[1:]:
                    combined_audio = combined_audio.append(chunk, crossfade=100)  # Add 100ms crossfade
                
                # Export to bytes with high quality
                buffer = io.BytesIO()
                combined_audio.export(buffer, format="mp3",
                                   parameters=["-q:a", "0", "-b:a", "320k"],
                                   tags={"artist": "Breakcore AI", "title": prompt[:30]})
                return buffer.getvalue()
            else:
                return audio_data
            
        except Exception as e:
            print(f"Error generating music: {str(e)}")
            return self._fallback_generation(duration)
    
    def generate_from_reference(self, reference_features: np.ndarray) -> bytes:
        """Generate breakcore music similar to a reference track"""
        try:
            if reference_features is None:
                raise ValueError("Reference features cannot be None")
            
            # Convert reference features to audio file
            temp_file = io.BytesIO()
            temp_segment = AudioSegment(
                reference_features.tobytes(),
                frame_rate=self.sample_rate,
                sample_width=2,
                channels=1
            )
            temp_segment.export(temp_file, format="mp3")
            temp_file.seek(0)
            
            # Generate music using OpenAI with reference
            response = openai.audio.generate(
                model="music-2",
                prompt="Create an intense breakcore track with fast drum breaks and glitch effects",
                reference_audio=temp_file,
                duration=min(300, len(reference_features) // self.sample_rate)  # Max 5 minutes
            )
            
            # Process the generated audio
            audio_data = response.read()
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
            audio_segment = audio_segment.normalize()
            
            # Export with high quality
            buffer = io.BytesIO()
            audio_segment.export(buffer, format="mp3",
                               parameters=["-q:a", "0", "-b:a", "320k"],
                               tags={"artist": "Breakcore AI", "title": "Reference-based Generation"})
            return buffer.getvalue()
            
        except Exception as e:
            print(f"Error generating music: {str(e)}")
            return self._fallback_generation(30)
    
    def _fallback_generation(self, duration: int = 30) -> bytes:
        """Fallback method for basic audio generation if model fails"""
        try:
            # Generate basic waveform
            samples = duration * self.sample_rate
            t = np.linspace(0, duration, samples)
            
            # Generate more complex base sound
            base_freq = 440
            audio = np.sin(2 * np.pi * base_freq * t)  # Base sine wave
            audio += 0.5 * np.sin(4 * np.pi * base_freq * t)  # First harmonic
            audio += 0.25 * np.sin(6 * np.pi * base_freq * t)  # Second harmonic
            
            # Normalize
            audio = audio / np.max(np.abs(audio))
            
            # Add effects
            audio = self.apply_breakcore_effects(torch.from_numpy(audio.astype(np.float32)))
            
            # Convert to MP3
            buffer = io.BytesIO()
            audio_segment = AudioSegment(
                audio.numpy().tobytes(),
                frame_rate=self.sample_rate,
                sample_width=2,
                channels=1
            )
            audio_segment = audio_segment.normalize()
            audio_segment.export(buffer, format="mp3",
                               parameters=["-q:a", "0", "-b:a", "320k"],
                               tags={"artist": "Breakcore AI", "title": "Fallback Generation"})
            return buffer.getvalue()
        except Exception as e:
            print(f"Error in fallback generation: {str(e)}")
            raise
    
    def apply_breakcore_effects(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply breakcore-style effects to the audio"""
        try:
            # Add distortion
            audio = torch.tanh(audio * 2)
            
            # Add random glitch effects
            glitch_mask = (torch.rand_like(audio) > 0.95)
            audio[glitch_mask] = audio[glitch_mask] * -1
            
            # Add compression
            audio = torch.sign(audio) * torch.log1p(torch.abs(audio) * 10)
            
            # Normalize
            audio = audio / torch.max(torch.abs(audio))
            
            return audio
        except Exception as e:
            print(f"Error applying effects: {str(e)}")
            return audio  # Return unmodified audio if effects fail
