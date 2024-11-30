import torch
import torch.nn as nn
import torchaudio
import numpy as np
import openai
from dotenv import load_dotenv
import os
import io
from pydub import AudioSegment

class BreakcoreGenerator:
    def __init__(self, device="cpu"):
        self.device = device
        self.sample_rate = 44100
        # Load environment variables
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment variables. Please set OPENAI_API_KEY in your .env file.")
        openai.api_key = self.api_key
    
    def generate_from_prompt(self, prompt: str, duration: int = 180):
        """Generate breakcore music from a text prompt using OpenAI"""
        try:
            if not prompt:
                raise ValueError("Prompt cannot be empty")
                
            # Validate duration
            if duration <= 0 or duration > 600:  # Max 10 minutes
                raise ValueError("Duration must be between 1 and 600 seconds")
                
            # Split generation into chunks if duration is long
            chunk_duration = 60  # OpenAI's limit per request
            num_chunks = (duration + chunk_duration - 1) // chunk_duration
            audio_chunks = []
            
            for i in range(num_chunks):
                current_duration = min(chunk_duration, duration - i * chunk_duration)
                if current_duration <= 0:
                    break
                    
                chunk_prompt = f"""Part {i+1}/{num_chunks} of: Create an intense breakcore track with the following characteristics:
                - Fast-paced drum breaks
                - Glitch effects and distortion
                - ULTRAKILL-style aggressive sound
                - {prompt}
                Make it chaotic but rhythmic, ensuring smooth transitions between parts."""

                # Generate music using OpenAI
                response = openai.audio.generate(
                    model="music-2",
                    prompt=chunk_prompt,
                    duration=current_duration
                )

                # Convert response to AudioSegment
                audio_data = response.audio.read()
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
            # Fallback to basic generation if API fails
            return self._fallback_generation(duration)
    
    def generate_from_reference(self, reference_features):
        """Generate breakcore music similar to a reference track"""
        try:
            if reference_features is None:
                raise ValueError("Reference features cannot be None")
                
            # Use reference features to guide the generation
            prompt = "Create a breakcore track similar to the reference, with intense drum breaks and glitch effects"
            
            # Generate music using OpenAI
            response = openai.audio.generate(
                model="music-2",
                prompt=prompt,
                reference_audio=reference_features
            )

            # Convert the response to audio data
            audio_data = response.audio.read()
            
            # Process the generated audio
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
            # Fallback to basic generation
            return self._fallback_generation(30)
    
    def _fallback_generation(self, duration: int = 30):
        """Fallback method for basic audio generation if API fails"""
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
            
            # Add some basic effects
            audio = self.apply_breakcore_effects(torch.from_numpy(audio.astype(np.float32)))
            
            # Convert to MP3 bytes with high quality
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
    
    def apply_breakcore_effects(self, audio):
        """Apply breakcore-style effects to the generated audio"""
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
