import torch
import torch.nn as nn
import torchaudio
import numpy as np
import openai
from dotenv import load_dotenv
import os

class BreakcoreGenerator:
    def __init__(self, device="cpu"):
        self.device = device
        self.sample_rate = 44100
        # Load environment variables
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
    
    def generate_from_prompt(self, prompt: str, duration: int = 30):
        """Generate breakcore music from a text prompt using OpenAI"""
        try:
            # Create a music generation prompt that emphasizes breakcore style
            enhanced_prompt = f"""Create an intense breakcore track with the following characteristics:
            - Fast-paced drum breaks
            - Glitch effects and distortion
            - ULTRAKILL-style aggressive sound
            - {prompt}
            Make it chaotic but rhythmic."""

            # Generate music using OpenAI
            response = openai.audio.generate(
                model="music-2",  # Use the latest music model
                prompt=enhanced_prompt,
                duration=duration
            )

            # Convert the response to numpy array
            audio_data = response.audio.read()
            return audio_data

        except Exception as e:
            print(f"Error generating music: {str(e)}")
            # Fallback to basic generation if API fails
            return self._fallback_generation(duration)
    
    def generate_from_reference(self, reference_features):
        """Generate breakcore music similar to a reference track"""
        try:
            # Use reference features to guide the generation
            prompt = "Create a breakcore track similar to the reference, with intense drum breaks and glitch effects"
            
            # Generate music using OpenAI
            response = openai.audio.generate(
                model="music-2",
                prompt=prompt,
                reference_audio=reference_features
            )

            # Convert the response to numpy array
            audio_data = response.audio.read()
            return audio_data

        except Exception as e:
            print(f"Error generating music: {str(e)}")
            # Fallback to basic generation
            return self._fallback_generation(30)
    
    def _fallback_generation(self, duration: int = 30):
        """Fallback method for basic audio generation if API fails"""
        # Generate basic waveform
        samples = duration * self.sample_rate
        t = np.linspace(0, duration, samples)
        
        # Generate base frequencies
        base_freq = 440
        audio = np.sin(2 * np.pi * base_freq * t)
        
        # Add some basic effects
        audio = self.apply_breakcore_effects(torch.from_numpy(audio))
        return audio.numpy()
    
    def apply_breakcore_effects(self, audio):
        """Apply breakcore-style effects to the generated audio"""
        # Add distortion
        audio = torch.tanh(audio * 2)
        
        # Add random glitch effects
        glitch_mask = (torch.rand_like(audio) > 0.95)
        audio[glitch_mask] = audio[glitch_mask] * -1
        
        # Add compression
        audio = torch.sign(audio) * torch.log1p(torch.abs(audio) * 10)
        
        return audio
