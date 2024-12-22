import torch
import torch.nn as nn
import torchaudio
import numpy as np
from jukebox.hparams import Hyperparams, setup_hparams
from jukebox.make_models import make_vqvae, make_prior, MODELS
from jukebox.sample import sample_partial_window, _sample
from jukebox.utils.dist_utils import setup_dist_from_mpi
from jukebox.utils.torch_utils import empty_cache
import librosa
import io
from pydub import AudioSegment

class BreakcoreGenerator:
    def __init__(self, device="cpu"):
        self.device = device
        self.sample_rate = 44100
        
        # Initialize Jukebox
        self.model = '1b_lyrics'  # using the smaller model for faster generation
        self.hps = Hyperparams()
        self.hps.sr = self.sample_rate
        self.hps.n_samples = 1
        self.hps.hop_fraction = 0.5
        
        # Load models
        self.vqvae, self.priors = self.load_models()
    
    def load_models(self):
        """Load the Jukebox models"""
        try:
            vqvae = make_vqvae(setup_hparams(MODELS[0][0], dict(sample_length=0)))
            prior = make_prior(setup_hparams(self.model, dict()), vqvae)
            
            # Move models to device
            vqvae.to(self.device)
            prior.to(self.device)
            
            return vqvae, [prior]
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise
    
    def generate_from_prompt(self, prompt: str, duration: int = 180):
        """Generate breakcore music from a text prompt using Jukebox"""
        try:
            if not prompt:
                raise ValueError("Prompt cannot be empty")
            
            # Set up generation parameters
            sample_tokens = duration * self.sample_rate // self.hps.hop_fraction
            conditioning = {
                "artist": "Breakcore",
                "genre": "Electronic",
                "lyrics": f"{prompt}\n" * 4,  # Repeat prompt for stronger conditioning
                "total_length": sample_tokens,
                "offset": 0
            }
            
            # Generate
            samples = sample_partial_window(
                self.priors,
                self.vqvae,
                [conditioning],
                sample_tokens,
                self.hps
            )
            
            # Convert to audio
            audio = samples.cpu().numpy().squeeze()
            
            # Normalize and add effects
            audio = self.apply_breakcore_effects(torch.from_numpy(audio))
            
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
                               tags={"artist": "Breakcore AI", "title": prompt[:30]})
            return buffer.getvalue()

        except Exception as e:
            print(f"Error generating music: {str(e)}")
            return self._fallback_generation(duration)
    
    def generate_from_reference(self, reference_features):
        """Generate breakcore music similar to a reference track"""
        try:
            if reference_features is None:
                raise ValueError("Reference features cannot be None")
            
            # Extract musical features from reference
            tempo, beat_frames = librosa.beat.beat_track(y=reference_features)
            
            # Use features to condition generation
            conditioning = {
                "artist": "Breakcore",
                "genre": "Electronic",
                "lyrics": f"Create intense breakcore with tempo {tempo} BPM\n" * 4,
                "total_length": len(reference_features),
                "offset": 0
            }
            
            # Generate
            samples = sample_partial_window(
                self.priors,
                self.vqvae,
                [conditioning],
                len(reference_features),
                self.hps
            )
            
            # Convert to audio
            audio = samples.cpu().numpy().squeeze()
            
            # Apply effects
            audio = self.apply_breakcore_effects(torch.from_numpy(audio))
            
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
                               tags={"artist": "Breakcore AI", "title": "Reference-based Generation"})
            return buffer.getvalue()

        except Exception as e:
            print(f"Error generating music: {str(e)}")
            return self._fallback_generation(30)
    
    def _fallback_generation(self, duration: int = 30):
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
    
    def apply_breakcore_effects(self, audio):
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
