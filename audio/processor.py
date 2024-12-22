import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import io

class AudioProcessor:
    def __init__(self):
        self.sample_rate = 44100
        self.hop_length = 512
        self.n_mels = 128
    
    def extract_features(self, audio_path):
        """Extract relevant features from an audio file"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # Ensure audio is the right length (max 8 minutes for Stable Audio)
            max_samples = 8 * 60 * self.sample_rate
            if len(y) > max_samples:
                y = y[:max_samples]
            
            # Normalize audio
            y = librosa.util.normalize(y)
            
            return y
            
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            raise
    
    def save_audio(self, audio_data: bytes, output_path: str):
        """Save audio data to file"""
        try:
            # Convert bytes to AudioSegment
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
            
            # Normalize
            audio_segment = audio_segment.normalize()
            
            # Export with high quality
            audio_segment.export(
                output_path,
                format='mp3',
                parameters=["-q:a", "0", "-b:a", "320k"],
                tags={"artist": "Breakcore AI"}
            )
        except Exception as e:
            print(f"Error saving audio: {str(e)}")
            raise
    
    def process_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Process audio data with breakcore effects"""
        try:
            # Normalize
            audio_data = librosa.util.normalize(audio_data)
            
            # Add effects
            audio_data = self.add_glitch_effects(audio_data)
            audio_data = self.add_distortion(audio_data)
            
            # Final normalization
            audio_data = librosa.util.normalize(audio_data)
            
            return audio_data
            
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            return audio_data
    
    def add_glitch_effects(self, audio: np.ndarray) -> np.ndarray:
        """Add glitch effects to audio"""
        try:
            # Random stutters
            stutter_points = np.random.choice(len(audio), size=5)
            for point in stutter_points:
                if point + 1000 < len(audio):
                    audio[point:point+1000] = np.repeat(audio[point:point+100], 10)
            
            # Random reverse segments
            reverse_points = np.random.choice(len(audio), size=3)
            for point in reverse_points:
                if point + 2000 < len(audio):
                    audio[point:point+2000] = audio[point:point+2000][::-1]
            
            return audio
        except Exception as e:
            print(f"Error adding glitch effects: {str(e)}")
            return audio
    
    def add_distortion(self, audio: np.ndarray) -> np.ndarray:
        """Add distortion effect"""
        try:
            # Soft clipping
            audio = np.tanh(audio * 2)
            
            # Bit crushing effect
            bits = 8
            audio = np.round(audio * (2**(bits-1))) / (2**(bits-1))
            
            return audio
        except Exception as e:
            print(f"Error adding distortion: {str(e)}")
            return audio
