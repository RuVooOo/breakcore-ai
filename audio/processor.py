import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment

class AudioProcessor:
    def __init__(self):
        self.sample_rate = 44100
        self.hop_length = 512
        self.n_mels = 128
    
    def extract_features(self, audio_path):
        """Extract relevant features from an audio file"""
        # Load audio file
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Extract features
        features = {}
        
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr,
            n_mels=self.n_mels,
            hop_length=self.hop_length
        )
        features['mel_spec'] = librosa.power_to_db(mel_spec)
        
        # Tempo and beat features
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo
        features['beat_frames'] = beat_frames
        
        # Onset detection
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        features['onset_env'] = onset_env
        
        # Chromagram
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        features['chroma'] = chroma
        
        return features
    
    def save_audio(self, audio_data, output_path):
        """Save audio data to file"""
        # Normalize audio
        audio_data = np.clip(audio_data, -1, 1)
        
        # Save as WAV first
        temp_wav = output_path.replace('.mp3', '.wav')
        sf.write(temp_wav, audio_data, self.sample_rate)
        
        # Convert to MP3
        audio = AudioSegment.from_wav(temp_wav)
        audio.export(output_path, format='mp3', bitrate='320k')
    
    def apply_breakcore_processing(self, audio_data):
        """Apply breakcore-specific audio processing"""
        # Add drum breaks
        drum_pattern = self.generate_drum_pattern()
        audio_data = self.mix_audio(audio_data, drum_pattern)
        
        # Add glitch effects
        audio_data = self.add_glitch_effects(audio_data)
        
        # Add distortion
        audio_data = self.add_distortion(audio_data)
        
        return audio_data
    
    def generate_drum_pattern(self):
        """Generate a breakcore drum pattern"""
        # Placeholder for drum pattern generation
        return np.random.randn(44100)  # 1 second of noise
    
    def mix_audio(self, audio1, audio2, ratio=0.7):
        """Mix two audio signals"""
        return audio1 * ratio + audio2 * (1 - ratio)
    
    def add_glitch_effects(self, audio):
        """Add glitch effects to audio"""
        # Random stutters
        stutter_points = np.random.choice(len(audio), size=5)
        for point in stutter_points:
            if point + 1000 < len(audio):
                audio[point:point+1000] = np.repeat(audio[point:point+100], 10)
        return audio
    
    def add_distortion(self, audio):
        """Add distortion effect"""
        return np.clip(audio * 2, -1, 1)
