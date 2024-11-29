import streamlit as st
import os
from pathlib import Path
import torch
import numpy as np
from model.generator import BreakcoreGenerator
from audio.processor import AudioProcessor
import tempfile

# Page config
st.set_page_config(
    page_title="Breakcore AI Music Generator",
    page_icon="🎵",
    layout="wide"
)

# Initialize models and processors
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return BreakcoreGenerator(device=device), AudioProcessor()

generator, audio_processor = load_models()

# Custom styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1a1a1a 0%, #4a0000 100%);
    }
    .main {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🎵 Breakcore AI Music Generator")
st.markdown("""
Generate intense breakcore music inspired by ULTRAKILL's OST using AI! 
You can either:
- Enter a text prompt describing the music you want
- Upload an MP3 file to generate similar music
""")

# Create tabs for different generation methods
text_tab, audio_tab = st.tabs(["Generate from Text", "Generate from Audio"])

with text_tab:
    st.header("Generate from Text Prompt")
    prompt = st.text_area(
        "Enter your prompt",
        placeholder="Example: Create an intense breakcore track with heavy drum breaks and glitch effects",
        height=100
    )
    duration = st.slider("Duration (seconds)", min_value=10, max_value=60, value=30)
    
    if st.button("Generate from Text", type="primary"):
        with st.spinner("Generating music..."):
            try:
                # Generate music
                audio_data = generator.generate_from_prompt(prompt, duration)
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                    audio_processor.save_audio(audio_data, tmp_file.name)
                    
                    # Display audio player
                    st.audio(tmp_file.name)
                    
                    # Download button
                    with open(tmp_file.name, 'rb') as f:
                        st.download_button(
                            "Download Generated Track",
                            f,
                            file_name="generated_breakcore.mp3",
                            mime="audio/mpeg"
                        )
            except Exception as e:
                st.error(f"Error generating music: {str(e)}")

with audio_tab:
    st.header("Generate from Reference Audio")
    uploaded_file = st.file_uploader("Upload an MP3 file", type=['mp3'])
    
    if uploaded_file is not None:
        if st.button("Generate from Audio", type="primary"):
            with st.spinner("Generating music..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_input:
                        tmp_input.write(uploaded_file.read())
                        
                    # Process the audio file
                    reference_features = audio_processor.extract_features(tmp_input.name)
                    
                    # Generate similar music
                    audio_data = generator.generate_from_reference(reference_features)
                    
                    # Save generated audio
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_output:
                        audio_processor.save_audio(audio_data, tmp_output.name)
                        
                        # Display audio player
                        st.audio(tmp_output.name)
                        
                        # Download button
                        with open(tmp_output.name, 'rb') as f:
                            st.download_button(
                                "Download Generated Track",
                                f,
                                file_name="generated_breakcore.mp3",
                                mime="audio/mpeg"
                            )
                    
                    # Clean up
                    os.unlink(tmp_input.name)
                    os.unlink(tmp_output.name)
                except Exception as e:
                    st.error(f"Error generating music: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using OpenAI and Streamlit")
