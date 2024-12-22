import streamlit as st
import os
from pathlib import Path
import torch
from model.generator import BreakcoreGenerator
import tempfile
import io

# Page config
st.set_page_config(
    page_title="Breakcore AI Trainer",
    page_icon="🎵",
    layout="wide"
)

# Initialize model
@st.cache_resource
def load_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return BreakcoreGenerator(device=device)
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        return None

generator = load_model()

if generator is None:
    st.error("Failed to initialize the application. Please check the error messages above.")
    st.stop()

# Custom styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1a1a1a 0%, #4a0000 100%);
    }
    .main {
        color: white;
    }
    .uploadedFile {
        background-color: #2a2a2a;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .meter-container {
        background: #2a2a2a;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    }
    .optimal-badge {
        background: #00aa00;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        text-align: center;
        margin: 10px 0;
    }
    .not-optimal-badge {
        background: #aa0000;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        text-align: center;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🎵 Breakcore AI Trainer")
st.markdown("""
Train your own Breakcore AI by uploading MP3 samples! 
1. Drop your breakcore MP3 files below
2. Add descriptions for each track
3. Train the model until optimal
""")

# Initialize session state for tracking files
if 'trained_files' not in st.session_state:
    st.session_state.trained_files = []
if 'total_training_time' not in st.session_state:
    st.session_state.total_training_time = 0

# Training Progress Meter
st.markdown("<div class='meter-container'>", unsafe_allow_html=True)
st.subheader("Training Progress")

# Calculate progress (minimum 10 files for optimal)
OPTIMAL_FILES = 10
current_files = len(st.session_state.trained_files)
progress = min(current_files / OPTIMAL_FILES * 100, 100)

# Display progress bar
progress_bar = st.progress(int(progress))

# Display status
if progress >= 100:
    st.markdown("<div class='optimal-badge'>AI Model Optimal - Ready for Use!</div>", unsafe_allow_html=True)
else:
    files_needed = OPTIMAL_FILES - current_files
    st.markdown(f"<div class='not-optimal-badge'>Model Not Optimal - Please Upload {files_needed} More Files</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# File uploader for multiple MP3s
uploaded_files = st.file_uploader(
    "Drop your breakcore MP3 files here",
    type=['mp3'],
    accept_multiple_files=True
)

if uploaded_files:
    st.write(f"Processing {len(uploaded_files)} new files")
    
    # Create a form for descriptions
    with st.form("training_form"):
        descriptions = {}
        
        for file in uploaded_files:
            if file.name not in st.session_state.trained_files:
                st.markdown(f"<div class='uploadedFile'>", unsafe_allow_html=True)
                st.audio(file, format='audio/mp3')
                descriptions[file.name] = st.text_area(
                    f"Description for {file.name}",
                    placeholder="Describe the style, elements, and characteristics of this track",
                    key=f"desc_{file.name}"
                )
                st.markdown("</div>", unsafe_allow_html=True)
        
        train_button = st.form_submit_button("Train on These Files")
        
        if train_button and all(descriptions.values()):
            try:
                with st.spinner("Processing audio files..."):
                    # Save uploaded files temporarily
                    temp_dir = Path("temp_training")
                    temp_dir.mkdir(exist_ok=True)
                    
                    audio_paths = []
                    desc_list = []
                    
                    for file in uploaded_files:
                        if file.name not in st.session_state.trained_files:
                            temp_path = temp_dir / file.name
                            with open(temp_path, "wb") as f:
                                f.write(file.getvalue())
                            audio_paths.append(str(temp_path))
                            desc_list.append(descriptions[file.name])
                    
                    if audio_paths:
                        # Prepare and start training
                        st.info("Preparing training data...")
                        config_path = generator.prepare_training_data(audio_paths, desc_list)
                        
                        st.info("Training on new files... This might take a while.")
                        progress_bar = st.progress(0)
                        
                        def progress_callback(epoch, total_epochs, status):
                            progress = (epoch / total_epochs) * 100
                            progress_bar.progress(int(progress))
                            st.write(f"Epoch {epoch}/{total_epochs}: {status}")
                        
                        model_id = generator.train_model(
                            config_path,
                            epochs=3,  # Fixed epochs for consistency
                            progress_callback=progress_callback
                        )
                        
                        # Update trained files list
                        for file in uploaded_files:
                            if file.name not in st.session_state.trained_files:
                                st.session_state.trained_files.append(file.name)
                        
                        # Clean up
                        for path in audio_paths:
                            os.remove(path)
                        temp_dir.rmdir()
                        
                        st.success(f"Training completed! Model ID: {model_id}")
                        
                        # Force rerun to update progress meter
                        st.experimental_rerun()
                    
            except Exception as e:
                st.error(f"Error during training: {str(e)}")
        elif train_button:
            st.warning("Please provide descriptions for all tracks.")

# Generate button (only show if model is optimal)
if progress >= 100:
    st.markdown("---")
    if st.button("Generate Breakcore", type="primary"):
        st.info("Generation feature coming soon! Model is ready for use.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using OpenAI")

# Display training stats
st.sidebar.header("Training Stats")
st.sidebar.write(f"Files Trained: {current_files}/{OPTIMAL_FILES}")
st.sidebar.write(f"Progress: {int(progress)}%")
if st.session_state.trained_files:
    st.sidebar.write("Trained Files:")
    for file in st.session_state.trained_files:
        st.sidebar.write(f"- {file}")
