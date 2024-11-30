import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
from pathlib import Path
import torch
import torchaudio
import librosa
import numpy as np
from pydub import AudioSegment
from model.generator import BreakcoreGenerator
from audio.processor import AudioProcessor
from fastapi.middleware.cors import CORSMiddleware

# Create the FastAPI app
app = FastAPI(title="Breakcore AI Generator")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create output and temp directories
os.makedirs("output", exist_ok=True)
os.makedirs("temp", exist_ok=True)

# Initialize models and processors
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    generator = BreakcoreGenerator(device=device)
    audio_processor = AudioProcessor()
except Exception as e:
    print(f"Error initializing models: {str(e)}")
    raise RuntimeError("Failed to initialize AI models. Please check if model files are present in the model directory.")

# Mount static files
static_dir = Path(__file__).parent / "web"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def read_root():
    return FileResponse(static_dir / "index.html")

@app.post("/generate/from-text")
async def generate_from_text(prompt: str = Form(...)):
    try:
        # Generate music from text prompt
        audio_data = generator.generate_from_prompt(prompt)
        
        # Save the generated audio
        output_path = "output/generated.mp3"
        audio_processor.save_audio(audio_data, output_path)
        
        return FileResponse(output_path, media_type="audio/mpeg")
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/generate/from-audio")
async def generate_from_audio(file: UploadFile = File(...)):
    temp_path = None
    try:
        # Save uploaded file temporarily
        temp_path = f"temp/{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the audio file
        reference_features = audio_processor.extract_features(temp_path)
        
        # Generate similar music
        audio_data = generator.generate_from_reference(reference_features)
        
        # Save the generated audio
        output_path = "output/generated.mp3"
        audio_processor.save_audio(audio_data, output_path)
        
        # Clean up
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        
        return FileResponse(output_path, media_type="audio/mpeg")
    except Exception as e:
        # Clean up on error
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing audio: {str(e)}"}
        )

# Add cleanup on shutdown
@app.on_event("shutdown")
async def cleanup():
    try:
        # Clean up temp and output directories
        for dir_path in ["temp", "output"]:
            if os.path.exists(dir_path):
                for file in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file)
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"Error removing file {file_path}: {str(e)}")
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    print("Starting Breakcore AI Generator...")
    print("Access the web interface at: http://127.0.0.1:8000")
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=True
    )
