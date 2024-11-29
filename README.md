# Breakcore AI Music Generator

An AI-powered music generation system specialized in creating breakcore music inspired by the ULTRAKILL OST. This project uses deep learning to generate intense, rhythmic breakcore tracks either from text prompts or by analyzing existing MP3 files.

## Features

- Generate breakcore music from text prompts
- Analyze and create music similar to provided MP3 files
- ULTRAKILL-inspired sound design
- High-quality audio synthesis
- Easy-to-use API interface

## Installation

1. Clone this repository
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the server:
```bash
python main.py
```

2. Access the web interface at http://localhost:8000

3. Either:
   - Upload an MP3 file to generate similar music
   - Enter a text prompt to generate music from scratch

## Project Structure

- `main.py`: Main application entry point
- `model/`: AI model architecture and training code
- `audio/`: Audio processing utilities
- `api/`: FastAPI routes and endpoints
- `web/`: Web interface files

## License

MIT License
