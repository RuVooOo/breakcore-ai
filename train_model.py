import argparse
from pathlib import Path
from model.generator import BreakcoreGenerator
import json

def main():
    parser = argparse.ArgumentParser(description="Train the Breakcore AI model on your music")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing training audio files")
    parser.add_argument("--descriptions-file", type=str, required=True, help="JSON file containing descriptions for each audio file")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    args = parser.parse_args()
    
    try:
        # Load descriptions
        with open(args.descriptions_file, "r") as f:
            descriptions_data = json.load(f)
        
        # Get audio files
        audio_files = list(Path(args.data_dir).glob("*.mp3"))
        if not audio_files:
            raise ValueError(f"No MP3 files found in {args.data_dir}")
        
        # Match audio files with descriptions
        audio_paths = []
        descriptions = []
        for audio_file in audio_files:
            if audio_file.stem in descriptions_data:
                audio_paths.append(str(audio_file))
                descriptions.append(descriptions_data[audio_file.stem])
            else:
                print(f"Warning: No description found for {audio_file.name}, skipping...")
        
        if not audio_paths:
            raise ValueError("No valid audio files with descriptions found")
        
        print(f"Found {len(audio_paths)} audio files with descriptions")
        
        # Initialize generator
        generator = BreakcoreGenerator()
        
        # Prepare training data
        print("Preparing training data...")
        config_path = generator.prepare_training_data(audio_paths, descriptions)
        print(f"Training data prepared and saved to {config_path}")
        
        # Start training
        print(f"Starting training for {args.epochs} epochs...")
        fine_tuned_model = generator.train_model(config_path, args.epochs)
        print(f"Training completed! Fine-tuned model: {fine_tuned_model}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 