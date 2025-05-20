import os
import sys
import argparse
import torch
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import urllib.request
from collections import OrderedDict
from transformers import BartTokenizer

# Import local modules
from modules import AudioEncoder
from bart import BartCaptionModel
from audio_utils import load_audio, STR_CH_FIRST

def get_audio(audio_path, duration=10, target_sr=16000, save_chunks=True):
    """
    Load and process audio for captioning
    
    Args:
        audio_path: Path to audio file
        duration: Duration of each chunk in seconds
        target_sr: Target sample rate
        save_chunks: Whether to save audio chunks
        
    Returns:
        Processed audio tensor and paths to saved chunks
    """
    print(f"Loading audio from: {audio_path}")
    n_samples = int(duration * target_sr)
    
    try:
        audio, sr = load_audio(
            path=audio_path,
            ch_format=STR_CH_FIRST,
            sample_rate=target_sr,
            downmix_to_mono=True,
        )
    except Exception as e:
        print(f"Error with custom audio loader: {e}")
        try:
            # Fallback to librosa
            print("Trying fallback with librosa...")
            import librosa
            audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        except Exception as e2:
            try:
                # Second fallback to soundfile
                print("Trying fallback with soundfile...")
                audio, sr = sf.read(audio_path)
                if len(audio.shape) == 2:  # If stereo
                    audio = np.mean(audio, axis=1)  # Correct axis for stereo format in soundfile
                if sr != target_sr:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            except Exception as e3:
                print(f"Critical error loading audio: {e3}")
                raise

    # Create directory for saving chunks if it doesn't exist
    save_dir = "audio_chunks"
    if save_chunks and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Make sure audio is mono and handle shape correctly
    if len(audio.shape) == 2:
        audio = np.mean(audio, axis=0)
    
    # Ensure audio is float32
    audio = audio.astype(np.float32)
    
    # Print debug info
    print(f"Audio shape after processing: {audio.shape}")
    print(f"Audio min: {audio.min()}, max: {audio.max()}, dtype: {audio.dtype}")

    # Handle audio size
    input_size = int(n_samples)
    if len(audio.shape) == 1:  # Ensure we're working with the right dimension
        if audio.shape[0] < input_size:  # pad sequence
            pad = np.zeros(input_size, dtype=np.float32)
            pad[: audio.shape[0]] = audio
            audio = pad
    else:
        print(f"Warning: Unexpected audio shape: {audio.shape}")

    # Calculate number of chunks
    ceil = int(audio.shape[0] // n_samples)
    if ceil == 0:
        print("Warning: Audio too short for processing. Padding to required length.")
        ceil = 1
        audio = np.pad(audio, (0, n_samples - audio.shape[0]))

    # Save each chunk if requested
    saved_paths = []
    if save_chunks and ceil > 0:
        chunks = np.split(audio[:ceil * n_samples], ceil)
        for i, chunk in enumerate(chunks):
            chunk_path = os.path.join(save_dir, f"chunk_{i}.wav")
            sf.write(chunk_path, chunk, target_sr)
            saved_paths.append(chunk_path)
            print(f"Saved chunk {i} to {chunk_path}")

    # Convert to tensor for model
    audio_chunks = np.split(audio[:ceil * n_samples], ceil)
    audio_tensor = torch.from_numpy(np.stack(audio_chunks))
    
    # Print tensor info for debugging
    print(f"Final audio tensor shape: {audio_tensor.shape}")
    return audio_tensor, saved_paths if save_chunks else []

def download_model():
    """Download the model weights if needed"""
    model_path = "transfer.pth"
    if not os.path.exists(model_path):
        print("Downloading model weights...")
        urllib.request.urlretrieve(
            "https://huggingface.co/seungheondoh/lp-music-caps/resolve/main/transfer.pth",
            model_path
        )
        print(f"Downloaded to {model_path}")
    return model_path

def load_model(model_path, max_length=128):
    """Load the captioning model"""
    # Create model
    model = BartCaptionModel(max_length=max_length)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {model_path}")
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Handle state dict
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            name = k[7:]  # remove 'module.'
        else:
            name = k
        new_state_dict[name] = v
    
    # Load into model
    model.load_state_dict(new_state_dict)
    print("Model loaded successfully")
    
    # Setup device
    device = torch.device("cpu")  # Use CPU for reliability
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()
    
    return model, device

def caption_audio(model, audio_tensor, device, num_beams=5):
    """Generate captions for audio segments"""
    # Transfer to device
    audio_tensor = audio_tensor.to(device)
    
    # Use the model to generate captions
    print(f"Generating captions for {audio_tensor.shape[0]} audio segments...")
    print(f"Audio tensor shape: {audio_tensor.shape}, device: {audio_tensor.device}")
    
    # Process in smaller batches if needed
    batch_size = 1  # Process one at a time for reliability
    num_segments = audio_tensor.shape[0]
    all_captions = []
    
    with torch.no_grad():
        for i in range(0, num_segments, batch_size):
            end = min(i + batch_size, num_segments)
            batch = audio_tensor[i:end]
            print(f"Processing batch {i//batch_size + 1}/{(num_segments+batch_size-1)//batch_size}: segments {i}-{end-1}")
            
            try:
                captions = model.generate(
                    samples=batch,
                    num_beams=num_beams,
                )
                all_captions.extend(captions)
                print(f"Successfully generated captions for segments {i}-{end-1}")
            except Exception as e:
                print(f"Error during caption generation for batch {i//batch_size + 1}: {e}")
                import traceback
                traceback.print_exc()
                all_captions.extend(["Error generating caption"] * (end - i))
    
    return all_captions

def visualize_audio(audio_path):
    """Visualize the waveform of an audio file"""
    try:
        import librosa
        import librosa.display
        
        y, sr = librosa.load(audio_path)
        
        plt.figure(figsize=(12, 4))
        librosa.display.waveshow(y, sr=sr)
        plt.title('Waveform')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.savefig("waveform.png")
        plt.close()
        print(f"Waveform visualization saved to waveform.png")
    except Exception as e:
        print(f"Error visualizing audio: {e}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Music Captioning Tool")
    parser.add_argument('--audio', '-a', type=str, default="vocal+music.wav", 
                        help='Path to audio file')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum caption length')
    parser.add_argument('--num_beams', type=int, default=5,
                        help='Number of beams for generation')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Visualize audio waveform')
    parser.add_argument('--no_save_chunks', action='store_true',
                        help='Do not save audio chunks')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with more verbose output')
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not os.path.exists(args.audio):
        print(f"Audio file not found: {args.audio}")
        return

    # Download and load model
    model_path = download_model()
    model, device = load_model(model_path, max_length=args.max_length)
    
    # Process audio
    try:
        audio_tensor, saved_paths = get_audio(
            audio_path=args.audio, 
            save_chunks=not args.no_save_chunks
        )
        
        # Generate captions
        print("\n===== GENERATING CAPTIONS =====\n")
        captions = caption_audio(model, audio_tensor, device, num_beams=args.num_beams)
        
        # Print results and save to file
        print("\n===== GENERATED CAPTIONS =====\n")
        with open("captions.txt", "w") as f:
            for chunk, text in enumerate(captions):
                time = f"{chunk * 10}:00-{(chunk + 1) * 10}:00"
                chunk_path = saved_paths[chunk] if chunk < len(saved_paths) else "No saved file"
                
                output = f"Time segment: {time}\n"
                output += f"Audio chunk: {chunk_path}\n"
                output += f"Caption: {text}\n"
                output += "-" * 50 + "\n"
                
                print(output)
                f.write(output)
        
        print(f"Captions saved to captions.txt")
        
        # Visualize audio if requested
        if args.visualize:
            visualize_audio(args.audio)
            
    except Exception as e:
        print(f"Error processing audio: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()