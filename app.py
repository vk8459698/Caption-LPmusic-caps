import os
import sys
import torch
import numpy as np
import soundfile as sf
import urllib.request
import argparse
from collections import OrderedDict

# Import local modules - make sure these are in the same directory
from modules import AudioEncoder
from bart import BartCaptionModel
from audio_utils import load_audio, STR_CH_FIRST

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()
    
    return model, device

def get_audio(audio_path, duration=10, target_sr=16000):
    """
    Load and process audio for captioning
    
    Args:
        audio_path: Path to audio file
        duration: Duration of each chunk in seconds
        target_sr: Target sample rate
        
    Returns:
        Processed audio tensor
    """
    print(f"Loading audio from: {audio_path}")
    n_samples = int(duration * target_sr)
    
    try:
        # First try soundfile - works well with WAV
        try:
            print("Trying with soundfile...")
            audio, sr = sf.read(audio_path)
            if len(audio.shape) == 2:  # If stereo
                audio = np.mean(audio, axis=1)
            if sr != target_sr:
                # Need librosa for resampling
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                
        except Exception as e1:
            print(f"Error with soundfile: {e1}")
            # Try librosa - handles more formats
            try:
                print("Trying with librosa...")
                import librosa
                audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
                
            except Exception as e2:
                print(f"Error with librosa: {e2}")
                # Last resort - use the original loader
                print("Trying with custom audio loader...")
                audio, sr = load_audio(
                    path=audio_path,
                    ch_format=STR_CH_FIRST,
                    sample_rate=target_sr,
                    downmix_to_mono=True,
                )
                
    except Exception as e:
        print(f"Critical error loading audio: {e}")
        raise

    # Make sure audio is mono and handle shape correctly
    if len(audio.shape) == 2:
        audio = np.mean(audio, axis=0)
    
    # Ensure audio is float32
    audio = audio.astype(np.float32)
    
    # Debug info
    print(f"Audio shape: {audio.shape}, min: {audio.min()}, max: {audio.max()}")
    
    # Calculate number of chunks
    audio_len = audio.shape[0]
    num_chunks = max(1, int(np.ceil(audio_len / n_samples)))
    print(f"Creating {num_chunks} chunks of {n_samples} samples each")
    
    # Process audio into chunks
    chunks = []
    for i in range(num_chunks):
        start_idx = i * n_samples
        end_idx = min(start_idx + n_samples, audio_len)
        chunk = audio[start_idx:end_idx]
        
        # Pad if necessary
        if chunk.shape[0] < n_samples:
            chunk = np.pad(chunk, (0, n_samples - chunk.shape[0]))
            
        chunks.append(chunk)
    
    # Convert to tensor for model
    audio_tensor = torch.from_numpy(np.stack(chunks))
    print(f"Final audio tensor shape: {audio_tensor.shape}")
    
    return audio_tensor

def caption_audio(model, audio_tensor, device, num_beams=5):
    """Generate captions for audio segments"""
    # Transfer to device
    audio_tensor = audio_tensor.to(device)
    
    # Process in smaller batches if needed
    batch_size = 4  # Adjust based on your memory constraints
    num_segments = audio_tensor.shape[0]
    all_captions = []
    
    with torch.no_grad():
        for i in range(0, num_segments, batch_size):
            end = min(i + batch_size, num_segments)
            batch = audio_tensor[i:end]
            
            try:
                captions = model.generate(
                    samples=batch,
                    num_beams=num_beams,
                )
                all_captions.extend(captions)
                # Show progress
                for j, caption in enumerate(captions):
                    print(f"  Segment {i+j+1}: {caption[:50]}...")
            except Exception as e:
                print(f"Error during caption generation: {e}")
                import traceback
                traceback.print_exc()
                all_captions.extend(["Error generating caption"] * (end - i))
    
    return all_captions

def combine_captions(captions):
    """Combine multiple captions into a single summary"""
    # Remove duplicates while preserving order
    unique_captions = []
    for caption in captions:
        if caption not in unique_captions and caption != "Error generating caption":
            unique_captions.append(caption)
    
    # If we have no valid captions, return error message
    if not unique_captions:
        return "Could not generate caption for this audio."
    
    # If we have just one caption, return it
    if len(unique_captions) == 1:
        return unique_captions[0]
    
    # Combine all captions into one text
    all_text = " ".join(unique_captions)
    
    # If the combined text is short enough, return it directly
    if len(all_text) < 200:
        return all_text
    
    # For longer text, try to create a better summary
    try:
        # Simple method - no NLTK required
        # Join the first part of each caption
        summary_parts = []
        for caption in unique_captions[:3]:  # Use up to 3 captions
            # Get first sentence or first part of caption
            first_part = caption.split('.')[0] + '.'
            if first_part not in summary_parts:
                summary_parts.append(first_part)
        
        if summary_parts:
            return " ".join(summary_parts)
        else:
            # Fallback to truncated version if needed
            return all_text[:200] + "..."
            
    except Exception as e:
        print(f"Warning: Error in caption combining: {e}")
        # Fallback - just return combined captions with length limit
        return all_text[:300] + "..." if len(all_text) > 300 else all_text

def summarize_audio(audio_path):
    """Process audio file and return a single caption summary"""
    try:
        # Download and load model
        model_path = download_model()
        model, device = load_model(model_path)
        
        # Process audio
        audio_tensor = get_audio(audio_path)
        
        # Generate captions
        print("\nGenerating captions...")
        captions = caption_audio(model, audio_tensor, device)
        
        # Combine captions into a single summary
        print("\nCombining captions into a summary...")
        summary = combine_captions(captions)
        
        return summary
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error processing audio: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Audio Summarization Tool")
    parser.add_argument("audio_path", help="Path to the audio file to summarize")
    parser.add_argument("--output", "-o", help="Optional path to save the caption to a file")
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_path):
        print(f"Error: Audio file not found: {args.audio_path}")
        sys.exit(1)
        
    print("AUDIO SUMMARIZATION TOOL")
    print(f"Processing audio: {args.audio_path}")
    
    summary = summarize_audio(args.audio_path)
    
    print("GENERATED CAPTION:")
    print(summary)
    
    # Save to file if specified
    if args.output:
        try:
            with open(args.output, 'w') as f:
                f.write(summary)
            print(f"\nCaption saved to: {args.output}")
        except Exception as e:
            print(f"Error saving caption to file: {e}")

if __name__ == "__main__":
    main()
