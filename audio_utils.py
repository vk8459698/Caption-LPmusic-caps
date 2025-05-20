import io
import os
import subprocess
from typing import Tuple
import numpy as np
import soundfile as sf
import itertools
from numpy.fft import irfft

# Constants for audio processing
STR_CH_FIRST = 'channels_first'
STR_CH_LAST = 'channels_last'

def _resample_load_ffmpeg(path: str, sample_rate: int, downmix_to_mono: bool) -> Tuple[np.ndarray, int]:
    """
    Decoding, downmixing, and downsampling by ffmpeg.
    Returns a channel-first audio signal.
    
    Args:
        path: Audio file path
        sample_rate: Target sample rate
        downmix_to_mono: Whether to downmix to mono
        
    Returns:
        (audio signal, sample rate)
    """
    def _decode_resample_by_ffmpeg(filename, sr):
        """decode, downmix, and resample audio file"""
        channel_cmd = '-ac 1 ' if downmix_to_mono else ''  # downmixing option
        resampling_cmd = f'-ar {str(sr)}' if sr else ''  # downsampling option
        cmd = f"ffmpeg -i \"{filename}\" {channel_cmd} {resampling_cmd} -f wav -"
        p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        return out

    src, sr = sf.read(io.BytesIO(_decode_resample_by_ffmpeg(path, sr=sample_rate)))
    return src.T, sr

def _resample_load_librosa(path: str, sample_rate: int, downmix_to_mono: bool, **kwargs) -> Tuple[np.ndarray, int]:
    """
    Decoding, downmixing, and downsampling by librosa.
    Returns a channel-first audio signal.
    """
    import librosa
    src, sr = librosa.load(path, sr=sample_rate, mono=downmix_to_mono, **kwargs)
    return src, sr

def load_audio(
    path: str,
    ch_format: str,
    sample_rate: int = None,
    downmix_to_mono: bool = False,
    resample_by: str = 'ffmpeg',
    **kwargs,
) -> Tuple[np.ndarray, int]:
    """
    A wrapper for audio loading that handles different formats and resampling.
    
    Args:
        path: audio file path
        ch_format: one of 'channels_first' or 'channels_last'
        sample_rate: target sampling rate. if None, use the rate of the audio file
        downmix_to_mono: whether to downmix to mono
        resample_by: 'librosa' or 'ffmpeg'
        **kwargs: keyword args for librosa.load - offset, duration, dtype, res_type.
        
    Returns:
        (audio, sr) tuple
    """
    if ch_format not in (STR_CH_FIRST, STR_CH_LAST):
        raise ValueError(f'ch_format is wrong here -> {ch_format}')

    if os.stat(path).st_size > 8000:
        try:
            if resample_by == 'librosa':
                src, sr = _resample_load_librosa(path, sample_rate, downmix_to_mono, **kwargs)
            elif resample_by == 'ffmpeg':
                src, sr = _resample_load_ffmpeg(path, sample_rate, downmix_to_mono)
            else:
                raise NotImplementedError(f'resample_by: "{resample_by}" is not supported yet')
        except Exception as e:
            # Fallback to the other method if one fails
            print(f"Error with {resample_by}, trying alternative method: {str(e)}")
            if resample_by == 'librosa':
                src, sr = _resample_load_ffmpeg(path, sample_rate, downmix_to_mono)
            else:
                try:
                    src, sr = _resample_load_librosa(path, sample_rate, downmix_to_mono, **kwargs)
                except Exception:
                    raise ValueError(f"Could not load audio file: {path}")
    else:
        raise ValueError('Given audio is too short!')

    # Ensure correct format
    if src.ndim == 1:
        src = np.expand_dims(src, axis=0)
    
    # Convert to requested channel format
    if ch_format == STR_CH_FIRST:
        return src, sr
    else:
        return src.T, sr

# Utility functions for noise generation (can be useful for testing)
def ms(x):
    """Mean value of signal `x` squared."""
    return (np.abs(x)**2.0).mean()

def normalize(y, x=None):
    """Normalize power in y to a (standard normal) white noise signal."""
    if x is not None:
        x = ms(x)
    else:
        x = 1.0
    return y * np.sqrt(x / ms(y))

def white(N, state=None):
    """White noise generator."""
    state = np.random.RandomState() if state is None else state
    return state.randn(N)

def pink(N, state=None):
    """Pink noise generator."""
    state = np.random.RandomState() if state is None else state
    uneven = N % 2
    X = state.randn(N // 2 + 1 + uneven) + 1j * state.randn(N // 2 + 1 + uneven)
    S = np.sqrt(np.arange(len(X)) + 1.)  # +1 to avoid divide by zero
    y = (irfft(X / S)).real
    if uneven:
        y = y[:-1]
    return normalize(y)

def noise(N, color='white', state=None):
    """Generic noise generator."""
    if color == 'white':
        return white(N, state)
    elif color == 'pink':
        return pink(N, state)
    else:
        raise ValueError("Unsupported noise color")