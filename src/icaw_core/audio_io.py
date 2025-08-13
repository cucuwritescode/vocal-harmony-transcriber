# created by Facundo Franchino
"""
Audio I/O utilities for loading, normalising, and preprocessing audio files.
"""

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import librosa


def load_audio(
    path: Path | str,
    sr: int = 16000,
    mono: bool = True,
    normalise: bool = True,
    hp_cutoff: float = 80.0,
) -> Tuple[np.ndarray, int]:
    """
    Load and preprocess audio file.
    
    Args:
        path: Path to audio file
        sr: Target sample rate (Hz)
        mono: Convert to mono if True
        normalise: Normalise to -1 dBFS peak
        hp_cutoff: High-pass filter cutoff frequency (Hz)
        
    Returns:
        Tuple of (waveform, sample_rate)
    """
    # load audio
    y, orig_sr = librosa.load(path, sr=sr, mono=mono)
    
    # apply high-pass filter to remove rumble
    if hp_cutoff > 0:
        # simple butterworth high-pass
        from scipy import signal
        sos = signal.butter(4, hp_cutoff, btype='high', fs=sr, output='sos')
        y = signal.sosfilt(sos, y)
    
    # normalise to -1 dBFS peak
    if normalise and np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y)) * 0.891  # -1 dBFS = 0.891 in linear scale
    
    return y, sr


def save_audio(
    path: Path | str,
    audio: np.ndarray,
    sr: int,
) -> None:
    """
    Save audio to file.
    
    Args:
        path: Output path
        audio: Audio waveform
        sr: Sample rate
    """
    import soundfile as sf
    sf.write(path, audio, sr)