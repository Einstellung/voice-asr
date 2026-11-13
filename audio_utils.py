"""Audio decoding and level normalization (no model dependencies)."""
from __future__ import annotations

import io

import numpy as np
import soundfile as sf
import soxr

TARGET_SR = 16000


def _to_mono_float(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data


def _decode_with_av(raw: bytes) -> tuple[np.ndarray, int]:
    import av  # bundled ffmpeg libs; handles mp3/m4a/webm/ogg/opus/...

    with av.open(io.BytesIO(raw)) as container:
        stream = next(s for s in container.streams if s.type == "audio")
        resampler = av.AudioResampler(format="flt", layout="mono", rate=TARGET_SR)
        parts = []
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                parts.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            parts.append(out.to_ndarray().reshape(-1))
    audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    return audio.astype(np.float32), TARGET_SR


def decode_to_16k_mono(raw: bytes) -> np.ndarray:
    """Decode any common audio container to 16 kHz mono float32 in [-1, 1]."""
    try:
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        data = _to_mono_float(data)
    except Exception:
        data, sr = _decode_with_av(raw)
    if sr != TARGET_SR and data.size:
        data = soxr.resample(data, sr, TARGET_SR).astype(np.float32)
    return data


def dbfs(x: float) -> float:
    return 20.0 * np.log10(max(x, 1e-10))


def voiced_stats(audio: np.ndarray, floor_dbfs: float = -50.0) -> tuple[float, int]:
    """RMS (dBFS) over samples louder than floor_dbfs, and how many such samples there are."""
    thr = 10 ** (floor_dbfs / 20.0)
    voiced = audio[np.abs(audio) > thr]
    if voiced.size == 0:
        return -200.0, 0
    return dbfs(float(np.sqrt(np.mean(voiced.astype(np.float64) ** 2)))), int(voiced.size)


def normalize_level(audio: np.ndarray, target_dbfs: float = -20.0, floor_dbfs: float = -50.0,
                    max_gain_db: float = 40.0, peak_limit: float = 0.99) -> tuple[np.ndarray, float, float]:
    """Scale so that the RMS of 'voiced' samples (> floor_dbfs) is ~target_dbfs.

    Returns (audio, rms_before_dbfs, applied_gain_db). Peaks are hard-clipped at peak_limit.
    """
    rms_db, n = voiced_stats(audio, floor_dbfs)
    if n == 0:
        return audio, rms_db, 0.0
    gain_db = min(target_dbfs - rms_db, max_gain_db)
    out = audio * (10 ** (gain_db / 20.0))
    np.clip(out, -peak_limit, peak_limit, out=out)
    return out.astype(np.float32), rms_db, gain_db
