"""Offline unit tests for audio_utils.py."""
from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audio_utils import TARGET_SR, decode_to_16k_mono, normalize_level, voiced_stats


class AudioUtilsTests(unittest.TestCase):
    def test_decode_resamples_stereo_48k_to_mono_16k(self) -> None:
        t = np.arange(48000, dtype=np.float32) / 48000.0
        mono = 0.2 * np.sin(2 * np.pi * 440.0 * t)
        stereo = np.stack([mono, mono], axis=1)
        buf = io.BytesIO()
        sf.write(buf, stereo, 48000, format="WAV", subtype="PCM_16")
        decoded = decode_to_16k_mono(buf.getvalue())
        self.assertEqual(decoded.dtype, np.float32)
        self.assertAlmostEqual(decoded.size, TARGET_SR, delta=16)

    def test_voiced_stats_and_normalize_quiet_signal(self) -> None:
        t = np.arange(TARGET_SR, dtype=np.float32) / TARGET_SR
        quiet = (0.01 * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32)
        rms_before, count = voiced_stats(quiet, floor_dbfs=-50.0)
        self.assertGreater(count, TARGET_SR // 2)
        self.assertLess(rms_before, -35.0)
        norm, _, gain_db = normalize_level(quiet, target_dbfs=-20.0, floor_dbfs=-50.0)
        self.assertGreater(gain_db, 10.0)
        rms_after, _ = voiced_stats(norm, floor_dbfs=-50.0)
        self.assertGreater(rms_after, rms_before)
        self.assertAlmostEqual(rms_after, -20.0, delta=1.5)

    def test_silence_returns_zero_count(self) -> None:
        silence = np.zeros(TARGET_SR, dtype=np.float32)
        rms_db, count = voiced_stats(silence, floor_dbfs=-50.0)
        self.assertEqual(count, 0)
        self.assertEqual(rms_db, -200.0)


if __name__ == "__main__":
    unittest.main()
