"""HTTP smoke test against a running voice-asr server.

Usage:
  .venv/bin/python tests/smoke_test.py [http://127.0.0.1:8765] [--chunks-dir path/to/chunks]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import soundfile as sf


def post(base: str, data: bytes, filename: str = "audio.wav", ctype: str = "audio/wav", fields: dict | None = None):
    fields = {"model": "qwen3-asr-1.7b", "response_format": "json", **(fields or {})}
    boundary = f"----t{uuid.uuid4().hex}"
    body = bytearray()
    for key, val in fields.items():
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{val}\r\n'.encode()
    body += (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {ctype}\r\n\r\n"
    ).encode() + data + b"\r\n" + f"--{boundary}--\r\n".encode()
    req = Request(
        f"{base}/v1/audio/transcriptions",
        data=bytes(body),
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "Authorization": "Bearer local"},
    )
    t0 = time.perf_counter()
    with urlopen(req, timeout=120) as resp:
        payload = resp.read().decode("utf-8")
    dt = time.perf_counter() - t0
    return (json.loads(payload)["text"] if fields["response_format"] != "text" else payload), dt


def wav_bytes(audio: np.ndarray, sr: int = 16000, fmt: str = "WAV") -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format=fmt, **({"subtype": "PCM_16"} if fmt == "WAV" else {}))
    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url", nargs="?", default="http://127.0.0.1:8765")
    parser.add_argument("--chunks-dir", type=Path, default=None, help="Optional dir with raw_1..5.wav")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    with urlopen(f"{base}/health", timeout=10) as resp:
        print("health:", resp.read().decode("utf-8"))

    rng = np.random.default_rng(0)
    tone = (0.05 * np.sin(2 * np.pi * 440 * np.arange(16000 * 2) / 16000)).astype(np.float32)

    print("empty     :", repr(post(base, b"")[0]))
    print("0.2s      :", repr(post(base, wav_bytes((rng.standard_normal(3200) * 0.01).astype(np.float32)))[0]))
    print("silence 2s:", repr(post(base, wav_bytes(np.zeros(32000, dtype=np.float32)))[0]))
    print("ogg 2s    :", repr(post(base, wav_bytes(tone, fmt="OGG"), "a.ogg", "audio/ogg")[0]))
    stereo_48k = np.stack([np.repeat(tone, 3)] * 2, axis=1)
    print("48k stereo:", repr(post(base, wav_bytes(stereo_48k, sr=48000))[0]))
    print("text fmt  :", repr(post(base, wav_bytes(tone), fields={"response_format": "text"})[0]))

    if args.chunks_dir:
        for i in range(1, 6):
            p = args.chunks_dir / f"raw_{i}.wav"
            txt, dt = post(base, p.read_bytes())
            print(f"raw_{i}: {sf.info(p).duration:.1f}s -> {dt * 1000:.0f} ms | {txt}")

    entry = os.environ.get("VINPUT_ENTRY")
    if entry:
        pcm = (np.clip(tone, -1, 1) * 32767).astype("<i2").tobytes()
        env = {
            **os.environ,
            "VINPUT_ASR_API_KEY": "local",
            "VINPUT_ASR_BASE_URL": f"{base}/v1",
            "VINPUT_ASR_MODEL": "qwen3-asr-1.7b",
            "VINPUT_ASR_LANGUAGE": "zh",
        }
        proc = subprocess.run([sys.executable, entry], input=pcm, capture_output=True, env=env)
        print(f"vinput entry: rc={proc.returncode} stdout={proc.stdout.decode()!r} stderr={proc.stderr.decode()!r}")


if __name__ == "__main__":
    main()
