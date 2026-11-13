"""Local OpenAI-compatible ASR service backed by Qwen3-ASR-1.7B (transformers native).

Endpoint used by fcitx5-vinput's `provider.openai-compatible.batch` script:
    POST /v1/audio/transcriptions   multipart: file, model, response_format, [language], [prompt], [temperature]

Run:  .venv/bin/python app.py      (all settings via VOICE_ASR_* env vars, see README.md)
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from audio_utils import TARGET_SR, decode_to_16k_mono, normalize_level, voiced_stats

HERE = Path(__file__).resolve().parent


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    host = _env("VOICE_ASR_HOST", "127.0.0.1")
    port = int(_env("VOICE_ASR_PORT", "8765"))
    model_id = _env("VOICE_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B-hf")
    model_revision = _env("VOICE_ASR_MODEL_REVISION", "bcd2b5b7f32b480ab5790554cfa8347f246a14f3")
    device = _env("VOICE_ASR_DEVICE", "cuda:0")
    language = _env("VOICE_ASR_LANGUAGE", "Chinese")
    context_file = Path(_env("VOICE_ASR_CONTEXT_FILE", str(HERE / "context.txt")))
    save_corpus = _env_bool("VOICE_ASR_SAVE_CORPUS", True)
    corpus_dir = Path(_env("VOICE_ASR_CORPUS_DIR", str(HERE / "corpus")))
    corpus_format = _env("VOICE_ASR_CORPUS_FORMAT", "flac")  # flac | wav
    target_dbfs = float(_env("VOICE_ASR_TARGET_DBFS", "-20"))
    floor_dbfs = float(_env("VOICE_ASR_FLOOR_DBFS", "-50"))
    min_duration_s = float(_env("VOICE_ASR_MIN_DURATION", "0.3"))
    min_voiced_s = float(_env("VOICE_ASR_MIN_VOICED", "0.1"))
    max_new_tokens = int(_env("VOICE_ASR_MAX_NEW_TOKENS", "1024"))
    warmup = _env_bool("VOICE_ASR_WARMUP", True)


S = Settings()
log = logging.getLogger("voice-asr")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# OpenAI-style ISO codes -> Qwen3-ASR language names
LANG_MAP = {"zh": "Chinese", "zh-cn": "Chinese", "cmn": "Chinese", "en": "English", "yue": "Cantonese",
            "ja": "Japanese", "ko": "Korean", "auto": ""}


class ContextFile:
    """Term/context prompt file, re-read whenever its mtime changes. Lines starting with '#' are ignored."""

    def __init__(self, path: Path):
        self.path, self._mtime, self._text = path, None, ""

    def get(self) -> str:
        try:
            m = self.path.stat().st_mtime_ns
        except FileNotFoundError:
            self._mtime, self._text = None, ""
            return ""
        if m != self._mtime:
            lines = [ln.strip() for ln in self.path.read_text(encoding="utf-8").splitlines()]
            self._text = " ".join(ln for ln in lines if ln and not ln.startswith("#"))
            self._mtime = m
            log.info("context reloaded from %s: %r", self.path, self._text)
        return self._text


class Engine:
    def __init__(self):
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.torch = torch
        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(S.model_id, revision=S.model_revision)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            S.model_id, revision=S.model_revision, dtype=torch.bfloat16, device_map=S.device)
        self.model.eval()
        self.lock = threading.Lock()
        log.info("model %s loaded in %.1fs", S.model_id, time.time() - t0)

    def transcribe(self, audio: np.ndarray, language: str, prompt: str) -> str:
        torch = self.torch
        kw = {"audio": audio}
        if language:
            kw["language"] = language
        if prompt:
            kw["prompt"] = prompt
        with self.lock, torch.inference_mode():
            inputs = self.processor.apply_transcription_request(**kw).to(self.model.device, self.model.dtype)
            out = self.model.generate(**inputs, max_new_tokens=S.max_new_tokens, do_sample=False)
            gen = out[:, inputs["input_ids"].shape[1]:]
            text = self.processor.decode(gen, return_format="transcription_only")[0]
        return text.strip()


context = ContextFile(S.context_file)
engine: Engine | None = None
corpus_lock = threading.Lock()


def _load_engine() -> None:
    global engine
    engine = Engine()
    if S.warmup:
        t = time.time()
        rng = np.random.default_rng(0)
        engine.transcribe((rng.standard_normal(TARGET_SR) * 0.01).astype(np.float32), S.language, "")
        log.info("warm-up done in %.2fs", time.time() - t)
    context.get()


@asynccontextmanager
async def lifespan(_app):
    _load_engine()
    yield


app = FastAPI(title="voice-asr", version="0.1.0", lifespan=lifespan)


def _save_corpus(req_id: str, now: dt.datetime, audio: np.ndarray, record: dict) -> None:
    day = S.corpus_dir / now.strftime("%Y-%m-%d")
    day.mkdir(parents=True, exist_ok=True)
    ext = "wav" if S.corpus_format == "wav" else "flac"
    name = f"{now.strftime('%H%M%S_%f')[:-3]}_{req_id}.{ext}"
    sf.write(day / name, audio, TARGET_SR, subtype="PCM_16")
    record["audio"] = f"{day.name}/{name}"
    with corpus_lock, open(day / "records.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _respond(text: str, fmt: str, extra: dict):
    if fmt == "text":
        return PlainTextResponse(text)
    if fmt == "verbose_json":
        return JSONResponse({"text": text, "task": "transcribe", **extra})
    return JSONResponse({"text": text})


@app.get("/health")
def health():
    return {"status": "ok" if engine else "loading", "model": S.model_id, "context": context.get(),
            "save_corpus": S.save_corpus}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": S.model_id, "object": "model", "owned_by": "local"}]}


@app.post("/v1/audio/transcriptions")
def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(""),
    language: str = Form(""),
    prompt: str = Form(""),
    response_format: str = Form("json"),
    temperature: str = Form(""),
):
    if engine is None:
        raise HTTPException(503, "model not loaded yet")
    t0 = time.perf_counter()
    now = dt.datetime.now().astimezone()
    req_id = uuid.uuid4().hex[:8]
    raw = file.file.read()
    try:
        audio = decode_to_16k_mono(raw) if raw else np.zeros(0, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"cannot decode audio: {exc}") from exc
    t_dec = time.perf_counter()
    duration = audio.size / TARGET_SR
    lang = LANG_MAP.get(language.strip().lower(), language.strip()) if language.strip() else S.language
    ctx = " ".join(p for p in (context.get(), prompt.strip()) if p)

    _, n_voiced = voiced_stats(audio, S.floor_dbfs)
    skipped = None
    if duration < S.min_duration_s:
        skipped = "too_short"
    elif n_voiced / TARGET_SR < S.min_voiced_s:
        skipped = "silent"

    rms_db = gain_db = None
    if skipped:
        text = ""
    else:
        norm, rms_db, gain_db = normalize_level(audio, S.target_dbfs, S.floor_dbfs)
        text = engine.transcribe(norm, lang, ctx)
    t_end = time.perf_counter()
    lat = {"decode_ms": round((t_dec - t0) * 1000, 1), "infer_ms": round((t_end - t_dec) * 1000, 1),
           "total_ms": round((t_end - t0) * 1000, 1)}
    log.info("[%s] %.2fs audio -> %.0f ms%s: %s", req_id, duration, lat["total_ms"],
             f" ({skipped})" if skipped else "", text)

    if S.save_corpus and not skipped:
        record = {"id": req_id, "timestamp": now.isoformat(timespec="milliseconds"), "audio": None,
                  "duration_s": round(duration, 3), "text": text, "corrected_text": "",
                  "latency_ms": lat, "language": lang, "prompt": ctx, "voiced_rms_dbfs": round(rms_db, 1),
                  "gain_db": round(gain_db, 1), "model": S.model_id,
                  "client": {"model": model, "language": language, "prompt": prompt}}
        try:
            _save_corpus(req_id, now, audio, record)
        except Exception:  # corpus must never break dictation
            log.exception("failed to save corpus record")

    return _respond(text, response_format, {"language": lang, "duration": round(duration, 3),
                                            "latency_ms": lat, "skipped": skipped})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=S.host, port=S.port, log_level="info")
