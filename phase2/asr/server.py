import os
import tempfile
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from faster_whisper import WhisperModel

app = FastAPI(title="Faster-Whisper English/Hindi ASR Service")
MODEL_ID = os.getenv("FASTER_WHISPER_MODEL", "large-v3-turbo")
DEVICE = os.getenv("FASTER_WHISPER_DEVICE", "auto")
COMPUTE_TYPE = os.getenv("FASTER_WHISPER_COMPUTE_TYPE", "default")
BEAM_SIZE = int(os.getenv("FASTER_WHISPER_BEAM_SIZE", "3"))

model = WhisperModel(MODEL_ID, device=DEVICE, compute_type=COMPUTE_TYPE)


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_ID, "device": DEVICE, "compute_type": COMPUTE_TYPE}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...), language: str = Form("en")):
    if language not in {"en", "hi"}:
        raise HTTPException(400, "Supported languages are en and hi")
    suffix = os.path.splitext(audio.filename or "")[1] or ".webm"
    data = await audio.read()
    if not data:
        raise HTTPException(400, "Empty audio file")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as f:
        f.write(data)
        f.flush()
        segments, info = model.transcribe(
            f.name,
            language=language,
            task="transcribe",
            beam_size=BEAM_SIZE,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
    if not text:
        raise HTTPException(500, "ASR returned empty transcription")
    return {"text": text, "detected_language": getattr(info, "language", language)}
