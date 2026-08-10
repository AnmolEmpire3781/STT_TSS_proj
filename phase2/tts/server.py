import io
import os
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

app = FastAPI(title="Indic Parler-TTS English/Hindi Service")
MODEL_ID = os.getenv("INDIC_PARLER_MODEL", "ai4bharat/indic-parler-tts")
DEVICE = os.getenv("INDIC_PARLER_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")

model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_ID).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

DEFAULT_VOICES = {"en": "Thoma", "hi": "Divya"}


class TTSRequest(BaseModel):
    text: str
    language: str = "en"
    voice: str | None = None


def description_for(language: str, voice: str) -> str:
    if language == "hi":
        return f"{voice}'s voice is clear and natural, with a moderate speaking rate and pitch. The recording is very clear with almost no background noise."
    return f"{voice}'s Indian English voice is clear and natural, with a moderate speaking rate and pitch. The recording is very clear with almost no background noise."


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_ID, "device": DEVICE, "languages": ["en", "hi"]}


@app.post("/synthesize")
def synthesize(req: TTSRequest):
    if req.language not in {"en", "hi"}:
        raise HTTPException(400, "Supported languages are en and hi")
    voice = req.voice or DEFAULT_VOICES[req.language]
    description = description_for(req.language, voice)

    description_inputs = description_tokenizer(description, return_tensors="pt").to(DEVICE)
    prompt_inputs = tokenizer(req.text, return_tensors="pt").to(DEVICE)

    with torch.inference_mode():
        generation = model.generate(
            input_ids=description_inputs.input_ids,
            attention_mask=description_inputs.attention_mask,
            prompt_input_ids=prompt_inputs.input_ids,
            prompt_attention_mask=prompt_inputs.attention_mask,
        )

    audio = generation.detach().cpu().numpy().squeeze()
    buf = io.BytesIO()
    sf.write(buf, audio, model.config.sampling_rate, format="WAV")
    return Response(content=buf.getvalue(), media_type="audio/wav")

