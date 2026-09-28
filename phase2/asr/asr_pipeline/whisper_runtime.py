"""Lazy Transformers Whisper runtime shared by base and PEFT evaluations."""

from __future__ import annotations

import os
import platform
import time
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from .artifacts import sha256_file, sha256_json
from .revisions import is_immutable_revision


BASE_MODEL_ID = "openai/whisper-large-v3-turbo"
TRANSCRIBE_TASK = "transcribe"
CANONICAL_LANGUAGES = ("en", "hi", "hi-en")
HINGLISH_MODES = ("auto", "hi", "en")


class WhisperRuntimeError(RuntimeError):
    """Raised for runtime configuration, dependency, or inference failures."""


def _local_tree_fingerprint(path: Path) -> str:
    entries: list[dict[str, Any]] = []
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file() or ".tmp-" in candidate.name:
            continue
        entries.append(
            {
                "path": candidate.relative_to(path).as_posix(),
                "size": candidate.stat().st_size,
                "sha256": sha256_file(candidate),
            }
        )
    if not entries:
        raise WhisperRuntimeError(f"adapter directory has no files: {path}")
    return sha256_json(entries)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str = BASE_MODEL_ID
    model_revision: str = "main"
    adapter_path: str | None = None
    adapter_revision: str | None = None
    processor_id: str | None = None
    local_files_only: bool = False

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must be non-empty")
        if not self.model_revision.strip():
            raise ValueError("model_revision must be non-empty")
        if self.adapter_revision and not self.adapter_path:
            raise ValueError("adapter_revision requires adapter_path")
        if self.adapter_path and not Path(self.adapter_path).exists():
            if not self.adapter_revision or not is_immutable_revision(
                self.adapter_revision
            ):
                raise ValueError(
                    "remote adapters require --adapter-revision as a full commit SHA"
                )

    @cached_property
    def _adapter_content_fingerprint(self) -> str | None:
        if not self.adapter_path:
            return None
        candidate = Path(self.adapter_path)
        if candidate.exists():
            if not candidate.is_dir():
                raise WhisperRuntimeError(
                    f"local adapter path must be a directory: {candidate}"
                )
            return _local_tree_fingerprint(candidate)
        return sha256_json(
            {
                "adapter_id": self.adapter_path,
                "adapter_revision": self.adapter_revision or "main",
            }
        )

    def adapter_fingerprint(self) -> str | None:
        return self._adapter_content_fingerprint

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["processor_id"] = self.processor_id or self.model_id
        value["adapter_fingerprint"] = self.adapter_fingerprint()
        return value

    @cached_property
    def identity(self) -> dict[str, Any]:
        #codec*change*+2026-09-17: A local adapter may move between Drive and
        # /content; its content hash, rather than its absolute path, is identity.
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "processor_id": self.processor_id or self.model_id,
            "adapter_fingerprint": self.adapter_fingerprint(),
            "adapter_revision": self.adapter_revision,
        }

    @cached_property
    def _fingerprint(self) -> str:
        return sha256_json(self.identity)

    @property
    def fingerprint(self) -> str:
        return self._fingerprint


@dataclass(frozen=True)
class DecodingConfig:
    hinglish_mode: str = "auto"
    num_beams: int = 1
    max_new_tokens: int = 440
    chunk_length_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.hinglish_mode not in HINGLISH_MODES:
            raise ValueError(
                f"hinglish_mode must be one of {', '.join(HINGLISH_MODES)}"
            )
        if self.num_beams < 1:
            raise ValueError("num_beams must be at least 1")
        if not 1 <= self.max_new_tokens <= 440:
            raise ValueError("max_new_tokens must be in [1, 440]")
        if not 0 < self.chunk_length_seconds <= 30:
            raise ValueError("chunk_length_seconds must be in (0, 30]")

    def forced_language(self, canonical_language: str) -> str | None:
        if canonical_language not in CANONICAL_LANGUAGES:
            raise ValueError(
                f"unsupported canonical language {canonical_language!r}; "
                f"expected one of {', '.join(CANONICAL_LANGUAGES)}"
            )
        if canonical_language == "en":
            return "en"
        if canonical_language == "hi":
            return "hi"
        return None if self.hinglish_mode == "auto" else self.hinglish_mode

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "whisper-decoding-v1",
            "task": TRANSCRIBE_TASK,
            "language_prefix_by_canonical_language": {
                "en": "en",
                "hi": "hi",
                "hi-en": (
                    None if self.hinglish_mode == "auto" else self.hinglish_mode
                ),
            },
            "hinglish_mode": self.hinglish_mode,
            "num_beams": self.num_beams,
            "max_new_tokens": self.max_new_tokens,
            "chunk_length_seconds": self.chunk_length_seconds,
            "do_sample": False,
        }

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    latency_seconds: float
    audio_duration_seconds: float
    chunks: int


class WhisperRuntime:
    """Direct feature-extraction/generation path for base or adapter models."""

    def __init__(
        self,
        model_spec: ModelSpec,
        decoding: DecodingConfig,
        *,
        device: str = "auto",
        dtype: str = "auto",
        seed: int = 17,
    ) -> None:
        if device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda, or mps")
        if dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError(
                "dtype must be auto, float32, float16, or bfloat16"
            )
        self.model_spec = model_spec
        self.decoding = decoding
        self.requested_device = device
        self.requested_dtype = dtype
        self.seed = seed
        self._torch: Any = None
        self._model: Any = None
        self._processor: Any = None
        self._device = ""
        self._dtype_name = ""
        self._actual_model_revision: str | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _import_dependencies(self) -> tuple[Any, Any, Any]:
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        except ImportError as exc:
            raise WhisperRuntimeError(
                "Whisper evaluation requires torch and transformers from "
                "requirements-train.txt"
            ) from exc
        return torch, AutoModelForSpeechSeq2Seq, AutoProcessor

    @staticmethod
    def _resolve_device(torch: Any, requested: str) -> str:
        if requested != "auto":
            if requested == "cuda" and not torch.cuda.is_available():
                raise WhisperRuntimeError("CUDA was requested but is unavailable")
            if requested == "mps" and not (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ):
                raise WhisperRuntimeError("MPS was requested but is unavailable")
            return requested
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _resolve_dtype(torch: Any, requested: str, device: str) -> tuple[str, Any]:
        if requested == "auto":
            if device == "cuda":
                supports_bfloat16 = bool(
                    hasattr(torch.cuda, "is_bf16_supported")
                    and torch.cuda.is_bf16_supported()
                )
                requested = "bfloat16" if supports_bfloat16 else "float16"
            else:
                requested = "float32"
        if device == "cpu" and requested != "float32":
            raise WhisperRuntimeError("CPU evaluation requires float32")
        return requested, getattr(torch, requested)

    def load(self) -> "WhisperRuntime":
        if self.loaded:
            return self
        torch, model_class, processor_class = self._import_dependencies()
        device = self._resolve_device(torch, self.requested_device)
        dtype_name, torch_dtype = self._resolve_dtype(
            torch, self.requested_dtype, device
        )
        torch.manual_seed(self.seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(self.seed)

        load_kwargs = {
            "revision": self.model_spec.model_revision,
            "local_files_only": self.model_spec.local_files_only,
            "low_cpu_mem_usage": True,
            "torch_dtype": torch_dtype,
            "use_safetensors": True,
        }
        try:
            model = model_class.from_pretrained(self.model_spec.model_id, **load_kwargs)
            processor = processor_class.from_pretrained(
                self.model_spec.processor_id or self.model_spec.model_id,
                revision=self.model_spec.model_revision,
                local_files_only=self.model_spec.local_files_only,
            )
            if self.model_spec.adapter_path:
                try:
                    from peft import PeftModel
                except ImportError as exc:
                    raise WhisperRuntimeError(
                        "adapter evaluation requires peft from requirements-train.txt"
                    ) from exc
                adapter_kwargs: dict[str, Any] = {
                    "is_trainable": False,
                    "local_files_only": self.model_spec.local_files_only,
                }
                if self.model_spec.adapter_revision:
                    adapter_kwargs["revision"] = self.model_spec.adapter_revision
                model = PeftModel.from_pretrained(
                    model, self.model_spec.adapter_path, **adapter_kwargs
                )
        except WhisperRuntimeError:
            raise
        except Exception as exc:
            raise WhisperRuntimeError(f"failed to load Whisper model: {exc}") from exc

        model.to(device)
        model.eval()
        #codec*change*+2026-09-17: Clear inherited prefix state and supply the
        # recorded language policy explicitly for every sample at generation.
        if hasattr(model, "generation_config"):
            model.generation_config.forced_decoder_ids = None
            model.generation_config.language = None
            model.generation_config.task = None
        config = getattr(model, "config", None)
        self._actual_model_revision = getattr(config, "_commit_hash", None)
        self._torch = torch
        self._model = model
        self._processor = processor
        self._device = device
        self._dtype_name = dtype_name
        return self

    @staticmethod
    def _read_audio(path: Path) -> tuple[Any, int]:
        try:
            import numpy as np
            import soundfile as sf
        except ImportError as exc:
            raise WhisperRuntimeError(
                "audio loading requires numpy and soundfile from "
                "requirements-train.txt"
            ) from exc
        try:
            audio, sample_rate = sf.read(
                str(path), dtype="float32", always_2d=True
            )
        except Exception as exc:
            raise WhisperRuntimeError(f"failed to read audio {path}: {exc}") from exc
        if audio.size == 0:
            raise WhisperRuntimeError(f"audio is empty: {path}")
        mono = audio.mean(axis=1, dtype=np.float32)
        return mono, int(sample_rate)

    @staticmethod
    def _resample(audio: Any, source_rate: int, target_rate: int = 16_000) -> Any:
        if source_rate == target_rate:
            return audio
        if source_rate <= 0:
            raise WhisperRuntimeError(f"invalid audio sample rate: {source_rate}")
        import numpy as np

        target_length = max(1, round(len(audio) * target_rate / source_rate))
        old_positions = np.arange(len(audio), dtype=np.float64)
        new_positions = np.linspace(
            0, max(len(audio) - 1, 0), target_length, dtype=np.float64
        )
        return np.interp(new_positions, old_positions, audio).astype(np.float32)

    def _synchronize(self) -> None:
        if self._device == "cuda":
            self._torch.cuda.synchronize()
        elif self._device == "mps" and hasattr(self._torch, "mps"):
            self._torch.mps.synchronize()

    def _generate_chunk(self, audio: Any, forced_language: str | None) -> str:
        features = self._processor(
            audio,
            sampling_rate=16_000,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = features.input_features.to(
            device=self._device,
            dtype=getattr(self._torch, self._dtype_name),
        )
        generate_kwargs: dict[str, Any] = {
            "task": TRANSCRIBE_TASK,
            "do_sample": False,
            "num_beams": self.decoding.num_beams,
            "max_new_tokens": self.decoding.max_new_tokens,
        }
        if forced_language is not None:
            generate_kwargs["language"] = forced_language
        attention_mask = getattr(features, "attention_mask", None)
        if attention_mask is not None:
            generate_kwargs["attention_mask"] = attention_mask.to(self._device)
        with self._torch.inference_mode():
            generated = self._model.generate(input_features, **generate_kwargs)
        return self._processor.batch_decode(
            generated, skip_special_tokens=True
        )[0].strip()

    def transcribe(
        self,
        audio_path: str | os.PathLike[str],
        language: str,
    ) -> TranscriptionResult:
        self.load()
        forced_language = self.decoding.forced_language(language)
        path = Path(audio_path)
        if not path.is_file():
            raise WhisperRuntimeError(f"audio file does not exist: {path}")

        #codec*change*+2026-09-17: Latency intentionally includes file loading,
        # resampling, feature extraction, and generation; metadata states this.
        self._synchronize()
        started = time.perf_counter()
        audio, source_rate = self._read_audio(path)
        audio = self._resample(audio, source_rate)
        duration = len(audio) / 16_000
        chunk_samples = max(1, round(self.decoding.chunk_length_seconds * 16_000))
        chunks = [
            audio[offset : offset + chunk_samples]
            for offset in range(0, len(audio), chunk_samples)
        ]
        texts = [self._generate_chunk(chunk, forced_language) for chunk in chunks]
        self._synchronize()
        elapsed = time.perf_counter() - started
        return TranscriptionResult(
            text=" ".join(text for text in texts if text).strip(),
            latency_seconds=elapsed,
            audio_duration_seconds=duration,
            chunks=len(chunks),
        )

    def metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "model": self.model_spec.to_dict(),
            "model_fingerprint": self.model_spec.fingerprint,
            "decoding": self.decoding.to_dict(),
            "decoding_fingerprint": self.decoding.fingerprint,
            "seed": self.seed,
            "requested_device": self.requested_device,
            "requested_dtype": self.requested_dtype,
            "latency_scope": "audio_load_resample_feature_extraction_and_generation",
            "platform": platform.platform(),
            "python": platform.python_version(),
        }
        if self.loaded:
            cuda_name = None
            if self._device == "cuda":
                cuda_name = self._torch.cuda.get_device_name(
                    self._torch.cuda.current_device()
                )
            metadata.update(
                {
                    "resolved_device": self._device,
                    "resolved_dtype": self._dtype_name,
                    "actual_model_revision": self._actual_model_revision,
                    "torch_version": self._torch.__version__,
                    "cuda_device_name": cuda_name,
                }
            )
        return metadata
