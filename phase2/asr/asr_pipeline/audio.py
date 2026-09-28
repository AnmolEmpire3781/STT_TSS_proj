"""Audio probing, validation, and non-destructive canonical conversion."""

from __future__ import annotations

from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Sequence
import wave


TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2


class AudioError(RuntimeError):
    """Raised when audio cannot be inspected or converted safely."""


@dataclass(frozen=True, slots=True)
class AudioInfo:
    """Properties read from a PCM WAV header."""

    path: Path
    sample_rate: int
    channels: int
    sample_width: int
    compression_type: str
    frames: int
    duration: float


def _inspect_wave(reader: wave.Wave_read, path: Path) -> AudioInfo:
    frames = reader.getnframes()
    sample_rate = reader.getframerate()
    channels = reader.getnchannels()
    sample_width = reader.getsampwidth()
    compression_type = reader.getcomptype()
    duration = frames / sample_rate if sample_rate else 0.0
    return AudioInfo(
        path=path,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        compression_type=compression_type,
        frames=frames,
        duration=duration,
    )


def inspect_wav(path: str | os.PathLike[str]) -> AudioInfo:
    """Read WAV header information without loading sample data."""

    audio_path = Path(path)
    try:
        with wave.open(str(audio_path), "rb") as reader:
            return _inspect_wave(reader, audio_path)
    except (OSError, EOFError, wave.Error) as exc:
        raise AudioError(f"cannot read WAV file {audio_path}: {exc}") from exc


def validate_wav(
    path: str | os.PathLike[str],
    *,
    sample_rate: int = TARGET_SAMPLE_RATE,
    channels: int = TARGET_CHANNELS,
    sample_width: int = TARGET_SAMPLE_WIDTH,
) -> AudioInfo:
    """Require canonical WAV/16 kHz/mono/PCM16 properties."""

    info = inspect_wav(path)
    mismatches: list[str] = []
    if info.sample_rate != sample_rate:
        mismatches.append(f"sample_rate={info.sample_rate} (expected {sample_rate})")
    if info.channels != channels:
        mismatches.append(f"channels={info.channels} (expected {channels})")
    if info.sample_width != sample_width:
        mismatches.append(
            f"sample_width={info.sample_width} (expected {sample_width} bytes)"
        )
    if info.compression_type != "NONE":
        mismatches.append(f"compression={info.compression_type} (expected PCM)")
    if info.frames <= 0:
        mismatches.append("audio has no frames")
    if mismatches:
        raise AudioError(f"non-canonical WAV {info.path}: {', '.join(mismatches)}")
    return info


def _run_ffmpeg(
    input_arguments: Sequence[str],
    temporary_output: Path,
    *,
    ffmpeg_bin: str,
    stdin_data: bytes | None = None,
) -> None:
    command = [
        ffmpeg_bin,
        "-y",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        *input_arguments,
        "-map_metadata",
        "-1",
        "-vn",
        "-ac",
        str(TARGET_CHANNELS),
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        str(temporary_output),
    ]
    try:
        completed = subprocess.run(
            command,
            input=stdin_data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as exc:
        raise AudioError(f"ffmpeg executable not found: {ffmpeg_bin}") from exc
    if completed.returncode:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise AudioError(f"ffmpeg conversion failed: {detail or 'unknown error'}")


def _temporary_wav_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".tmp.wav", dir=destination.parent
    )
    os.close(handle)
    Path(name).unlink(missing_ok=True)
    return Path(name)


def _finish_conversion(
    temporary_output: Path, destination: Path, *, overwrite: bool
) -> AudioInfo:
    validate_wav(temporary_output)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"destination already exists: {destination}")
    #codec*change*+2026-09-17: Conversion lands atomically only after the
    # temporary file passes canonical WAV validation, leaving raw input alone.
    os.replace(temporary_output, destination)
    return validate_wav(destination)


def convert_audio_to_wav(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    ffmpeg_bin: str = "ffmpeg",
    overwrite: bool = False,
) -> AudioInfo:
    """Convert a file to canonical WAV without changing the source file."""

    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.is_file():
        raise AudioError(f"audio source does not exist or is not a file: {source_path}")
    if source_path.resolve() == destination_path.resolve():
        raise AudioError("source and destination must be different paths")
    if destination_path.exists() and not overwrite:
        raise FileExistsError(f"destination already exists: {destination_path}")

    temporary_output = _temporary_wav_path(destination_path)
    try:
        _run_ffmpeg(["-i", str(source_path)], temporary_output, ffmpeg_bin=ffmpeg_bin)
        return _finish_conversion(temporary_output, destination_path, overwrite=overwrite)
    finally:
        temporary_output.unlink(missing_ok=True)


def _array_as_float32_bytes(array: Any) -> tuple[bytes, int]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised in dependency env
        raise AudioError("NumPy is required for decoded array audio payloads") from exc

    values = np.asarray(array)
    if values.ndim not in (1, 2) or values.size == 0:
        raise AudioError("decoded audio array must be non-empty and one- or two-dimensional")
    if values.ndim == 1:
        channels = 1
    else:
        # Hugging Face commonly emits (channels, samples); ffmpeg raw input
        # expects interleaved (samples, channels).
        if values.shape[0] <= 8 and values.shape[1] > values.shape[0]:
            values = values.T
        channels = int(values.shape[1])
    if channels < 1 or channels > 64:
        raise AudioError(f"unsupported decoded channel count: {channels}")
    if np.issubdtype(values.dtype, np.integer):
        integer_info = np.iinfo(values.dtype)
        scale = float(max(abs(integer_info.min), integer_info.max))
        values = values.astype(np.float32) / scale
    else:
        values = values.astype(np.float32, copy=False)
    values = np.ascontiguousarray(values, dtype="<f4")
    return values.tobytes(order="C"), channels


def convert_audio_payload_to_wav(
    audio_value: Any,
    destination: str | os.PathLike[str],
    *,
    ffmpeg_bin: str = "ffmpeg",
    overwrite: bool = False,
) -> AudioInfo:
    """Convert an HF path/bytes/decoded-array audio payload to canonical WAV."""

    if isinstance(audio_value, (str, os.PathLike)):
        return convert_audio_to_wav(
            audio_value, destination, ffmpeg_bin=ffmpeg_bin, overwrite=overwrite
        )
    if not isinstance(audio_value, Mapping):
        raise AudioError("unsupported audio payload; expected path or mapping")

    payload_path = audio_value.get("path")
    if payload_path and Path(payload_path).is_file():
        return convert_audio_to_wav(
            payload_path, destination, ffmpeg_bin=ffmpeg_bin, overwrite=overwrite
        )

    destination_path = Path(destination)
    if destination_path.exists() and not overwrite:
        raise FileExistsError(f"destination already exists: {destination_path}")
    temporary_output = _temporary_wav_path(destination_path)
    temporary_input: Path | None = None
    try:
        payload_bytes = audio_value.get("bytes")
        if payload_bytes is not None:
            if not isinstance(payload_bytes, (bytes, bytearray, memoryview)):
                raise AudioError("audio payload 'bytes' must contain bytes")
            handle, name = tempfile.mkstemp(
                prefix=".source-audio.", suffix=".bin", dir=destination_path.parent
            )
            temporary_input = Path(name)
            with os.fdopen(handle, "wb") as stream:
                stream.write(bytes(payload_bytes))
            _run_ffmpeg(
                ["-i", str(temporary_input)], temporary_output, ffmpeg_bin=ffmpeg_bin
            )
        elif audio_value.get("array") is not None:
            sample_rate = audio_value.get("sampling_rate")
            if not isinstance(sample_rate, int) or sample_rate <= 0:
                raise AudioError("decoded audio payload needs a positive sampling_rate")
            stdin_data, channels = _array_as_float32_bytes(audio_value["array"])
            _run_ffmpeg(
                [
                    "-f",
                    "f32le",
                    "-ar",
                    str(sample_rate),
                    "-ac",
                    str(channels),
                    "-i",
                    "pipe:0",
                ],
                temporary_output,
                ffmpeg_bin=ffmpeg_bin,
                stdin_data=stdin_data,
            )
        else:
            raise AudioError("audio payload has no usable path, bytes, or array")
        return _finish_conversion(temporary_output, destination_path, overwrite=overwrite)
    finally:
        temporary_output.unlink(missing_ok=True)
        if temporary_input is not None:
            temporary_input.unlink(missing_ok=True)


def materialize_source_audio(
    source_audio: Any,
    destination: str | os.PathLike[str],
    *,
    ffmpeg_bin: str = "ffmpeg",
    overwrite: bool = False,
) -> AudioInfo:
    """Materialize an :class:`AdapterExample` audio payload as canonical WAV.

    This named wrapper keeps snapshot code independent from the source payload
    representation (local path, Hugging Face bytes, or decoded array).
    """

    return convert_audio_payload_to_wav(
        source_audio,
        destination,
        ffmpeg_bin=ffmpeg_bin,
        overwrite=overwrite,
    )


def probe_audio_duration(
    path: str | os.PathLike[str], *, ffprobe_bin: str = "ffprobe"
) -> float:
    """Return duration for any ffprobe-supported audio file."""

    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
        )
    except FileNotFoundError as exc:
        raise AudioError(f"ffprobe executable not found: {ffprobe_bin}") from exc
    if completed.returncode:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise AudioError(f"ffprobe failed for {path}: {detail or 'unknown error'}")
    try:
        payload = json.loads(completed.stdout.decode("utf-8"))
        duration = float(payload["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AudioError(f"ffprobe returned no valid duration for {path}") from exc
    if duration <= 0:
        raise AudioError(f"audio duration must be positive: {path}")
    return duration


def duration_from_audio_payload(audio_value: Any) -> float:
    """Infer duration from a decoded HF audio payload or a local file."""

    if isinstance(audio_value, (str, os.PathLike)):
        return probe_audio_duration(audio_value)
    if not isinstance(audio_value, Mapping):
        raise AudioError("cannot infer duration from unsupported audio payload")
    array = audio_value.get("array")
    sample_rate = audio_value.get("sampling_rate")
    if array is not None and isinstance(sample_rate, int) and sample_rate > 0:
        shape = getattr(array, "shape", None)
        if shape:
            samples = int(shape[-1]) if len(shape) > 1 and shape[0] <= 8 else int(shape[0])
        else:
            samples = len(array)
        duration = samples / sample_rate
        if duration > 0:
            return duration
    path = audio_value.get("path")
    if path and Path(path).is_file():
        return probe_audio_duration(path)
    payload_bytes = audio_value.get("bytes")
    if isinstance(payload_bytes, (bytes, bytearray, memoryview)):
        try:
            with wave.open(io.BytesIO(bytes(payload_bytes)), "rb") as reader:
                rate = reader.getframerate()
                if rate > 0 and reader.getnframes() > 0:
                    return reader.getnframes() / rate
        except (EOFError, wave.Error):
            pass
    raise AudioError("audio duration is unavailable; decode the payload or supply duration")
