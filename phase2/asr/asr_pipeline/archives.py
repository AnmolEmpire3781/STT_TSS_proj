"""Small, Drive-friendly tar shards containing canonical WAV files."""

from __future__ import annotations

import os
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping


def _safe_member_name(name: str) -> str:
    member = PurePosixPath(name)
    if member.is_absolute() or ".." in member.parts or not member.parts:
        raise ValueError(f"Unsafe archive member path: {name!r}")
    return member.as_posix()


@dataclass(frozen=True)
class ArchivedAudio:
    archive: str
    audio: str


class TarShardWriter:
    """Write bounded uncompressed tar shards and promote only closed shards."""

    def __init__(self, output_dir: Path, max_bytes: int = 512 * 1024 * 1024):
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self._index = -1
        self._size = 0
        self._tar: tarfile.TarFile | None = None
        self._temporary: Path | None = None
        self._final: Path | None = None
        self.completed: list[Path] = []

    def _open_next(self) -> None:
        self._finish_current()
        self._index += 1
        name = f"audio-{self._index:05d}.tar"
        self._final = self.output_dir / name
        self._temporary = self.output_dir / f"{name}.incomplete"
        if self._final.exists() or self._temporary.exists():
            raise FileExistsError(f"Refusing to overwrite archive shard {name}")
        self._tar = tarfile.open(self._temporary, mode="w")
        self._size = 0

    def add(self, source: Path, member_name: str) -> ArchivedAudio:
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(source)
        member_name = _safe_member_name(member_name)
        size = source.stat().st_size
        if self._tar is None or (self._size and self._size + size > self.max_bytes):
            self._open_next()
        assert self._tar is not None and self._final is not None
        self._tar.add(source, arcname=member_name, recursive=False)
        self._size += size
        return ArchivedAudio(
            archive=f"shards/{self._final.name}",
            audio=member_name,
        )

    def _finish_current(self) -> None:
        if self._tar is None:
            return
        assert self._temporary is not None and self._final is not None
        self._tar.close()
        self._tar = None
        os.replace(self._temporary, self._final)
        self.completed.append(self._final)

    def close(self) -> list[Path]:
        self._finish_current()
        return list(self.completed)

    def __enter__(self) -> "TarShardWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()
        elif self._tar is not None:
            self._tar.close()
            self._tar = None


def safe_extract_tar(archive: Path, destination: Path) -> None:
    """Extract regular files while rejecting traversal, links, and devices."""

    archive = Path(archive).resolve()
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, mode="r:*") as handle:
        members = handle.getmembers()
        for member in members:
            name = _safe_member_name(member.name)
            target = (destination / Path(*PurePosixPath(name).parts)).resolve()
            if destination not in target.parents:
                raise ValueError(f"Archive member escapes destination: {member.name}")
            if not member.isfile() and not member.isdir():
                raise ValueError(f"Unsupported archive member type: {member.name}")
        #codec*change*+2026-09-17 Python's filter blocks additional tar tricks,
        # while the checks above retain the same policy on older Colab runtimes.
        try:
            handle.extractall(destination, members=members, filter="data")
        except TypeError:
            handle.extractall(destination, members=members)


def stage_archived_audio(
    rows: Iterable[Mapping[str, object]], artifact_dir: Path, staging_dir: Path
) -> dict[str, Path]:
    """Extract each referenced shard once and return canonical IDs to WAV paths."""

    artifact_dir = Path(artifact_dir).resolve()
    staging_dir = Path(staging_dir).resolve()
    staging_dir.mkdir(parents=True, exist_ok=True)
    extracted: set[str] = set()
    resolved: dict[str, Path] = {}
    for row in rows:
        sample_id = str(row["id"])
        audio = _safe_member_name(str(row["audio"]))
        archive_value = row.get("archive")
        if archive_value:
            archive_rel = _safe_member_name(str(archive_value))
            archive_path = (artifact_dir / Path(*PurePosixPath(archive_rel).parts)).resolve()
            if artifact_dir not in archive_path.parents or not archive_path.is_file():
                raise FileNotFoundError(f"Missing or unsafe archive: {archive_value}")
            if archive_rel not in extracted:
                safe_extract_tar(archive_path, staging_dir)
                extracted.add(archive_rel)
            path = (staging_dir / Path(*PurePosixPath(audio).parts)).resolve()
        else:
            path = (artifact_dir / Path(*PurePosixPath(audio).parts)).resolve()
            if artifact_dir not in path.parents:
                raise ValueError(f"Audio path escapes artifact: {audio}")
        if not path.is_file():
            raise FileNotFoundError(path)
        resolved[sample_id] = path
    return resolved


def copy_or_link(source: Path, destination: Path) -> None:
    """Prefer a local hardlink and fall back to a metadata-preserving copy."""

    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
