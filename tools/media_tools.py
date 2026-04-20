"""Audio transcription and video parsing tools."""

from __future__ import annotations

import json
import logging
import math
import mimetypes
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urljoin, urlparse

import httpx

from hermes_constants import display_hermes_home, get_hermes_dir
from tools.registry import registry, tool_error
from tools.transcription_tools import MAX_FILE_SIZE, SUPPORTED_FORMATS, _find_ffmpeg_binary, transcribe_audio
from tools.url_safety import is_safe_url
from tools.website_policy import check_website_access

logger = logging.getLogger(__name__)

_VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".flv", ".m4v", ".mpeg", ".mpg",
}
_DEFAULT_SEGMENT_SECONDS = 600
_DEFAULT_MAX_FRAMES = 4
_DEFAULT_TIMEOUT = 300.0
_DEFAULT_FRAME_PROMPT = (
    "Describe the important visible content in this video frame. Focus on people, on-screen text, "
    "UI, charts, products, scene context, and anything that would help summarize the video."
)


def _default_media_dir() -> Path:
    return get_hermes_dir("cache/media", "media_cache")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _looks_like_url(source: str) -> bool:
    parsed = urlparse(str(source or ""))
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _safe_slug(source: str) -> str:
    raw = Path(urlparse(source).path or source).stem or "media"
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in raw)
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned[:80] or "media"


def _format_seconds(total_seconds: int) -> str:
    total_seconds = max(0, int(total_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _extension_for_kind(kind: str, url: str, content_type: str = "") -> str:
    ext = Path(urlparse(url).path).suffix.lower()
    if kind == "audio" and ext in SUPPORTED_FORMATS:
        return ext
    if kind == "video" and ext in _VIDEO_EXTENSIONS:
        return ext
    guessed = mimetypes.guess_extension(content_type.split(";", 1)[0].strip()) if content_type else None
    if guessed:
        return guessed.lower()
    return ".m4a" if kind == "audio" else ".mp4"


def _check_remote_source(url: str) -> Optional[str]:
    if not is_safe_url(url):
        return f"Blocked unsafe or private URL: {url}"
    blocked = check_website_access(url)
    if blocked:
        return f"Blocked by website policy ({blocked['pattern']} from {blocked['source']}): {url}"
    return None


def _download_direct_media(url: str, dest_dir: Path, kind: str) -> tuple[Optional[Path], Optional[str]]:
    error = _check_remote_source(url)
    if error:
        return None, error

    current_url = url
    try:
        with httpx.Client(follow_redirects=False, timeout=_DEFAULT_TIMEOUT) as client:
            for _ in range(5):
                response = client.get(current_url)
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location", "")
                    if not location:
                        return None, f"Redirect from {current_url} missing Location header"
                    current_url = urljoin(current_url, location)
                    error = _check_remote_source(current_url)
                    if error:
                        return None, error
                    continue

                response.raise_for_status()
                ext = _extension_for_kind(kind, current_url, response.headers.get("content-type", ""))
                dest = dest_dir / f"downloaded{ext}"
                dest.write_bytes(response.content)
                return dest, None
        return None, f"Too many redirects while downloading {url}"
    except Exception as exc:
        return None, f"Failed to download media: {exc}"


def _download_with_ytdlp(url: str, dest_dir: Path, kind: str) -> tuple[Optional[Path], Optional[str]]:
    ytdlp = shutil.which("yt-dlp")
    if not ytdlp:
        return None, (
            "yt-dlp is required for non-direct media URLs. Install yt-dlp or provide a direct file URL/local path."
        )

    output_template = str(dest_dir / "downloaded.%(ext)s")
    command = [ytdlp, "--no-playlist", "--restrict-filenames", "-o", output_template]
    if kind == "audio":
        command += ["-f", "bestaudio/best"]
    else:
        command += ["-f", "bestvideo*+bestaudio/best"]
    command.append(url)

    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=_DEFAULT_TIMEOUT, check=False)
    except Exception as exc:
        return None, f"yt-dlp failed: {exc}"

    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip() or "unknown error"
        return None, f"yt-dlp failed: {details}"

    files = sorted(path for path in dest_dir.iterdir() if path.is_file() and path.name.startswith("downloaded."))
    if not files:
        return None, f"yt-dlp finished but no file was produced for {url}"
    return files[0], None


def _resolve_media_source(source: str, work_dir: Path, kind: str) -> tuple[Optional[Path], Optional[str], Dict[str, Any]]:
    metadata: Dict[str, Any] = {"source": source, "kind": kind}
    if not source:
        return None, "source is required", metadata

    candidate = Path(source).expanduser()
    if candidate.exists() and candidate.is_file():
        resolved = candidate.resolve()
        metadata["resolved_source"] = str(resolved)
        metadata["downloaded"] = False
        return resolved, None, metadata

    if not _looks_like_url(source):
        return None, f"Media source not found: {source}", metadata

    direct_ext = Path(urlparse(source).path).suffix.lower()
    downloaded_path = None
    if (kind == "audio" and direct_ext in SUPPORTED_FORMATS) or (kind == "video" and direct_ext in _VIDEO_EXTENSIONS):
        downloaded_path, error = _download_direct_media(source, work_dir, kind)
    else:
        downloaded_path, error = _download_with_ytdlp(source, work_dir, kind)
        if error and direct_ext:
            downloaded_path, direct_error = _download_direct_media(source, work_dir, kind)
            error = direct_error if downloaded_path is None else None
    if error:
        return None, error, metadata
    metadata["resolved_source"] = str(downloaded_path)
    metadata["downloaded"] = True
    return downloaded_path, None, metadata


def _probe_duration_seconds(media_path: Path) -> Optional[float]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    except Exception:
        return None
    if result.returncode != 0:
        return None
    try:
        duration = float((result.stdout or "").strip())
    except ValueError:
        return None
    return duration if duration > 0 else None


def _segment_media_for_transcription(
    media_path: Path,
    work_dir: Path,
    segment_seconds: int,
    is_video: bool,
) -> tuple[list[Path], Optional[str]]:
    _ensure_dir(work_dir)
    ffmpeg = _find_ffmpeg_binary()
    if not ffmpeg:
        return [], "ffmpeg is required to process this media source"

    output_pattern = str(work_dir / "segment_%03d.wav")
    command = [ffmpeg, "-y", "-i", str(media_path)]
    if is_video:
        command.append("-vn")
    command += [
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "segment",
        "-segment_time",
        str(max(60, int(segment_seconds))),
        "-c:a",
        "pcm_s16le",
        output_pattern,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=_DEFAULT_TIMEOUT, check=False)
    except Exception as exc:
        return [], f"ffmpeg failed while preparing media: {exc}"
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip() or "unknown error"
        return [], f"ffmpeg failed while preparing media: {details}"

    segments = sorted(work_dir.glob("segment_*.wav"))
    if not segments:
        return [], "ffmpeg completed but produced no audio segments"
    return segments, None


def _transcribe_prepared_segments(segments: list[Path], model: Optional[str], segment_seconds: int) -> Dict[str, Any]:
    transcript_blocks = []
    providers = []
    segment_rows = []
    multiple_segments = len(segments) > 1

    for index, segment_path in enumerate(segments):
        result = transcribe_audio(str(segment_path), model=model)
        if not result.get("success"):
            return {
                "success": False,
                "error": result.get("error") or f"Failed to transcribe segment {segment_path.name}",
                "transcript": "",
            }

        transcript = (result.get("transcript") or "").strip()
        offset_seconds = index * max(60, int(segment_seconds))
        label = _format_seconds(offset_seconds)
        if transcript:
            transcript_blocks.append(f"[{label}] {transcript}" if multiple_segments else transcript)
        providers.append(result.get("provider", ""))
        segment_rows.append(
            {
                "index": index,
                "path": str(segment_path),
                "offset_seconds": offset_seconds,
                "offset": label,
                "provider": result.get("provider"),
                "transcript_length": len(transcript),
            }
        )

    provider_values = sorted({provider for provider in providers if provider})
    provider_name = provider_values[0] if len(provider_values) == 1 else "mixed"
    return {
        "success": True,
        "transcript": "\n\n".join(block for block in transcript_blocks if block).strip(),
        "provider": provider_name,
        "providers": provider_values,
        "chunk_count": len(segments),
        "segments": segment_rows,
    }


def _transcribe_media_path(
    media_path: Path,
    model: Optional[str],
    segment_seconds: int,
    work_dir: Path,
    is_video: bool,
) -> Dict[str, Any]:
    if not is_video and media_path.suffix.lower() in SUPPORTED_FORMATS:
        try:
            if media_path.stat().st_size <= MAX_FILE_SIZE:
                result = transcribe_audio(str(media_path), model=model)
                if result.get("success"):
                    result.setdefault("chunk_count", 1)
                    result.setdefault("providers", [result.get("provider")] if result.get("provider") else [])
                    result.setdefault(
                        "segments",
                        [
                            {
                                "index": 0,
                                "path": str(media_path),
                                "offset_seconds": 0,
                                "offset": "00:00:00",
                                "provider": result.get("provider"),
                                "transcript_length": len((result.get("transcript") or "").strip()),
                            }
                        ],
                    )
                    return result
        except OSError:
            pass

    segments, error = _segment_media_for_transcription(
        media_path=media_path,
        work_dir=work_dir,
        segment_seconds=segment_seconds,
        is_video=is_video,
    )
    if error:
        return {"success": False, "error": error, "transcript": ""}
    return _transcribe_prepared_segments(segments, model=model, segment_seconds=segment_seconds)


def _write_text(path: Path, content: str) -> None:
    _ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _extract_video_frames(
    video_path: Path,
    frames_dir: Path,
    max_frames: int,
) -> tuple[list[Dict[str, Any]], Optional[str]]:
    ffmpeg = _find_ffmpeg_binary()
    if not ffmpeg:
        return [], "ffmpeg is required for frame extraction"

    duration = _probe_duration_seconds(video_path)
    frame_count = max(1, min(int(max_frames or _DEFAULT_MAX_FRAMES), 12))
    interval = max(1.0, (duration / frame_count) if duration else 30.0)
    output_pattern = str(frames_dir / "frame_%03d.jpg")
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps=1/{interval}",
        "-frames:v",
        str(frame_count),
        output_pattern,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=_DEFAULT_TIMEOUT, check=False)
    except Exception as exc:
        return [], f"ffmpeg failed while extracting frames: {exc}"
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip() or "unknown error"
        return [], f"ffmpeg failed while extracting frames: {details}"

    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if not frames:
        return [], "No frames were extracted from the video"

    rows = []
    for index, frame_path in enumerate(frames):
        timestamp_seconds = int(round(index * interval))
        if duration is not None:
            timestamp_seconds = min(timestamp_seconds, max(0, int(duration)))
        rows.append(
            {
                "index": index,
                "path": str(frame_path),
                "timestamp_seconds": timestamp_seconds,
                "timestamp": _format_seconds(timestamp_seconds),
            }
        )
    return rows, None


def _analyze_video_frames(frames: list[Dict[str, Any]], question: str) -> list[Dict[str, Any]]:
    analyses = []
    for frame in frames:
        raw = registry.dispatch(
            "vision_analyze",
            {"image_url": frame["path"], "question": question},
        )
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {"success": False, "analysis": raw}
        analyses.append(
            {
                "index": frame["index"],
                "timestamp_seconds": frame["timestamp_seconds"],
                "timestamp": frame["timestamp"],
                "frame_path": frame["path"],
                "success": bool(result.get("success")),
                "analysis": result.get("analysis") or result.get("error") or "",
            }
        )
    return analyses


def _resolve_output_dir(source: str, output_dir: Optional[str]) -> Path:
    if output_dir:
        return _ensure_dir(Path(output_dir).expanduser().resolve())
    return _ensure_dir(_default_media_dir() / _safe_slug(source))


def _check_vision_available() -> bool:
    try:
        from tools.vision_tools import check_vision_requirements

        return bool(check_vision_requirements())
    except Exception:
        return False


def audio_transcribe_tool(
    source: str,
    model: Optional[str] = None,
    output_dir: Optional[str] = None,
    segment_seconds: int = _DEFAULT_SEGMENT_SECONDS,
) -> str:
    work_dir = Path(tempfile.mkdtemp(prefix="hermes-audio-"))
    try:
        media_path, error, metadata = _resolve_media_source(source, work_dir, kind="audio")
        if error:
            return tool_error(error, success=False)

        save_dir = _resolve_output_dir(source, output_dir)
        transcription = _transcribe_media_path(
            media_path=media_path,
            model=model,
            segment_seconds=segment_seconds,
            work_dir=work_dir / "segments",
            is_video=False,
        )
        if not transcription.get("success"):
            return tool_error(transcription.get("error") or "Audio transcription failed", success=False)

        transcript = transcription.get("transcript", "")
        transcript_path = save_dir / f"{media_path.stem}.transcript.txt"
        metadata_path = save_dir / f"{media_path.stem}.transcript.json"
        _write_text(transcript_path, transcript)
        _write_json(
            metadata_path,
            {
                **metadata,
                "model": model,
                "segment_seconds": int(segment_seconds),
                **transcription,
                "transcript_path": str(transcript_path),
            },
        )

        result = {
            "success": True,
            "source": source,
            "resolved_source": str(media_path),
            "downloaded": metadata.get("downloaded", False),
            "provider": transcription.get("provider"),
            "providers": transcription.get("providers", []),
            "chunk_count": transcription.get("chunk_count", 1),
            "transcript": transcript,
            "transcript_path": str(transcript_path),
            "metadata_path": str(metadata_path),
        }
        return json.dumps(result, ensure_ascii=False, indent=2)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def video_analyze_tool(
    source: str,
    model: Optional[str] = None,
    output_dir: Optional[str] = None,
    segment_seconds: int = _DEFAULT_SEGMENT_SECONDS,
    deep: bool = False,
    max_frames: int = _DEFAULT_MAX_FRAMES,
    frame_question: str = _DEFAULT_FRAME_PROMPT,
) -> str:
    work_dir = Path(tempfile.mkdtemp(prefix="hermes-video-"))
    warnings = []
    try:
        media_path, error, metadata = _resolve_media_source(source, work_dir, kind="video")
        if error:
            return tool_error(error, success=False)

        save_dir = _resolve_output_dir(source, output_dir)
        transcription = _transcribe_media_path(
            media_path=media_path,
            model=model,
            segment_seconds=segment_seconds,
            work_dir=work_dir / "segments",
            is_video=True,
        )
        if not transcription.get("success"):
            return tool_error(transcription.get("error") or "Video transcription failed", success=False)

        transcript = transcription.get("transcript", "")
        transcript_path = save_dir / f"{media_path.stem}.transcript.txt"
        report_path = save_dir / f"{media_path.stem}.analysis.json"
        _write_text(transcript_path, transcript)

        frame_analyses = []
        if deep:
            if not _check_vision_available():
                warnings.append("Vision is not configured; skipped frame analysis.")
            else:
                frames_dir = _ensure_dir(save_dir / "frames")
                frames, frame_error = _extract_video_frames(
                    video_path=media_path,
                    frames_dir=frames_dir,
                    max_frames=max_frames,
                )
                if frame_error:
                    warnings.append(frame_error)
                else:
                    frame_analyses = _analyze_video_frames(frames, frame_question)

        result = {
            "success": True,
            "source": source,
            "resolved_source": str(media_path),
            "downloaded": metadata.get("downloaded", False),
            "provider": transcription.get("provider"),
            "providers": transcription.get("providers", []),
            "chunk_count": transcription.get("chunk_count", 1),
            "transcript": transcript,
            "transcript_path": str(transcript_path),
            "deep": bool(deep),
            "frame_count": len(frame_analyses),
            "frame_analyses": frame_analyses,
            "report_path": str(report_path),
        }
        if warnings:
            result["warnings"] = warnings

        _write_json(
            report_path,
            {
                **result,
                "segment_seconds": int(segment_seconds),
                "max_frames": int(max_frames),
                "frame_question": frame_question,
                "cache_root": str(_default_media_dir()),
            },
        )
        return json.dumps(result, ensure_ascii=False, indent=2)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _handle_audio_transcribe(args: Dict[str, Any], **_: Any) -> str:
    source = str(args.get("source", "") or "").strip()
    model = str(args.get("model", "") or "").strip() or None
    output_dir = str(args.get("output_dir", "") or "").strip() or None
    segment_seconds = int(args.get("segment_seconds", _DEFAULT_SEGMENT_SECONDS) or _DEFAULT_SEGMENT_SECONDS)
    return audio_transcribe_tool(
        source=source,
        model=model,
        output_dir=output_dir,
        segment_seconds=segment_seconds,
    )


def _handle_video_analyze(args: Dict[str, Any], **_: Any) -> str:
    source = str(args.get("source", "") or "").strip()
    model = str(args.get("model", "") or "").strip() or None
    output_dir = str(args.get("output_dir", "") or "").strip() or None
    segment_seconds = int(args.get("segment_seconds", _DEFAULT_SEGMENT_SECONDS) or _DEFAULT_SEGMENT_SECONDS)
    deep = bool(args.get("deep", False))
    max_frames = int(args.get("max_frames", _DEFAULT_MAX_FRAMES) or _DEFAULT_MAX_FRAMES)
    frame_question = str(args.get("frame_question", _DEFAULT_FRAME_PROMPT) or _DEFAULT_FRAME_PROMPT).strip()
    return video_analyze_tool(
        source=source,
        model=model,
        output_dir=output_dir,
        segment_seconds=segment_seconds,
        deep=deep,
        max_frames=max_frames,
        frame_question=frame_question,
    )


AUDIO_TRANSCRIBE_SCHEMA = {
    "name": "audio_transcribe",
    "description": "Transcribe local or remote audio files into text. Supports long audio by chunking with ffmpeg when needed.",
    "parameters": {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "Local audio file path or remote audio URL. Non-direct media pages can also be used when yt-dlp is installed.",
            },
            "model": {
                "type": "string",
                "description": "Optional STT model override. Uses the configured provider-specific model when omitted.",
            },
            "output_dir": {
                "type": "string",
                "description": f"Optional directory to save transcript artifacts. Defaults to {display_hermes_home()}/cache/media/<source>/.",
            },
            "segment_seconds": {
                "type": "integer",
                "description": "Optional chunk size for long media in seconds. Default: 600.",
            },
        },
        "required": ["source"],
    },
}


VIDEO_ANALYZE_SCHEMA = {
    "name": "video_analyze",
    "description": "Parse local or remote videos by extracting speech transcript and, optionally, analyzing sampled frames with the vision tool.",
    "parameters": {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "Local video file path or remote video URL. Non-direct media pages can also be used when yt-dlp is installed.",
            },
            "model": {
                "type": "string",
                "description": "Optional STT model override. Uses the configured provider-specific model when omitted.",
            },
            "output_dir": {
                "type": "string",
                "description": f"Optional directory to save transcript and frame analysis artifacts. Defaults to {display_hermes_home()}/cache/media/<source>/.",
            },
            "segment_seconds": {
                "type": "integer",
                "description": "Optional chunk size for the extracted audio in seconds. Default: 600.",
            },
            "deep": {
                "type": "boolean",
                "description": "When true, sample frames from the video and analyze them with the vision tool.",
            },
            "max_frames": {
                "type": "integer",
                "description": "Maximum number of frames to analyze when deep=true. Default: 4.",
            },
            "frame_question": {
                "type": "string",
                "description": "Optional prompt passed to vision analysis for each sampled frame.",
            },
        },
        "required": ["source"],
    },
}


registry.register(
    name="audio_transcribe",
    toolset="audio",
    schema=AUDIO_TRANSCRIBE_SCHEMA,
    handler=_handle_audio_transcribe,
    emoji="🎧",
    max_result_size_chars=120_000,
)

registry.register(
    name="video_analyze",
    toolset="video",
    schema=VIDEO_ANALYZE_SCHEMA,
    handler=_handle_video_analyze,
    emoji="🎬",
    max_result_size_chars=140_000,
)
