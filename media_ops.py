from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from shared.utils.video_decode import resolve_media_binary


def _run(command: list[str], cancel_check=None) -> subprocess.CompletedProcess[str]:
    kwargs: dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    process = subprocess.Popen(command, **kwargs)
    while True:
        if cancel_check and cancel_check():
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
            process.communicate()
            raise InterruptedError("FFmpeg operation cancelled.")
        try:
            stdout, stderr = process.communicate(timeout=0.1)
            break
        except subprocess.TimeoutExpired:
            time.sleep(0.02)
    result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "FFmpeg command failed.").strip())
    return result


def probe(path: str | os.PathLike[str]) -> dict[str, Any]:
    ffprobe = resolve_media_binary("ffprobe")
    if not ffprobe:
        raise RuntimeError("Wan2GP FFprobe binary was not found.")
    result = _run([ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", os.fspath(path)])
    payload = json.loads(result.stdout or "{}")
    streams = list(payload.get("streams") or [])
    return {
        "path": os.fspath(path),
        "streams": streams,
        "format": payload.get("format") or {},
        "has_video": any(item.get("codec_type") == "video" for item in streams),
        "has_audio": any(item.get("codec_type") == "audio" for item in streams),
    }


def _output_path(root: Path, node_id: str, suffix: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{node_id}-{time.time_ns()}{suffix}"


def _ffmpeg_output(command: list[str], output: Path, cancel_check=None) -> str:
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    try:
        _run(command + [os.fspath(temporary)], cancel_check=cancel_check)
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise RuntimeError("FFmpeg produced an empty output.")
        temporary.replace(output)
        return os.fspath(output)
    finally:
        temporary.unlink(missing_ok=True)


def extract_last_frame(video_path: str, output: Path, cancel_check=None) -> str:
    ffmpeg = resolve_media_binary("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Wan2GP FFmpeg binary was not found.")
    return _ffmpeg_output([
        ffmpeg, "-y", "-v", "error", "-sseof", "-1", "-i", video_path,
        "-an", "-frames:v", "1", "-update", "1",
    ], output.with_suffix(".png"), cancel_check)


def extract_frame(video_path: str, frame: int, output: Path, cancel_check=None) -> str:
    ffmpeg = resolve_media_binary("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Wan2GP FFmpeg binary was not found.")
    return _ffmpeg_output([
        ffmpeg, "-y", "-v", "error", "-i", video_path, "-vf", f"select=eq(n\\,{max(0, int(frame) - 1)})",
        "-frames:v", "1", "-vsync", "0",
    ], output.with_suffix(".png"), cancel_check)


def trim(video_path: str, start: float, duration: float, output: Path, cancel_check=None) -> str:
    ffmpeg = resolve_media_binary("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Wan2GP FFmpeg binary was not found.")
    return _ffmpeg_output([
        ffmpeg, "-y", "-v", "error", "-ss", str(max(0, float(start))), "-i", video_path,
        "-t", str(max(0.01, float(duration))), "-c", "copy",
    ], output.with_suffix(".mp4"), cancel_check)


def concat(videos: list[str], output: Path, cancel_check=None) -> str:
    if len(videos) < 2:
        raise ValueError("FFmpeg concat requires at least two videos.")
    ffmpeg = resolve_media_binary("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Wan2GP FFmpeg binary was not found.")
    inputs: list[str] = [ffmpeg, "-y", "-v", "error"]
    for path in videos:
        inputs.extend(["-i", path])
    has_audio = all(bool(probe(path).get("has_audio")) for path in videos)
    if has_audio:
        labels = "".join(f"[{index}:v:0][{index}:a:0]" for index in range(len(videos)))
        filters = f"{labels}concat=n={len(videos)}:v=1:a=1[v][a]"
        args = ["-filter_complex", filters, "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-c:a", "aac"]
    else:
        labels = "".join(f"[{index}:v:0]" for index in range(len(videos)))
        filters = f"{labels}concat=n={len(videos)}:v=1:a=0[v]"
        args = ["-filter_complex", filters, "-map", "[v]", "-an", "-c:v", "libx264"]
    return _ffmpeg_output(inputs + args, output.with_suffix(".mp4"), cancel_check)


def mux_audio(video_path: str, audio_path: str, output: Path, cancel_check=None) -> str:
    ffmpeg = resolve_media_binary("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Wan2GP FFmpeg binary was not found.")
    return _ffmpeg_output([
        ffmpeg, "-y", "-v", "error", "-i", video_path, "-i", audio_path,
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest",
    ], output.with_suffix(".mp4"), cancel_check)


def remove_audio(video_path: str, output: Path, cancel_check=None) -> str:
    ffmpeg = resolve_media_binary("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Wan2GP FFmpeg binary was not found.")
    return _ffmpeg_output([ffmpeg, "-y", "-v", "error", "-i", video_path, "-an", "-c:v", "copy"], output.with_suffix(".mp4"), cancel_check)


def resize(video_path: str, width: int, height: int, output: Path, cancel_check=None) -> str:
    ffmpeg = resolve_media_binary("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Wan2GP FFmpeg binary was not found.")
    return _ffmpeg_output([ffmpeg, "-y", "-v", "error", "-i", video_path, "-vf", f"scale={int(width)}:{int(height)}", "-c:v", "libx264", "-c:a", "aac"], output.with_suffix(".mp4"), cancel_check)


def fps(video_path: str, value: float, output: Path, cancel_check=None) -> str:
    ffmpeg = resolve_media_binary("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Wan2GP FFmpeg binary was not found.")
    return _ffmpeg_output([ffmpeg, "-y", "-v", "error", "-i", video_path, "-vf", f"fps={float(value):g}", "-c:v", "libx264", "-c:a", "aac"], output.with_suffix(".mp4"), cancel_check)


def image_sequence(image_paths: list[str], fps_value: float, output: Path, cancel_check=None) -> str:
    """Encode connected IMAGE[] files into a video using Wan2GP's FFmpeg."""
    paths = [str(path) for path in image_paths if Path(path).is_file()]
    if not paths:
        raise ValueError("Image Sequence needs at least one image.")
    ffmpeg = resolve_media_binary("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Wan2GP FFmpeg binary was not found.")
    duration = 1.0 / max(0.1, float(fps_value))
    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as handle:
        concat_file = Path(handle.name)
        for path in paths:
            escaped = path.replace("'", "'\\''")
            handle.write(f"file '{escaped}'\n")
            handle.write(f"duration {duration:.6f}\n")
        # The concat demuxer applies the last duration only when the final
        # file is repeated. Without it a one-image sequence becomes an empty
        # or zero-duration MP4 on some FFmpeg builds.
        escaped = paths[-1].replace("'", "'\\''")
        handle.write(f"file '{escaped}'\n")
    try:
        return _ffmpeg_output([
            ffmpeg, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-t", f"{duration * len(paths):.6f}",
            "-vf", f"fps={max(0.1, float(fps_value)):g}", "-vsync", "cfr", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        ], output.with_suffix(".mp4"), cancel_check)
    finally:
        concat_file.unlink(missing_ok=True)
