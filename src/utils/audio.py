import logging
import mimetypes
import struct
import subprocess
import wave
from pathlib import Path
from typing import Optional

import streamlit as st

logger = logging.getLogger(__name__)

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}

def wav_duration_from_bytes(data: bytes) -> Optional[float]:
    """Минимальный парсер WAV для получения длительности (byte_rate и data_size)."""
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    i, byte_rate, data_size = 12, None, None
    while i + 8 <= len(data):
        chunk_id = data[i : i + 4]
        chunk_sz = struct.unpack("<I", data[i + 4 : i + 8])[0]
        if chunk_id == b"fmt " and chunk_sz >= 16:
            _, _, _, byte_rate, _, _ = struct.unpack("<HHIIHH", data[i + 8 : i + 8 + 16])
        elif chunk_id == b"data":
            data_size = chunk_sz
        i += 8 + chunk_sz
        if byte_rate and data_size:
            break
    return None if not (byte_rate and data_size) else data_size / float(byte_rate)

def duration_seconds(path: Path) -> Optional[float]:
    """Попытка через wave для WAV, иначе ffprobe."""
    if path.suffix.lower() in (".wav", ".wave"):
        try:
            with wave.open(str(path), "rb") as wf:
                return wf.getnframes() / float(wf.getframerate())
        except Exception:
            pass
    try:
        out = subprocess.check_output(
            ["ffprobe", "-i", str(path), "-show_entries", "format=duration", "-v", "quiet", "-of", 'csv=p=0'],
            stderr=subprocess.STDOUT,
        )
        return float(out.decode().strip())
    except Exception:
        return None

def extract_audio_if_video(path: Path) -> Path:
    """Если видео — извлекаем WAV 16kHz mono PCM s16le, иначе возвращаем исходный путь."""
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        return path
    wav = path.with_suffix(".wav")
    # Принудительно кодируем в WAV PCM 16k mono — пригодно для STT
    # ffmpeg -y -i input -vn -acodec pcm_s16le -ar 16000 -ac 1 out.wav
    res = subprocess.run(
        ["ffmpeg", "-y", "-i", str(path), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(wav)],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        st.error("ffmpeg не смог извлечь аудио из видео.")
        st.code(res.stderr or res.stdout)
        raise RuntimeError("ffmpeg failed")
    return wav

def guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "audio/wav"

