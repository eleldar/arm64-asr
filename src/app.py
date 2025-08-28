import mimetypes
import struct
import subprocess
import tempfile
import time
import wave
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Optional, Tuple

import streamlit as st
from streamlit_mic_recorder import mic_recorder

from src.config import state
from src.summarizer import summary
from src.transcriber import transcribe

# ---------------- helpers ----------------

logger = logging.getLogger(__name__)

def _wav_duration_from_bytes(data: bytes) -> Optional[float]:
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


def _duration_seconds(path: Path) -> Optional[float]:
    if path.suffix.lower() in (".wav", ".wave"):
        try:
            with wave.open(str(path), "rb") as wf:
                return wf.getnframes() / float(wf.getframerate())
        except Exception:
            return None
    try:
        out = subprocess.check_output(
            ["ffprobe", "-i", str(path), "-show_entries", "format=duration", "-v", "quiet", "-of", "csv=p=0"],
            stderr=subprocess.STDOUT,
        )
        return float(out.decode().strip())
    except Exception:
        return None


def _extract_audio_if_video(path: Path) -> Path:
    if path.suffix.lower() not in {".mp4", ".mov", ".mkv", ".webm"}:
        return path
    wav = path.with_suffix(".wav")
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


def _run_pipeline(audio_path: Path) -> Tuple[str, str]:
    t = transcribe(str(audio_path))
    logger.info(f"{t=}")
    s = summary(t)
    logger.info(f"{s=}")
    return t, s


def _init_state():
    s = st.session_state
    if "exec" not in s:
        s.exec = ThreadPoolExecutor(max_workers=1)
    s.setdefault("job", None)
    s.setdefault("t_start", 0.0)
    s.setdefault("est_sec", 0.0)
    s.setdefault("tmp_files", [])
    # запись
    s.setdefault("rec_id", None)
    s.setdefault("rec_bytes", None)
    s.setdefault("rec_duration", 0.0)
    s.setdefault("rec_path", None)
    s.setdefault("mic_key", 0)
    # загрузка
    s.setdefault("upload_path", None)
    s.setdefault("upload_duration", 0.0)
    s.setdefault("uploader_key", 0)


def _cleanup_tmp():
    for p in st.session_state.tmp_files:
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass
    st.session_state.tmp_files.clear()


def _clear_recording():
    st.session_state.rec_id = None
    st.session_state.rec_bytes = None
    st.session_state.rec_duration = 0.0
    rp = st.session_state.rec_path
    if rp:
        try:
            Path(rp).unlink(missing_ok=True)
        except Exception:
            pass
    st.session_state.rec_path = None
    st.session_state.mic_key += 1


def _clear_upload():
    _cleanup_tmp()  # удалит все, что накопили
    st.session_state.upload_path = None
    st.session_state.upload_duration = 0.0
    st.session_state.uploader_key += 1


# ----------- UI -----------

st.set_page_config(page_title="TMG", layout="centered")
st.title("Документирование встреч")

_init_state()

tab_rec, tab_upload = st.tabs(["🎙️ Записать", "📁 Загрузить"])

# --- Запись речи ---
with tab_rec:
    st.caption("Нажми, чтобы начать/остановить запись.")
    left, spacer, right = st.columns([3, 6, 3], vertical_alignment="center")
    with left:
        audio = mic_recorder(
            start_prompt="⏺️ Записать",
            stop_prompt="⏹️ Стоп",
            just_once=False,
            use_container_width=True,
            format="wav",
            key=f"mic_rec_{st.session_state.mic_key}",
        )
    if audio and (st.session_state.rec_id != audio["id"]):
        st.session_state.rec_id = audio["id"]
        st.session_state.rec_bytes = audio["bytes"]
        st.session_state.rec_duration = _wav_duration_from_bytes(audio["bytes"]) or 0.0
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio["bytes"])
            st.session_state.rec_path = Path(tmp.name)
        st.session_state.tmp_files.append(st.session_state.rec_path)
    if st.session_state.rec_bytes:
        st.audio(st.session_state.rec_bytes, format="audio/wav")
        st.caption(f"Записано: ~{int(st.session_state.rec_duration)} сек")
    with right:
        if st.button(
            "Очистить запись",
            disabled=not bool(st.session_state.rec_bytes),
            use_container_width=True,
            key="clear_rec_btn",
        ):
            _clear_recording()
            st.rerun()
            st.stop()

# --- Загрузка файла ---
with tab_upload:
    up = st.file_uploader(
        "Выбери аудио/видео",
        type=["wav", "mp3", "m4a", "ogg", "aac", "mp4", "mov", "mkv", "webm"],
        accept_multiple_files=False,
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if up is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix or ".wav") as tmp:
            tmp.write(up.getbuffer())
            orig_path = Path(tmp.name)

        try:
            new_path = _extract_audio_if_video(orig_path)
            # если это было видео — удаляем исходный временный файл с видео
            if new_path != orig_path:
                try:
                    orig_path.unlink(missing_ok=True)
                except Exception:
                    pass
            st.session_state.upload_path = new_path
            st.session_state.tmp_files.append(new_path)  # храним только актуальный путь
        except Exception:
            # при ошибке — подчистить исходник
            try:
                orig_path.unlink(missing_ok=True)
            except Exception:
                pass
            st.stop()

        st.session_state.upload_duration = _duration_seconds(st.session_state.upload_path) or 0.0
        try:
            with open(st.session_state.upload_path, "rb") as f:
                up_bytes = f.read()
        except Exception:
            st.error("Не удалось прочитать загруженный файл.")
            st.stop()

        mime, _ = mimetypes.guess_type(str(st.session_state.upload_path))
        st.audio(up_bytes, format=mime or "audio/wav")

        # сброс записи
        st.session_state.rec_id = None
        st.session_state.rec_bytes = None
        st.session_state.rec_duration = 0.0
        st.session_state.rec_path = None
    if st.button(
        "Очистить файл", disabled=st.session_state.upload_path is None, key="clear_upload_btn", use_container_width=True
    ):
        _clear_upload()
        st.rerun()
        st.stop()

# --- выбор источника и запуск пайплайна ---
source_path: Optional[Path] = st.session_state.rec_path or st.session_state.upload_path
source_dur: float = st.session_state.rec_duration or st.session_state.upload_duration or 0.0

start_btn = st.button(
    "Обработать",
    type="primary",
    use_container_width=True,
    disabled=(source_path is None) or (st.session_state.job is not None),
)

if start_btn and (source_path is not None) and (st.session_state.job is None):
    st.session_state.est_sec = 0.43 * source_dur + 60 # 10 мин аудио → 5 мин
    st.session_state.t_start = time.time()
    st.session_state.job = st.session_state.exec.submit(_run_pipeline, source_path)

# --- прогресс/результат ---
if st.session_state.job is not None:
    progress = st.progress(0, text="Обработка…")
    hard_timeout = st.session_state.est_sec * 4
    try:
        while True:
            if st.session_state.job.done():
                break
            elapsed = time.time() - st.session_state.t_start
            denom = st.session_state.est_sec or 20.0
            pct = min(99, int((elapsed / denom) * 100))
            progress.progress(pct, text=f"Обработка… ~{int(max(0, denom - elapsed))} сек")
            time.sleep(0.5)
        transcript_text, resume_text = st.session_state.job.result(timeout=5)
        progress.progress(100, text="")
        logger.info(f"{transcript_text=}")
        logger.info(f"{resume_text=}")
        
        combined = (
            "АННОТАЦИЯ\n"
            + resume_text.strip()
            + "\n\n────────────────────────────────────────\n\n"
            + "ПРОТОКОЛ\n"
            + transcript_text.strip()
        )
        st.text_area("Результат", combined, height=600, label_visibility="hidden")
    except TimeoutError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Ошибка: {type(e).__name__}: {e}")
    finally:
        st.session_state.job = None
        _cleanup_tmp()
