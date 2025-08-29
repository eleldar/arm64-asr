import logging
import time
from pathlib import Path
from typing import Optional

import streamlit as st

from src.state import (
    init_state,
    clear_upload,
    clear_recording,
    cleanup_tmp,
)
from src.ui.record import render_record_section
from src.ui.upload import render_upload_section
from src.services.pipeline import run_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="TMG", layout="centered")
st.title("Документирование встреч")

init_state()

# --- Переключатель режима с очисткой встречного состояния ---
mode = st.segmented_control(
    options=("🎙️ Записать", "📁 Загрузить"),
    label="Режим",
    selection_mode="single",
    default=st.session_state.get("mode", "🎙️ Записать"),
    key="mode",
)

prev_mode = st.session_state.get("_prev_mode")
if prev_mode is None:
    st.session_state._prev_mode = mode
elif prev_mode != mode:
    # симметричная очистка при смене режима
    if mode == "🎙️ Записать":
        clear_upload()
    else:
        clear_recording()
    cleanup_tmp()
    st.session_state._prev_mode = mode
    st.rerun()

# --- Контент по режимам ---
if mode == "🎙️ Записать":
    render_record_section()
else:
    render_upload_section()

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
    # простая эвристика времени обработки
    st.session_state.est_sec = 0.43 * source_dur + 60  # 10 мин аудио → ~5 мин
    st.session_state.t_start = time.time()
    st.session_state.job = st.session_state.exec.submit(run_pipeline, source_path)

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
        logger.info("transcript_text len=%s, resume_text len=%s", len(transcript_text), len(resume_text))

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
        cleanup_tmp()

