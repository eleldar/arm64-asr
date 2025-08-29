import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import streamlit as st

logger = logging.getLogger(__name__)

def init_state() -> None:
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

def remember_tmp(p: Path) -> None:
    st.session_state.tmp_files.append(Path(p))

def cleanup_tmp() -> None:
    for p in list(st.session_state.tmp_files):
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass
    st.session_state.tmp_files.clear()

def clear_recording() -> None:
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

def clear_upload() -> None:
    # удаляет загруженный + все вспомогательные временные
    if st.session_state.upload_path:
        try:
            Path(st.session_state.upload_path).unlink(missing_ok=True)
        except Exception:
            pass
    st.session_state.upload_path = None
    st.session_state.upload_duration = 0.0
    st.session_state.uploader_key += 1

