import tempfile
from pathlib import Path

import streamlit as st

from src.state import remember_tmp, clear_recording
from src.utils.audio import extract_audio_if_video, duration_seconds, guess_mime

ACCEPT = ["wav", "mp3", "m4a", "ogg", "aac", "mp4", "mov", "mkv", "webm"]

def render_upload_section() -> None:
    up = st.file_uploader(
        "Выбери аудио/видео",
        type=ACCEPT,
        accept_multiple_files=False,
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if up is not None:
        # при загрузке гарантируем очистку текущей записи
        clear_recording()

        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix or ".wav") as tmp:
            tmp.write(up.getbuffer())
            orig_path = Path(tmp.name)

        try:
            new_path = extract_audio_if_video(orig_path)
            if new_path != orig_path:
                try:
                    orig_path.unlink(missing_ok=True)
                except Exception:
                    pass
            st.session_state.upload_path = new_path
            remember_tmp(new_path)
        except Exception:
            try:
                orig_path.unlink(missing_ok=True)
            except Exception:
                pass
            st.stop()

        st.session_state.upload_duration = duration_seconds(st.session_state.upload_path) or 0.0
        try:
            with open(st.session_state.upload_path, "rb") as f:
                up_bytes = f.read()
        except Exception:
            st.error("Не удалось прочитать загруженный файл.")
            st.stop()

        st.audio(up_bytes, format=guess_mime(st.session_state.upload_path))

    if st.button(
        "Очистить файл",
        disabled=st.session_state.upload_path is None,
        key="clear_upload_btn",
        use_container_width=True,
    ):
        # фактическая очистка делается в app.cleanup_tmp() после обработки/в finally
        st.session_state.upload_path = None
        st.session_state.upload_duration = 0.0
        st.session_state.uploader_key += 1
        st.rerun()
        st.stop()

