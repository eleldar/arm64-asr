import tempfile
from pathlib import Path

import streamlit as st
from streamlit_mic_recorder import mic_recorder

from src.state import remember_tmp, clear_recording, clear_upload
from src.utils.audio import wav_duration_from_bytes

def render_record_section() -> None:
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
        # при новой записи гарантируем очистку загруженного файла
        clear_upload()

        st.session_state.rec_id = audio["id"]
        st.session_state.rec_bytes = audio["bytes"]
        st.session_state.rec_duration = wav_duration_from_bytes(audio["bytes"]) or 0.0
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio["bytes"])
            st.session_state.rec_path = Path(tmp.name)
        remember_tmp(st.session_state.rec_path)

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
            clear_recording()
            st.rerun()
            st.stop()

