import os
import gc
import logging
from typing import Any, Dict, List
import torch
import whisperx
import numpy as np
from pathlib import Path
from time import time
from datetime import datetime

from src.config import state

logger = logging.getLogger(__name__)

def transcribe(path: Path, offset: float | None = None) -> str:
    offset = offset if offset else os.stat(path).st_ctime
    audio = _get_array(path)
    transcription = _align(audio, _transcribe(audio))
    transcription = _diarize(audio, transcription)
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    segments = _filter(transcription.get("segments", []))
    result = _get_result(segments, offset)
    text = result.get("text", "")
    logger.info(f"{text=}")
    return text

def _get_array(path: Path) -> np.ndarray:
    logger.info("Start get array")
    try:
        audio = whisperx.load_audio(str(path))
        return audio
    except Exception as error:
        logger.exception(f"Failed get array: {error=}")
        return np.array([])
    finally:
        logger.info("Finish get array")


def _transcribe(audio: np.ndarray) -> Dict[str, Any]:
    logger.info("Start transcribe")
    transcription = {"language": state.language_code, "segments": []}
    try:
        model = whisperx.load_model(
            state.model_mapping.get(state.model, "large-v3"), 
            device=state.device,
            compute_type=state.compute_type,
            asr_options={"temperatures": state.temperature, "initial_prompt": state.prompt},
            download_root=str(state.recognition_model_dir)
        )
        result = model.transcribe(
            audio=audio, batch_size=state.batch_size, 
            language=state.language_code, task="transcribe"
        )
        gc.collect()
        torch.cuda.empty_cache()
        while ("model" in locals() or "model" in globals()):
            del model
        transcription = dict(result)
    except Exception as error:
        logger.exception(f"Failed transcribe: {error=}")
    finally:
        logger.info(f"Finish transcribe: {transcription=}")
        return transcription
        

def _align(audio: np.ndarray, transcription: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Start align")
    align = transcription
    try:
        segments = transcription.get("segments", [])
        language_code = transcription.get("language", state.language_code)
        model, metadata = whisperx.load_align_model(
            language_code=language_code, 
            device=state.device, 
            model_dir=state.alignment_model_dir
        )
        result = whisperx.align(
            segments, model, metadata, audio, state.device, 
            return_char_alignments=False, print_progress=True
        )
        gc.collect()
        torch.cuda.empty_cache()
        while ("model" in locals() or "model" in globals()):
            del model
        align = dict(result) if result.get("segments") else transcription
    except Exception as error:
        logger.exception(f"Failed align: {error=}")
    finally:
        logger.info(f"Finish align: {align=}")
        return align

def _diarize(audio: np.ndarray, transcription: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Start diarize")
    diarize = transcription
    try:
        model = whisperx.DiarizationPipeline(
            model_name=str(state.diarization_model_path), 
            device=state.device
        )
        diarize_segments = model(audio) # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
        result = whisperx.assign_word_speakers(diarize_segments, transcription)
        gc.collect()
        torch.cuda.empty_cache()
        while ("model" in locals() or "model" in globals()):
            del model
        diarize = result
    except Exception as error:
        logger.exception(f"Failed diarize: {error=}")
    finally:
        logger.info(f"Finish diarize: {diarize=}")
        return diarize
    
def _filter(transcription: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    logger.info("Start filter")
    filter = []
    try:
        result = [
            {'speaker': row['speaker'], 'text': row['text'].strip(), 'start': row.get('start', 0.0), 'end': row.get('end', 0.0)} 
            for row in transcription
            if (
                row.get('text','').strip() and row.get('speaker') and
                not any([word.lower() in row['text'].lower() for word in state.hallucination_words])
            )    
        ]
        filter = result
    except Exception as error:
        logger.exception(f"Failed filter: {error=}")
        return []
    finally:
        logger.info(f"Finish filter: {filter=}")
        return filter

def _format_time(seconds: float) -> str:
    dt = datetime.fromtimestamp(seconds)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def _format(segments: List[Dict[str, Any]], offset: float) -> str:
    result = []
    current_speaker = None
    current_text = []
    start_time, end_time = None, None

    for entry in segments:
        speaker, text, start, end = entry["speaker"], entry["text"], entry["start"], entry["end"]
        if speaker != current_speaker:
            if current_speaker is not None:
                start_time = start_time + offset if start_time else offset
                end_time = end_time + offset if end_time else offset
                result.append(
                    f"{current_speaker} [{_format_time(start_time)} – {_format_time(end_time)}]\n"
                    + " ".join(current_text) + "\n"
                )
            current_speaker = speaker
            current_text = [text]
            start_time, end_time = start, end
        else:
            current_text.append(text)
            end_time = end
    if current_speaker is not None:
        start_time = start_time + offset if start_time else offset
        end_time = end_time + offset if end_time else offset
        result.append(
            f"{current_speaker} [{_format_time(start_time)} – {_format_time(end_time)}]\n"
            + " ".join(current_text) + "\n"
        )
    return "\n".join(result)


def _get_result(
    segments: List[Dict[str, Any]], 
    offset: float, default_speaker_name="SPEAKER_01"
) -> Dict[str, Any]:
    speakers = []
    for segment in segments:
        speaker = segment.get("speaker", default_speaker_name)
        if speaker not in speakers:
            speakers.append(speaker)
    speaker_map = {
        speaker: f"SPEAKER_{str(e).zfill(2)}"
        for e, speaker in enumerate(speakers, start=1)
    }
    sentences = []
    for segment in segments:
        segment["speaker"] = speaker_map.get(segment["speaker"], default_speaker_name)
        sentences.append(segment.get("text"))
    return {
        "segments": segments,
        "text": _format(segments, offset),
    }   

if __name__ == "__main__":
    file = Path(".") / "tests" / "datasets" / "short.mp3"
    # file = Path(".") / "tests" / "datasets" / "long.mp4"
    offset = time()
    transcription = transcript(file, offset)
    assert "text" in transcription
    assert isinstance(transcription.get("text"), str)
    text = transcription.get("text", "")
    assert len(text) > 0
    assert "segments" in transcription
    assert isinstance(transcription.get("segments"), list)
    assert len(transcription.get("segments", [])) > 0
    for segment in transcription.get("segments", []):
        assert {"speaker", "text", "start", "end"} == set(segment)
        assert isinstance(segment.get("speaker"), str)
        assert isinstance(segment.get("text"), str)
        assert isinstance(segment.get("start"), float)
        assert isinstance(segment.get("end"), float)
        assert segment.get("end") > segment.get("start")
    print(transcription['segments'])
    print(transcription['text'])
