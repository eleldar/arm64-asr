import logging
from pathlib import Path
from typing import Tuple

from src.summarizer import summary
from src.transcriber import transcribe

logger = logging.getLogger(__name__)

def run_pipeline(audio_path: Path) -> Tuple[str, str]:
    t = transcribe(str(audio_path))
    logger.info("transcribe ok, len=%s", len(t))
    s = summary(t)
    logger.info("summary ok, len=%s", len(s))
    return t, s

