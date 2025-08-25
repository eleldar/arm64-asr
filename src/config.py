import sys
import logging
from pathlib import Path
from typing import List
import torch

from pydantic import BaseModel, Field

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

class Config(BaseModel):
    model_mapping: dict[str, str] = {
        "openai/whisper-tiny": "tiny",
        "openai/whisper-base": "base",
        "openai/whisper-small": "small",
        "openai/whisper-medium": "medium",
        "openai/whisper-large": "large-v1",
        "openai/whisper-large-v2": "large-v2",
        "openai/whisper-large-v3": "large-v3",
        "openai/whisper-large-v3-turbo": "large-v3-turbo",
    }
    model: str = Field("openai/whisper-large-v3")
    device: str = Field("cuda")
    batch_size: int = Field(6)
    compute_type: str = Field("float16")
    temperature: float = Field(0.0)
    language_code: str = Field("ru")
    prompt: str | None = Field(None)
    recognition_model_dir: Path = Path("/mnt/data/models/recognition").resolve()
    alignment_model_dir: Path = Path("/mnt/data/models/alignment").resolve()
    diarization_model_path: Path = Path("/mnt/data/models/diarization/speaker-diarization-3.1/config.yaml").resolve()

    hallucination_words: List[str] = ["DimaTorzok"]


state = Config()
