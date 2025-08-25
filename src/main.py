import os
from pathlib import Path

from src.transcriber import transcript

file = Path(".") / "tests" / "datasets" / "short.mp3"
file = Path(".") / "tests" / "datasets" / "long.mp4"

transcription = transcript(file)

print(transcription)
