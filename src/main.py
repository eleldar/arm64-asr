import os
from pathlib import Path

from src.transcriber import transcript

file = Path(".") / "tests" / "datasets" / "short.mp3"

transcription = transcript(file)

print(transcription)
