import torch
import whisperx
from pathlib import Path

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

example = Path(".") / "tests" / "datasets" / "short.mp3"

device = "cuda"
audio_file = str(example)
batch_size = 6 # 16
compute_type = "float16"
model = whisperx.load_model("large-v3", device, compute_type=compute_type)
audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
print(result["segments"])
