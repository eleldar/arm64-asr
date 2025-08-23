import gc
import torch
import whisperx
from pathlib import Path
from tempfile import NamedTemporaryFile

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

file = Path(".") / "tests" / "datasets" / "short.mp3"
file = Path(".") / "tests" / "datasets" / "long.mp4"

model = "large-v3"
device = "cuda"
batch_size = 6 # 16
compute_type = "float16"
temperature=0
prompt=None
language="ru"
modeldir="models"


audio_file = str(file.resolve())
with open(file, 'rb') as f:
    with NamedTemporaryFile() as tempfile:
        tempfile.write(f.read())
        audio = whisperx.load_audio(tempfile.name)

# transcribe
model = whisperx.load_model(
    model, 
    device=device,
    compute_type=compute_type,
    asr_options={"temperatures": temperature, "initial_prompt": prompt},
    download_root=modeldir
)
result = model.transcribe(audio, batch_size=batch_size, language=language, task="transcribe")
gc.collect()
torch.cuda.empty_cache()
del model

# align
model, metadata = whisperx.load_align_model(
    language_code=result["language"], device=device, model_dir=modeldir
)
result = whisperx.align(
    result["segments"], model, metadata, audio, device, return_char_alignments=False, print_progress=True
)
gc.collect()
torch.cuda.empty_cache()
del model
gc.collect()
torch.cuda.empty_cache()
if "model" in locals() or "model" in globals():
    del model
gc.collect()


print(result["segments"])
