import os
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

# dictors
model = whisperx.diarize.DiarizationPipeline(
    use_auth_token=os.getenv("HF_TOKEN"), 
    device=device
)
# add min/max number of speakers if known
# diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
diarize_segments = model(audio)

gc.collect()
torch.cuda.empty_cache()
if "model" in locals() or "model" in globals():
    del model
gc.collect()

# result
result = whisperx.assign_word_speakers(diarize_segments, result)
with open("result.txt", "w") as f:
    f.write(str(result))
print(diarize_segments)
print(result["segments"])
