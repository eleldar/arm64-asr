# ASR

## Install
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip uninstall torch torchvision torchaudio
uv sync
uv run src/main.py
```
