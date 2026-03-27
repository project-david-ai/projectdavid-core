# From inside WSL or your container
from huggingface_hub import snapshot_download

snapshot_download(repo_id="Qwen/Qwen2.5-VL-3B-Instruct")
