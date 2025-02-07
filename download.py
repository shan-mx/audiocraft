from huggingface_hub import hf_hub_download, snapshot_download

import audiocraft

MODEL_NAME = "facebook/magnet-small-30secs"


hf_hub_download(
    repo_id=MODEL_NAME,
    filename="state_dict.bin",
    library_name="audiocraft",
    library_version=audiocraft.__version__,
)


hf_hub_download(
    repo_id=MODEL_NAME,
    filename="compression_state_dict.bin",
    library_name="audiocraft",
    library_version=audiocraft.__version__,
)

snapshot_download(
    repo_id="t5-base",
    allow_patterns=[
        "config.json",
        "spiece.model",
        "tokenizer.json",
        "model.safetensors",
    ],
    library_name="transformers",
)
