from huggingface_hub import hf_hub_download

import audiocraft
from api import MODEL_NAME

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
