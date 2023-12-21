
from accelerate.utils import write_basic_config
write_basic_config()
import os
os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'
from huggingface_hub import snapshot_download
cache_dir = "dog"
snapshot_download(
    "diffusers/dog-example",
    cache_dir=cache_dir,
    repo_type="dataset",
    ignore_patterns=".gitattributes",
)

