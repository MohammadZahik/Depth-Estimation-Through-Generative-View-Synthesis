import os
import shutil
import torch

# Clear the Torch Hub cache
torch_cache_dir = os.path.expanduser('~/.cache/torch/hub')
if os.path.exists(torch_cache_dir):
    shutil.rmtree(torch_cache_dir)
    print(f"Cleared Torch Hub cache at {torch_cache_dir}")

# Re-download the MiDaS model
try:
    model = torch.hub.load('intel-isl/MiDaS', 'MiDaS')
    print("MiDaS model downloaded successfully.")
except Exception as e:
    print(f"Error downloading the MiDaS model: {e}")
