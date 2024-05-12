import torch
import os

use_cpu = not torch.cuda.is_available() or not os.environ.get("USE_CUDA")
device = torch.device("cpu" if use_cpu else "cuda")
