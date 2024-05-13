import torch
import os

use_cuda = os.environ.get("USE_CUDA")
use_cuda = torch.cuda.is_available() and (
    use_cuda.lower() == "true" if use_cuda is not None else False
)
use_cpu = not use_cuda
device = torch.device("cuda" if use_cuda else "cpu")
evaluation_strategy = os.environ.get("EVALUATE_MODEL")
evaluation_strategy = evaluation_strategy if evaluation_strategy is not None else "no"
save_strategy = os.environ.get("SAVE_STRATEGY")
save_strategy = save_strategy if save_strategy is not None else "steps"
save_steps = os.environ.get("SAVE_STEPS")
save_steps = save_steps if save_steps is not None else (20 if use_cpu else 200)
num_train_epochs = os.environ.get("TRAIN_EPOCHS")
num_train_epochs = num_train_epochs if num_train_epochs is not None else 4
