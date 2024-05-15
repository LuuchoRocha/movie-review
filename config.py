import torch
import os

use_cuda = os.environ.get("USE_CUDA")
use_cuda = torch.cuda.is_available() and (use_cuda is not None and use_cuda.lower() == "true")
use_cpu = not use_cuda
device = torch.device("cuda" if use_cuda else "cpu")
evaluation_strategy = os.environ.get("EVALUATION_STRATEGY")
evaluation_strategy = evaluation_strategy if evaluation_strategy is not None else "no"
save_strategy = os.environ.get("SAVE_STRATEGY")
save_strategy = save_strategy if save_strategy is not None else "steps"
save_steps = os.environ.get("SAVE_STEPS")
save_steps = int(save_steps) if save_steps is not None else (50 if use_cpu else 500)
eval_steps = os.environ.get("EVAL_STEPS")
eval_steps = int(eval_steps) if eval_steps is not None else (50 if use_cpu else 500)
num_train_epochs = os.environ.get("NUM_TRAIN_EPOCHS")
num_train_epochs = int(num_train_epochs) if num_train_epochs is not None else 2
max_steps = os.environ.get("MAX_STEPS")
max_steps = int(max_steps) if max_steps is not None else -1
weight_decay = os.environ.get("WEIGHT_DECAY")
weight_decay = float(weight_decay) if weight_decay is not None else 0.1
learning_rate = os.environ.get("LEARNING_RATE")
learning_rate = float(learning_rate) if learning_rate is not None else 1e-5
batch_size = os.environ.get("BATCH_SIZE")
batch_size = int(batch_size) if batch_size is not None else 4
log_level = os.environ.get("LOG_LEVEL")
log_level = log_level if log_level is not None else "info"
