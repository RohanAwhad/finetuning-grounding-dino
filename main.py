import torch
torch.set_float32_matmul_precision('high')

import math
import os
import torch
import wandb

from transformers import AutoModelForZeroShotObjectDetection

import engine
from dataset import CustomDataloader


ROOT_DIR = '/home/rawhad/personal_jobs/GUI_Detection/rico'
SAVE_DIR = os.path.join(ROOT_DIR, '/screen_ai/processed_data')

MODEL_CKPT = "IDEA-Research/grounding-dino-base"
MODEL_PATH = os.path.join(ROOT_DIR, 'grounding_dino_screen_ai_model')

device = 'cpu'
if torch.cuda.is_available(): device = 'cuda'


# create optimizer dataloader and scheduler
max_lr = 3e-5
min_lr = max_lr * 0.1
warmup_steps = 92 # 0.037 * 2500
max_steps = 2500
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


def configure_optimizers(lr, weight_decay, model):
  # from andrej karpathy's Train GPT-2 from scratch video

  param_dict = {pn: p for pn, p in model.named_parameters()}
  param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

  # only decay weights for parameters with 2 or more dims
  decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
  nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
  optim_groups = [
    {"params": decay_params, "weight_decay": weight_decay},
    {"params": nodecay_params, "weight_decay": 0.0},
  ]

  # if fused AdamW is installed, use it
  import inspect
  fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
  use_fused = fused_available and 'cuda' in device
  print('Using fused AdamW:', use_fused)
  optimizer = torch.optim.AdamW(optim_groups, lr=lr, eps=1e-8, fused=use_fused)
  return optimizer

TOTAL_BATCH_SIZE = 32
BATCH_SIZE = 8
assert TOTAL_BATCH_SIZE % BATCH_SIZE == 0, "Total batch size must be divisible by batch size"
GRAD_ACCUM_STEPS = TOTAL_BATCH_SIZE // BATCH_SIZE

TRAIN_DATALOADER = CustomDataloader('train', BATCH_SIZE)
VALID_DATALOADER = CustomDataloader('valid', BATCH_SIZE)


# create model
model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_CKPT)
print('Model loaded')

model.to(device)
optimizer = configure_optimizers(max_lr, 0.01, model)

print('Starting training ...')
# setup wandb logger
wandb.init(project="grounding-dino-screen-ai", name='test-5')
engine.run(model, TRAIN_DATALOADER, VALID_DATALOADER, optimizer, get_lr, num_steps=max_steps, val_every_n_steps=200, val_steps=20, grad_accum_steps=GRAD_ACCUM_STEPS, device=device, logger=wandb, model_path=MODEL_PATH)
wandb.finish()

print(f'Saving model to {MODEL_PATH}')
model.cpu()
model.save_pretrained(MODEL_PATH)
print('Model saved')
