

import torch

# ---- data ----
DATA_PATH = "data/input.txt"
TRAIN_SPLIT = 0.9

# ---- model architecture ----
BLOCK_SIZE = 128     # context length (max sequence length model can attend over)
N_EMBD = 256         # embedding dimension (d_model)
N_HEAD = 8            # number of attention heads (N_EMBD must be divisible by N_HEAD)
N_LAYER = 6           # number of transformer blocks
DROPOUT = 0.1

# ---- training ----
BATCH_SIZE = 64
MAX_ITERS = 3000
EVAL_INTERVAL = 300
EVAL_ITERS = 50
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

# ---- checkpointing ----
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = "checkpoints/model.pt"

# ---- misc ----
SEED = 1337
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# ------------------------------------------------------------------
# Full GPT-2 (124M) config — uncomment and use to scale up.
# Needs a proper GPU (several GB VRAM) and a large dataset.
# ------------------------------------------------------------------
# BLOCK_SIZE = 1024
# N_EMBD     = 768
# N_HEAD     = 12
# N_LAYER    = 12
# BATCH_SIZE = 12          # reduce to fit in GPU memory at this scale
