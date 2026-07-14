# mini-gpt-from-scratch

A decoder-only, GPT-2–style transformer language model implemented **from scratch** in PyTorch (no `transformers` / `nanoGPT` library used) — built as a hands-on study of tokenization, self-attention, and autoregressive sequence modeling, inspired by Andrej Karpathy's *"Let's build GPT"* series.

The model is intentionally kept small (a few million parameters by default) so it can be trained on a laptop/Colab GPU in minutes, but the architecture is config-driven and can be scaled up to the full **GPT-2 124M** configuration by changing a handful of numbers in `config.py`.

---

## Features

- Character-level tokenizer (simple, dependency-free) — swappable for a BPE tokenizer
- Multi-head causal self-attention implemented from first principles (no `nn.MultiheadAttention`)
- Pre-norm Transformer blocks with residual connections, exactly like GPT-2
- Configurable model size — scale from ~1M params to 124M by editing one file
- Training script with train/val loss tracking and checkpointing
- Text generation script with temperature and top-k sampling

## Architecture

```
Input text
   │
   ▼
Tokenizer (char-level) ──► token ids
   │
   ▼
Token Embedding + Positional Embedding
   │
   ▼
┌─────────────────────────────┐
│  Transformer Block  × N     │
│  ┌────────────────────────┐│
│  │ LayerNorm               ││
│  │ Multi-Head Self-Attention││   (causal masked)
│  │ + residual               ││
│  │ LayerNorm                ││
│  │ Feed-Forward (MLP)       ││
│  │ + residual                ││
│  └────────────────────────┘│
└─────────────────────────────┘
   │
   ▼
LayerNorm → Linear (lm_head) → logits over vocab
   │
   ▼
Softmax → next-token probability distribution
```

## Repo structure

```
mini-gpt-from-scratch/
├── config.py              # all hyperparameters (model size, training, etc.)
├── model/
│   ├── attention.py       # Head, MultiHeadAttention
│   ├── transformer.py     # FeedForward, Block
│   └── gpt.py             # full GPT model (embeddings + blocks + head)
├── tokenizer/
│   └── char_tokenizer.py  # character-level encode/decode
├── data/
│   └── input.txt          # <-- put your training text here
├── train.py                # training loop
├── generate.py             # load a checkpoint and sample text from it
├── checkpoints/             # saved model weights land here
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone <this-repo-url>
cd mini-gpt-from-scratch
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

## Usage

1. **Add your dataset**: drop any plain-text file at `data/input.txt` (English, Hindi, code-mixed — anything works with the char-level tokenizer).
2. **Configure model size** in `config.py` (defaults are a ~10M-param "mini" model that trains fast on a single GPU/CPU).
3. **Train**:
   ```bash
   python train.py
   ```
   This prints train/val loss every `eval_interval` steps and saves a checkpoint to `checkpoints/model.pt`.
4. **Generate text**:
   ```bash
   python generate.py --prompt "Once upon a time" --max_new_tokens 300
   ```

## Scaling to full GPT-2 (124M)

Edit `config.py`:
```python
n_embd  = 768
n_head  = 12
n_layer = 12
block_size = 1024
```
(Needs a real GPU with several GB of VRAM and a much larger dataset to train meaningfully — the default "mini" config is meant for fast local experimentation.)

## Key concepts implemented (for reference)

| Concept | Where | Notes |
|---|---|---|
| Tokenization | `tokenizer/char_tokenizer.py` | maps characters ↔ integer ids |
| Token + positional embeddings | `model/gpt.py` | gives the model identity + order of tokens |
| Causal self-attention | `model/attention.py` | each token attends only to past tokens (upper-triangular mask) |
| Multi-head attention | `model/attention.py` | multiple attention "views" run in parallel, concatenated |
| Feed-forward network | `model/transformer.py` | per-token MLP after attention |
| Residual connections + LayerNorm | `model/transformer.py` | pre-norm style, stabilizes deep training |
| Autoregressive generation | `generate.py` | sample one token at a time, feed it back in |

## Possible extensions (bonus scope)

- Swap the character tokenizer for a BPE tokenizer (e.g. `tiktoken`) for better sample efficiency
- Train on multilingual / code-mixed (Hindi-English) data
- Add rotary positional embeddings (RoPE) instead of learned positional embeddings
- Add KV-caching to speed up generation
- Scale to the full 124M GPT-2 config and compare loss curves

## Acknowledgements

Architecture and training approach inspired by Andrej Karpathy's public educational material on building GPT from scratch (nanoGPT / "Let's build GPT" lecture). This implementation was written independently for learning purposes.
