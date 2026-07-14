

import os
import torch

import config
from model import MiniGPT
from tokenizer.char_tokenizer import CharTokenizer

torch.manual_seed(config.SEED)


def load_data():
    with open(config.DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    tok = CharTokenizer.from_text(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)

    n = int(config.TRAIN_SPLIT * len(data))
    train_data, val_data = data[:n], data[n:]
    return tok, train_data, val_data


def get_batch(split, train_data, val_data):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - config.BLOCK_SIZE, (config.BATCH_SIZE,))
    x = torch.stack([data[i : i + config.BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + config.BLOCK_SIZE + 1] for i in ix])
    return x.to(config.DEVICE), y.to(config.DEVICE)


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(config.EVAL_ITERS)
        for k in range(config.EVAL_ITERS):
            xb, yb = get_batch(split, train_data, val_data)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    if not os.path.exists(config.DATA_PATH):
        raise FileNotFoundError(
            f"No training data found at '{config.DATA_PATH}'. "
            f"Add a plain-text file there before running train.py."
        )

    tok, train_data, val_data = load_data()
    print(f"Dataset: {len(train_data) + len(val_data):,} chars | vocab size: {tok.vocab_size}")

    model = MiniGPT(
        vocab_size=tok.vocab_size,
        block_size=config.BLOCK_SIZE,
        n_embd=config.N_EMBD,
        n_head=config.N_HEAD,
        n_layer=config.N_LAYER,
        dropout=config.DROPOUT,
    ).to(config.DEVICE)
    print(f"Model parameters: {model.num_params() / 1e6:.2f}M | device: {config.DEVICE}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    for it in range(config.MAX_ITERS):
        if it % config.EVAL_INTERVAL == 0 or it == config.MAX_ITERS - 1:
            losses = estimate_loss(model, train_data, val_data)
            print(f"step {it:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

        xb, yb = get_batch("train", train_data, val_data)
        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        optimizer.step()

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "chars": tok.chars,
            "config": {
                "block_size": config.BLOCK_SIZE,
                "n_embd": config.N_EMBD,
                "n_head": config.N_HEAD,
                "n_layer": config.N_LAYER,
                "dropout": config.DROPOUT,
            },
        },
        config.CHECKPOINT_PATH,
    )
    print(f"Checkpoint saved to {config.CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
