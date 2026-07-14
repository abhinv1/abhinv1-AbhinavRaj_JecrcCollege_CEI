

import argparse
import torch

import config
from model import MiniGPT
from tokenizer.char_tokenizer import CharTokenizer


def load_model(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=config.DEVICE)
    tok = CharTokenizer(chars=ckpt["chars"])

    model = MiniGPT(
        vocab_size=tok.vocab_size,
        block_size=ckpt["config"]["block_size"],
        n_embd=ckpt["config"]["n_embd"],
        n_head=ckpt["config"]["n_head"],
        n_layer=ckpt["config"]["n_layer"],
        dropout=ckpt["config"]["dropout"],
    ).to(config.DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, tok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=config.CHECKPOINT_PATH)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    args = parser.parse_args()

    model, tok = load_model(args.checkpoint)

    if args.prompt:
        idx = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=config.DEVICE)
    else:
        idx = torch.zeros((1, 1), dtype=torch.long, device=config.DEVICE)

    out = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
