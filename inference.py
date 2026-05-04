"""
Inference script — generate text from a seed phrase.

Usage:
    python generate.py \
        --checkpoint checkpoints/ckpt_1000.pt \
        --prompt "Once upon a time" \
        --max_tokens 200 \
        --temperature 0.8 \
        --top_k 50
"""

import argparse
import torch
torch.set_float32_matmul_precision('high')
from mugpt.tokenization import GPT2Tokenizer, BPETokenizer

from mugpt.models.transformer import DecoderOnlyTransformer, ModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt",     type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--temperature",type=float, default=0.8)
    parser.add_argument("--top_k",      type=int, default=50)
    parser.add_argument("--encoding",      type=str, default="BPE")
    parser.add_argument("--tok_path",      type=str, default=None)
    return parser.parse_args()


@torch.no_grad()
def generate(
    model: DecoderOnlyTransformer,
    input_ids: torch.Tensor,      # [1, T]
    max_tokens: int,
    temperature: float,
    top_k: int,
    device: str,
) -> torch.Tensor:
    model.eval()
    for _ in range(max_tokens):
        # crop to model's context window if needed
        input_ids_cropped = input_ids[:, -model.cfg.seq_len:]
        logits = model(input_ids_cropped)          # [1, T, vocab_size]
        logits = logits[:, -1, :] / temperature    # last token only [1, vocab_size]

        # top-k sampling
        top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
        probs = torch.softmax(top_k_logits, dim=-1)
        next_token = top_k_indices[0, torch.multinomial(probs[0], num_samples=1)]

        input_ids = torch.cat([input_ids, next_token.view(1, 1)], dim=1)

    return input_ids[0]


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config = ModelConfig(**checkpoint["config"])
    model = DecoderOnlyTransformer(model_config).to(device)
    model = torch.compile(model)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from {args.checkpoint}")

    # tokenize prompt
    enc = GPT2Tokenizer() if args.encoding == "gpt2" else BPETokenizer().load_from_file(args.tok_path)
    input_ids = torch.tensor(enc.encode(args.prompt), dtype=torch.long).unsqueeze(0).to(device)

    # generate
    output_ids = generate(
        model=model,
        input_ids=input_ids,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )

    print("\n--- Generated Text ---")
    print(enc.decode(output_ids.tolist()))


if __name__ == "__main__":
    main()