import os
import argparse
import time
import json
import numpy as np
import torch
from model import ModelArgs, Transformer
from tokenizer import get_encoding

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--max_new_tokens', type=int, default=200, help='Number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=200, help='Top-k sampling value')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--tokenizer_name', type=str, default='sp_model', help='Name of the SentencePiece model')
    return parser.parse_args()

def load_model(checkpoint_path, device):
    print(f"🔄 Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})

    model_args = ModelArgs(
        dim=config.get('dim', 4096),
        n_layers=config.get('n_layers', 32),
        n_heads=config.get('n_heads', 32),
        n_kv_heads=config.get('n_kv_heads', None),
        vocab_size=config.get('vocab_size', 32000),
        hidden_dim=config.get('hidden_dim', None),
        multiple_of=config.get('multiple_of', 256),
        norm_eps=config.get('norm_eps', 1e-5),
        max_seq_len=config.get('max_seq_len', 2048),
        dropout=0.0,
        condition_dim=config.get('condition_dim', 768),
        condition_proj_dim=config.get('condition_proj_dim', None),
    )

    model = Transformer(model_args)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    return model, config

def generate_text(model, tokenizer, prompt, condition=None, max_new_tokens=200, temperature=0.8, top_k=200, device='cuda'):
    prompt_tokens = tokenizer.encode(prompt)
    input_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)

    if condition is not None:
        condition_tensor = torch.tensor(condition, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        condition_tensor = None

    print("🚀 Generating...")
    start_time = time.time()
    with torch.no_grad():
        output_tokens = model.generate(
            input_tokens,
            max_new_tokens=max_new_tokens,
            condition=condition_tensor,
            temperature=temperature,
            top_k=top_k
        )

    output_text = tokenizer.decode(output_tokens[0].tolist())
    elapsed = time.time() - start_time
    print(f"✅ Generated in {elapsed:.2f}s")

    return output_text

def validate_continuation(generated_text):
    if '=' not in generated_text:
        print("⚠️ No '=' found in output. Skipping validation.")
        return

    pair_map = {
        'A': 'W', 'B': 'X', 'C': 'Y', 'D': 'Z',
        'a': 'w', 'b': 'x', 'c': 'y', 'd': 'z'
    }

    prompt_part, continuation_part = generated_text.split('=', 1)

    correct = 0
    total = min(len(prompt_part), len(continuation_part))

    for i in range(total):
        expected = pair_map.get(prompt_part[i], None)
        if expected is not None and continuation_part[i] == expected:
            correct += 1

    accuracy = 100 * correct / total if total > 0 else 0
    print(f"✅ Match accuracy: {correct}/{total} = {accuracy:.2f}%")

def main():
    args = get_args()
    device = torch.device(args.device)

    model, config = load_model(args.checkpoint, device)
    tokenizer = get_encoding(args.tokenizer_name)

    print("💡 Model is ready.")
    print(f"📊 Model expects condition vectors of dimension: {config['condition_dim']}")
    print("🔁 Type 'quit' anytime to exit.\n")

    while True:
        prompt = input("📝 Enter a starting prompt: ").strip()
        if prompt.lower() == "quit":
            break

        condition = np.ones(1024)

        output = generate_text(
            model,
            tokenizer,
            prompt,
            condition=condition,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device
        )

        print("\n📜 Generated Text:")
        print(output)
        validate_continuation(output)
        print("-" * 50)

if __name__ == "__main__":
    main()

