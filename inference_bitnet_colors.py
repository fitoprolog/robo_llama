import os
import json
import torch
import numpy as np
import sentencepiece as spm
import argparse
from bitnet_model import BitModelArgs, BitTransformer

def generate_color_embedding(rgb_values):
    """Generate a color embedding from RGB values"""
    r, g, b = rgb_values
    # Simply return the RGB values as a numpy array
    return np.array([r, g, b])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to SentencePiece model')
    parser.add_argument('--r', type=float, required=True, help='Red value (0-1)')
    parser.add_argument('--g', type=float, required=True, help='Green value (0-1)')
    parser.add_argument('--b', type=float, required=True, help='Blue value (0-1)')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=200, help='Top-k sampling parameter')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(args.tokenizer_path)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model_args = BitModelArgs(
        dim=512,
        n_layers=6,
        n_heads=6,
        vocab_size=len(tokenizer),
        condition_dim=3,  # RGB values
        condition_proj_dim=512,
        max_seq_len=2048,
        dropout=0.0,
        norm_eps=1e-5,
        multiple_of=256
    )
    
    model = BitTransformer(model_args)
    model.load_state_dict(checkpoint['model'])
    model.to(args.device)
    model.eval()

    # Generate color embedding
    rgb_values = (args.r, args.g, args.b)
    condition = generate_color_embedding(rgb_values)
    condition = torch.tensor(condition, dtype=torch.float32, device=args.device).unsqueeze(0)

    # Generate text
    with torch.no_grad():
        # Start with a single token (BOS)
        tokens = torch.tensor([[tokenizer.bos_id()]], dtype=torch.long, device=args.device)
        
        for _ in range(args.max_new_tokens):
            # Forward pass
            logits = model(tokens, condition=condition)
            logits = logits[:, -1, :] / args.temperature
            
            # Top-k sampling
            if args.top_k is not None:
                v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            tokens = torch.cat([tokens, next_token], dim=1)
            
            # Stop if we generate EOS token
            if next_token.item() == tokenizer.eos_id():
                break

    # Decode and print generated text
    generated_text = tokenizer.decode(tokens[0].tolist())
    print(f"\nGenerated description for RGB({args.r:.2f}, {args.g:.2f}, {args.b:.2f}):")
    print(generated_text)

if __name__ == '__main__':
    main() 