import torch
import argparse
import sentencepiece as spm
from bitnet_model import BitModelArgs, BitTransformer

def generate_color_description(model, tokenizer, rgb_values, device='cuda', max_tokens=50, temperature=0.6):
    # Prepare condition (RGB values)
    condition = torch.tensor(rgb_values, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Start with BOS token if it exists, otherwise use first token
    start_token = torch.tensor([[1]], dtype=torch.long).to(device)  # Usually 1 is BOS token in SentencePiece
    
    # Generate text
    with torch.no_grad():
        output = model.generate(
            start_token,
            max_new_tokens=max_tokens,
            condition=condition,
            temperature=temperature,
            top_k=50  # Increased top_k for more diversity
        )
    
    try:
        # Filter out any token IDs that are out of vocabulary range
        valid_tokens = [t for t in output[0].cpu().tolist() if t < tokenizer.vocab_size()]
        # Remove any special tokens
        valid_tokens = [t for t in valid_tokens if t > 2]  # Skip BOS, EOS, and PAD tokens
        # Decode the generated tokens
        text = tokenizer.decode(valid_tokens)
        return text.strip()
    except Exception as e:
        print(f"Error decoding tokens: {e}")
        print(f"Raw tokens: {output[0].cpu().tolist()}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to tokenizer model')
    parser.add_argument('--red', type=float, required=True, help='Red value (0-1)')
    parser.add_argument('--green', type=float, required=True, help='Green value (0-1)')
    parser.add_argument('--blue', type=float, required=True, help='Blue value (0-1)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(args.tokenizer_path)
    print(f"Loaded tokenizer with vocabulary size: {tokenizer.vocab_size()}")

    # Load model
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = checkpoint['config']
    
    model_args = BitModelArgs(
        dim=config['dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        vocab_size=32000,  # Use the same vocab size as training
        condition_dim=config['condition_dim'],
        condition_proj_dim=config['condition_proj_dim']
    )
    
    model = BitTransformer(model_args)
    model.load_state_dict(checkpoint['model'])
    model.to(args.device)
    model.eval()

    # Generate description
    rgb_values = [args.red, args.green, args.blue]
    print(f"\nGenerating description for RGB values: ({args.red:.2f}, {args.green:.2f}, {args.blue:.2f})")
    description = generate_color_description(model, tokenizer, rgb_values, args.device)
    
    if description:
        print(f"Generated description: {description}")
    else:
        print("Failed to generate description")

if __name__ == '__main__':
    main() 