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
    parser.add_argument('--condition_vector', type=str, default=None, help='Path to condition vector (numpy file)')
    parser.add_argument('--prompt', type=str, default='', help='Text prompt to start generation')
    parser.add_argument('--max_new_tokens', type=int, default=500, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=200, help='Sample from top k probable tokens')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--tokenizer_name', type=str, default='sp_model', help='Name of the SentencePiece model')
    parser.add_argument('--output_file', type=str, default=None, help='Path to save output')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    return parser.parse_args()

def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    
    # Create model args
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
        dropout=0.0,  # No dropout during inference
        condition_dim=config.get('condition_dim', 768),
        condition_proj_dim=config.get('condition_proj_dim', None),
    )
    
    # Create model
    model = Transformer(model_args)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    return model, config

def load_condition_vector(condition_path):
    """Load conditioning vector from file."""
    if condition_path is None:
        return None
    
    condition = np.load(condition_path).astype(np.float32)
    if condition.ndim == 1:
        # Single vector
        return condition
    else:
        # Multiple vectors, use the first one
        return condition[0]

def generate_text(model, tokenizer, prompt, condition=None, max_new_tokens=500, temperature=0.8, top_k=200, device='cuda'):
    """Generate text from prompt and condition."""
    # Tokenize prompt
    prompt_tokens = tokenizer.encode(prompt)
    print(f"Prompt has {len(prompt_tokens)} tokens")
    
    # Prepare input tokens
    input_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    # Prepare condition if provided
    if condition is not None:
        condition_tensor = torch.tensor(condition, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        condition_tensor = None
    
    # Generate
    print("Generating...")
    start_time = time.time()
    with torch.no_grad():
        output_tokens = model.generate(
            input_tokens, 
            max_new_tokens=max_new_tokens, 
            condition=condition_tensor,
            temperature=temperature, 
            top_k=top_k
        )
    
    # Decode output
    output_text = tokenizer.decode(output_tokens[0].tolist())
    
    generation_time = time.time() - start_time
    tokens_per_sec = max_new_tokens / generation_time
    
    print(f"\nGenerated {max_new_tokens} tokens in {generation_time:.2f}s ({tokens_per_sec:.2f} tokens/s)")
    
    return output_text

def interactive_mode(model, tokenizer, condition=None, device='cuda'):
    """Interactive generation mode."""
    print("\n===== Interactive Mode =====")
    print("Enter a prompt to generate text. Type 'exit' to quit.")
    
    while True:
        prompt = input("\nPrompt> ")
        if prompt.lower() == 'exit':
            break
        
        try:
            # Get generation parameters
            max_tokens = int(input("Max tokens to generate (default 200): ") or 200)
            temp = float(input("Temperature (default 0.8): ") or 0.8)
            top_k = int(input("Top-k (default 200): ") or 200)
            
            # Generate
            output = generate_text(
                model, 
                tokenizer, 
                prompt, 
                condition=condition,
                max_new_tokens=max_tokens, 
                temperature=temp,
                top_k=top_k, 
                device=device
            )
            
            print("\n----- Generated Text -----")
            print(output)
            print("-------------------------")
            
        except KeyboardInterrupt:
            print("\nGeneration interrupted")
        except Exception as e:
            print(f"Error: {e}")

def main():
    args = get_args()
    
    # Set device
    device = torch.device(args.device)
    
    # Load model
    model, config = load_model(args.checkpoint, device)
    
    # Load tokenizer
    tokenizer = get_encoding(args.tokenizer_name)
    
    # Load condition vector
    condition = load_condition_vector(args.condition_vector)
    if condition is not None:
        print(f"Loaded condition vector with shape {condition.shape}")
    else:
        print("No condition vector provided, running in text-only mode")
    
    if args.interactive:
        # Interactive mode
        interactive_mode(model, tokenizer, condition, device)
    else:
        # Single generation
        output = generate_text(
            model,
            tokenizer,
            args.prompt,
            condition=condition,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device
        )
        
        print("\n----- Generated Text -----")
        print(output)
        print("-------------------------")
        
        # Save output if requested
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'prompt': args.prompt,
                    'generated_text': output,
                    'parameters': {
                        'temperature': args.temperature,
                        'top_k': args.top_k,
                        'max_new_tokens': args.max_new_tokens,
                        'has_condition': condition is not None
                    }
                }, f, ensure_ascii=False, indent=2)
            print(f"Output saved to {args.output_file}")

if __name__ == '__main__':
    main()
