import os
import argparse
import json
import torch
import numpy as np
from typing import Optional, List, Union

from bitnet_model import BitModelArgs, BitTransformer
from tokenizer import SPTokenizer

def get_args():
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to SentencePiece model')
    parser.add_argument('--config', type=str, default=None, help='Path to model config (if different from checkpoint)')
    
    # Generation parameters
    parser.add_argument('--prompt', type=str, default="", help='Input prompt')
    parser.add_argument('--condition_path', type=str, default=None, help='Path to condition vector (e.g., CLIP embedding)')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=200, help='Top-k sampling parameter')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # System parameters
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--dtype', type=str, default='bfloat16', help='Data type (float32, bfloat16, or float16)')
    
    return parser.parse_args()

def setup_dtype_context(dtype_str):
    """Set up dtype for model and dtype context for operations"""
    if dtype_str == 'float32':
        dtype = torch.float32
    elif dtype_str == 'bfloat16':
        dtype = torch.bfloat16
    elif dtype_str == 'float16':
        dtype = torch.float16
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    
    # Context to use during inference
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype_str]
    ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype) if 'cuda' in args.device else nullcontext()
    
    return dtype, ctx

def load_model(checkpoint_path: str, config_path: Optional[str] = None, device: str = 'cuda'):
    """Load model from checkpoint"""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load config
    if config_path:
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = checkpoint.get('config', {})
    
    # Create model args
    model_args = BitModelArgs(
        dim=config.get('dim', 4096),
        n_layers=config.get('n_layers', 32),
        n_heads=config.get('n_heads', 32),
        n_kv_heads=config.get('n_kv_heads', None),
        vocab_size=config.get('vocab_size', 32000),
        multiple_of=config.get('multiple_of', 256),
        norm_eps=config.get('norm_eps', 1e-5),
        max_seq_len=config.get('max_seq_len', 2048),
        dropout=config.get('dropout', 0.0),
        condition_dim=config.get('condition_dim', 768),
        condition_proj_dim=config.get('condition_proj_dim', None),
    )
    
    # Create model
    model = BitTransformer(model_args)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    return model, model_args

def load_condition(condition_path: str, device: str = 'cuda') -> Optional[torch.Tensor]:
    """Load condition vector from file"""
    if condition_path is None:
        return None
    
    # Load condition vector
    if condition_path.endswith('.npy'):
        condition = np.load(condition_path)
    elif condition_path.endswith('.pt'):
        condition = torch.load(condition_path)
    else:
        raise ValueError(f"Unsupported condition file format: {condition_path}")
    
    # Convert to tensor and move to device
    if isinstance(condition, np.ndarray):
        condition = torch.from_numpy(condition)
    condition = condition.to(device)
    
    return condition

def generate_text(
    model: BitTransformer,
    tokenizer: SPTokenizer,
    prompt: str,
    condition: Optional[torch.Tensor] = None,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 200,
    device: str = 'cuda',
) -> str:
    """Generate text from prompt using the model"""
    # Encode prompt
    prompt_tokens = tokenizer.encode(prompt)
    x = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    # Generate
    with torch.no_grad():
        y = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            condition=condition,
            temperature=temperature,
            top_k=top_k,
        )
    
    # Decode generated tokens
    generated_tokens = y[0].tolist()
    generated_text = tokenizer.decode(generated_tokens)
    
    return generated_text

def main():
    args = get_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set up device and dtype
    device = args.device
    dtype, ctx = setup_dtype_context(args.dtype)
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, model_args = load_model(args.checkpoint, args.config, device)
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = SPTokenizer(args.tokenizer_path)
    
    # Load condition if provided
    condition = None
    if args.condition_path:
        print(f"Loading condition from {args.condition_path}")
        condition = load_condition(args.condition_path, device)
    
    # Generate text
    print(f"Generating text with prompt: {args.prompt}")
    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        condition=condition,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    
    # Print results
    print("\nGenerated text:")
    print("-" * 80)
    print(generated_text)
    print("-" * 80)

if __name__ == '__main__':
    main() 