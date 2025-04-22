import os
import time
import math
import pickle
import argparse
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import ModelArgs, Transformer
from dataloader import create_dataloader, preprocess_text_file, create_dummy_conditions

def get_args():
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument('--dim', type=int, default=288, help='Model dimension')
    parser.add_argument('--n_layers', type=int, default=6, help='Number of layers')
    parser.add_argument('--n_heads', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--n_kv_heads', type=int, default=6, help='Number of key/value heads (if None, use n_heads)')
    parser.add_argument('--vocab_size', type=int, default=50257, help='Vocabulary size')
    parser.add_argument('--multiple_of', type=int, default=256, help='Hidden dimension will be multiple of this value')
    parser.add_argument('--norm_eps', type=float, default=1e-5, help='Normalization epsilon')
    parser.add_argument('--max_seq_len', type=int, default=2048, help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    
    # Multimodal parameters
    parser.add_argument('--condition_dim', type=int, default=6, help='Dimension of conditioning vector')
    parser.add_argument('--condition_proj_dim', type=int, default=288, help='Projection dimension for condition (if None, use model dim)')
    
    # Data parameters 
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--text_path', type=str, required=True, help='Path to text data file (jsonl)')
    parser.add_argument('--condition_path', type=str, required=True, help='Path to condition vectors')
    parser.add_argument('--tokenizer_name', type=str, default='gpt2', help='Name of tokenizer')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size per GPU')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--max_iters', type=int, default=20000, help='Maximum number of training iterations')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=6e-4, help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=6e-5, help='Minimum learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.95, help='Adam beta2')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--decay_lr', action='store_true', help='Decay learning rate')
    parser.add_argument('--warmup_iters', type=int, default=1000, help='Warmup iterations')
    
    # System parameters
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--dtype', type=str, default='bfloat16', help='Data type (float32, bfloat16, or float16)')
    parser.add_argument('--compile', action='store_true', help='Use PyTorch 2.0 compiler')
    parser.add_argument('--grad_checkpoint', action='store_true', help='Use gradient checkpointing')
    
    # Distributed training parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--ddp', action='store_true', help='Use DDP for distributed training')
    
    # Logging and saving
    parser.add_argument('--log_interval', type=int, default=1, help='Log training stats every N steps')
    parser.add_argument('--eval_interval', type=int, default=1000, help='Evaluate every N steps')
    parser.add_argument('--eval_iters', type=int, default=200, help='Number of iterations for evaluation')
    parser.add_argument('--save_interval', type=int, default=1000, help='Save checkpoint every N steps')
    parser.add_argument('--out_dir', type=str, default='out', help='Output directory')
    parser.add_argument('--eval_only', action='store_true', help='Only run evaluation')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Load checkpoint from file')
    
    # Preprocessing
    parser.add_argument('--preprocess', action='store_true', help='Run preprocessing')
    parser.add_argument('--raw_text_file', type=str, default=None, help='Raw text file to preprocess')
    parser.add_argument('--create_dummy_conditions', action='store_true', help='Create dummy condition vectors')
    parser.add_argument('--random_conditions', action='store_true', help='Initialize dummy conditions with random values')
    
    args = parser.parse_args()
    return args

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
    
    # Context to use during training
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype_str]
    ctx = nullcontext() if dtype_str == 'float32' else torch.amp.autocast(device_type='cuda', dtype=ptdtype)
    
    return dtype, ctx

def get_lr(it, args):
    """Learning rate scheduler with warmup and cosine decay"""
    # Warmup
    if it < args.warmup_iters:
        return args.learning_rate * it / args.warmup_iters
    # Cosine decay if enabled
    if args.decay_lr:
        decay_ratio = (it - args.warmup_iters) / (args.max_iters - args.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return args.min_lr + coeff * (args.learning_rate - args.min_lr)
    # Constant learning rate
    return args.learning_rate

def train(args):
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize distributed training if needed
    if args.ddp:
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        assert args.gradient_accumulation_steps % ddp_world_size == 0
        args.gradient_accumulation_steps //= ddp_world_size
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        device = args.device

    # Set up output directory
    if master_process:
        os.makedirs(args.out_dir, exist_ok=True)
    
    # Set up device and dtype
    torch.manual_seed(args.seed + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    dtype, ctx = setup_dtype_context(args.dtype)
    
    # Create data loader
    train_loader = create_dataloader(
        text_path=args.text_path,
        condition_path=args.condition_path,
        batch_size=args.batch_size,
        max_length=args.max_seq_len,
        tokenizer_name=args.tokenizer_name,
        condition_dim=args.condition_dim,
        num_workers=args.num_workers,
    )
    
    # Create model
    model_args = ModelArgs(
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        vocab_size=args.vocab_size,
        multiple_of=args.multiple_of,
        norm_eps=args.norm_eps,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        condition_dim=args.condition_dim,
        condition_proj_dim=args.condition_proj_dim,
    )
    model = Transformer(model_args)
    
    # Set up model for training
    model.to(device)
    
    # Compile model with PyTorch 2.0
    if args.compile and device_type == 'cuda':
        model = torch.compile(model)
    
    # Use DDP for distributed training
    if args.ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    
    # Optimizer
    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(args.beta1, args.beta2),
        device_type=device_type
    )
    
    # Load checkpoint if provided
    if args.load_checkpoint:
        checkpoint = torch.load(args.load_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Loaded checkpoint from {args.load_checkpoint}, resuming from iteration {iter_num}")
    else:
        iter_num = 0
        best_val_loss = float('inf')
    
    # Training loop
    raw_model = model.module if args.ddp else model
    
    if args.eval_only:
        model.eval()
        val_loss = evaluate(model, train_loader, ctx, device)
        print(f"Validation loss: {val_loss:.4f}")
        return
    
    # Start training
    print(f"Starting training with a total of {args.max_iters} iterations")
    
    train_iter = iter(train_loader)
    t0 = time.time()
    while True:
        # Termination condition
        if iter_num >= args.max_iters:
            break
        
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        # Move batch to device
        tokens = batch['tokens'].to(device)
        conditions = batch['conditions'].to(device)  
        targets = batch['targets'].to(device)
        
        # Set learning rate
        lr = get_lr(iter_num, args) if args.decay_lr else args.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Forward and backward pass
        model.train()
        with ctx:
            # Forward pass
            logits = model(tokens, condition=conditions, targets=targets)
            loss = raw_model.last_loss
            # Scale loss for gradient accumulation
            loss = loss / args.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation and update
        if (iter_num + 1) % args.gradient_accumulation_steps == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # Update parameters
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        # Logging
        if (iter_num + 1) % args.log_interval == 0 and master_process:
            t1 = time.time()
            dt = t1 - t0
            tokens_per_sec = args.batch_size * args.gradient_accumulation_steps * ddp_world_size * tokens.numel() / dt
            print(f"Iter {iter_num+1}/{args.max_iters} | Loss: {loss.item()*args.gradient_accumulation_steps:.4f} | LR: {lr:.6f} | Tokens/sec: {tokens_per_sec:.2f}")
            t0 = time.time()
        
        # Evaluation
        if (iter_num + 1) % args.eval_interval == 0 and master_process:
            val_loss = evaluate(model, train_loader, ctx, device, args.eval_iters)
            print(f"Validation loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if master_process:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': vars(args),
                    }
                    print(f"Saving best model to {os.path.join(args.out_dir, 'best_checkpoint.pt')}")
                    torch.save(checkpoint, os.path.join(args.out_dir, 'best_checkpoint.pt'))
        
        # Regular checkpoint saving
        if (iter_num + 1) % args.save_interval == 0 and master_process:
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': vars(args),
            }
            print(f"Saving checkpoint to {os.path.join(args.out_dir, f'checkpoint_{iter_num+1}.pt')}")
            torch.save(checkpoint, os.path.join(args.out_dir, f'checkpoint_{iter_num+1}.pt'))
        
        iter_num += 1
    
    # Save final model
    if master_process:
        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
            'config': vars(args),
        }
        print(f"Saving final model to {os.path.join(args.out_dir, 'final_checkpoint.pt')}")
        torch.save(checkpoint, os.path.join(args.out_dir, 'final_checkpoint.pt'))
    
    # Clean up
    if args.ddp:
        destroy_process_group()

@torch.no_grad()
def evaluate(model, dataloader, ctx, device, max_iters=None):
    """Evaluate the model on the provided dataloader"""
    model.eval()
    losses = []
    eval_iter = iter(dataloader)
    
    for k in range(max_iters or 1):
        try:
            batch = next(eval_iter)
        except StopIteration:
            eval_iter = iter(dataloader)
            batch = next(eval_iter)
        
        tokens = batch['tokens'].to(device)
        conditions = batch['conditions'].to(device)
        targets = batch['targets'].to(device)
        
        with ctx:
            logits = model(tokens, condition=conditions, targets=targets)
            raw_model = model.module if isinstance(model, DDP) else model
            loss = raw_model.last_loss
        
        losses.append(loss.item())
        
        if max_iters is None:
            break
    
    model.train()
    return torch.tensor(losses).mean().item()

@torch.no_grad()
def generate_sample(model, prompt_tokens, condition=None, max_new_tokens=100, temperature=0.8, top_k=200, device='cuda'):
    """Generate text sample using the model"""
    model.eval()
    
    # Prepare input tokens
    x = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    # Prepare condition if provided
    if condition is not None:
        condition = torch.tensor(condition, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Generate
    y = model.generate(x, max_new_tokens, condition=condition, temperature=temperature, top_k=top_k)
    
    return y[0].tolist()

def preprocess(args):
    """Run preprocessing steps"""
    if args.preprocess and args.raw_text_file:
        print(f"Preprocessing raw text file: {args.raw_text_file}")
        preprocess_text_file(
            input_file=args.raw_text_file,
            output_jsonl=args.text_path,
            chunk_size=args.max_seq_len,
            tokenizer_name=args.tokenizer_name,
        )
    
    if args.create_dummy_conditions:
        print(f"Creating dummy condition vectors at: {args.condition_path}")
        create_dummy_conditions(
            jsonl_file=args.text_path,
            output_file=args.condition_path,
            condition_dim=args.condition_dim,
            random_init=args.random_conditions,
        )

def main():
    args = get_args()
    
    # Run preprocessing if requested
    if args.preprocess or args.create_dummy_conditions:
        preprocess(args)
    
    # Train or evaluate
    train(args)

if __name__ == '__main__':
    main()
