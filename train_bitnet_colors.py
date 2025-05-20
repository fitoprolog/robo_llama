import os
import time
import math
import json
import argparse
import pickle
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
import sentencepiece as spm
from torch.distributed import init_process_group, destroy_process_group

from bitnet_model import BitModelArgs, BitTransformer
from dataloader import create_dataloader

class ColorDataset(Dataset):
    def __init__(self, text_path, embeddings_path, tokenizer_path, max_seq_len=256):
        self.text_data = []
        with open(text_path, 'r') as f:
            for line in f:
                self.text_data.append(json.loads(line))
        
        self.embeddings = np.load(embeddings_path)
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(tokenizer_path)
        self.max_seq_len = max_seq_len
        
        # Verify embeddings shape
        if self.embeddings.shape[1] != 3:
            raise ValueError(f"Expected RGB values (3 dimensions), got {self.embeddings.shape[1]} dimensions")
        
    def __len__(self):
        return len(self.text_data)
    
    def pad_sequence(self, seq, pad_id=0):
        if len(seq) > self.max_seq_len:
            return seq[:self.max_seq_len]
        return seq + [pad_id] * (self.max_seq_len - len(seq))
    
    def __getitem__(self, idx):
        item = self.text_data[idx]
        text = item['text']
        embedding = self.embeddings[idx]
        
        # Tokenize text
        tokens = self.tokenizer.encode_as_ids(text)
        tokens = self.pad_sequence(tokens)
        tokens = torch.tensor(tokens, dtype=torch.long)
        
        # Create targets (shifted by 1 position)
        targets = torch.roll(tokens, shifts=-1)
        targets[-1] = -1  # Mask the last target
        
        # Ensure condition vector has the correct shape (batch_size, condition_dim)
        condition = torch.tensor(embedding, dtype=torch.float32).view(-1)  # Flatten to 1D
        if condition.shape[0] != 3:  # RGB values
            raise ValueError(f"Expected condition vector of length 3 (RGB), got {condition.shape[0]}")
        
        return {
            'tokens': tokens,
            'targets': targets,
            'condition': condition
        }

def create_dataloader(text_path, embeddings_path, tokenizer_path, max_seq_len=256, batch_size=32, num_workers=4):
    dataset = ColorDataset(
        text_path=text_path,
        embeddings_path=os.path.join(os.path.dirname(embeddings_path), 'conditions.npy'),  # Use conditions.npy instead
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

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
        text_path=os.path.join(args.data_dir, 'text_data.jsonl'),
        embeddings_path=os.path.join(args.data_dir, 'embeddings.npy'),
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    model_args = BitModelArgs(
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
        condition_proj_dim=args.condition_proj_dim or args.dim,  # Default to model dim if not specified
    )
    model = BitTransformer(model_args)
    
    # Set up model for training
    model.to(device)
    
    # Compile model with PyTorch 2.0
    if args.compile and device_type == 'cuda':
        model = torch.compile(model)
    
    # Use DDP for distributed training
    if args.ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )
    
    # Load checkpoint if provided
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Loaded checkpoint from {args.resume}, resuming from iteration {iter_num}")
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
        conditions = batch['condition'].to(device)  
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
        conditions = batch['condition'].to(device)
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

def get_args():
    parser = argparse.ArgumentParser()
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='color_dataset', help='Directory containing the dataset')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to SentencePiece model')
    parser.add_argument('--out_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    
    # Model parameters
    parser.add_argument('--dim', type=int, default=512, help='Model dimension')
    parser.add_argument('--n_layers', type=int, default=6, help='Number of layers')
    parser.add_argument('--n_heads', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--n_kv_heads', type=int, default=None, help='Number of key/value heads (optional)')
    parser.add_argument('--vocab_size', type=int, default=32000, help='Vocabulary size')
    parser.add_argument('--multiple_of', type=int, default=256, help='Hidden dimension must be multiple of this')
    parser.add_argument('--norm_eps', type=float, default=1e-5, help='Layer norm epsilon')
    parser.add_argument('--max_seq_len', type=int, default=2048, help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--condition_dim', type=int, default=3, help='Dimension of condition vectors (RGB)')
    parser.add_argument('--condition_proj_dim', type=int, default=None, help='Projection dimension for conditions')
    
    # Training parameters
    parser.add_argument('--tokenizer_name', type=str, default='sp_model', help='Name of the SentencePiece model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')
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
    parser.add_argument('--eval_only', action='store_true', help='Only run evaluation')
    
    # Distributed training parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--ddp', action='store_true', help='Use DDP for distributed training')
    
    # Logging and saving
    parser.add_argument('--log_interval', type=int, default=1, help='Log training stats every N steps')
    parser.add_argument('--save_interval', type=int, default=1000, help='Save checkpoint every N steps')
    parser.add_argument('--eval_interval', type=int, default=500, help='Evaluate every N steps')
    parser.add_argument('--eval_iters', type=int, default=100, help='Number of iterations for evaluation')
    parser.add_argument('--resume', type=str, default=None, help='Resume training from checkpoint')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    train(args) 