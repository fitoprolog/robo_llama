import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tokenizer import get_encoding

class MultimodalDataset(Dataset):
    """Dataset for training a multimodal LLM with paired text and conditioning vectors"""
    
    def __init__(
        self, 
        text_path: str,
        condition_path: str,
        tokenizer_name: str = "robollama",  # default to use our SentencePiece model
        max_length: int = 2048,
        condition_dim: int = 1024,  # RGB values
    ):
        """
        Args:
            text_path: Path to text data file (jsonl format where each line has a 'text' field)
            condition_path: Path to conditioning vectors (numpy files matching the text data)
            tokenizer_name: Name of the SentencePiece model to use
            max_length: Maximum sequence length
            condition_dim: Dimension of conditioning vectors (3 for RGB)
        """
        self.max_length = max_length
        self.condition_dim = condition_dim
        
        # Load tokenizer
        self.tokenizer = get_encoding(tokenizer_name)
        
        # Load text data
        self.texts = []
        with open(text_path, 'r', encoding='utf-8') as f:
            self.texts = json.load(f)
        for t in self.texts:
            assert len(t) > 0
        # Load condition vectors
        if condition_path.endswith('.npy'):
            # Single file with all vectors
            self.conditions = np.load(condition_path)
        else:
            # Directory with individual files
            self.conditions = []
            for i in range(len(self.texts)):
                cond_file = os.path.join(condition_path, f"{i}.npy")
                if os.path.exists(cond_file):
                    self.conditions.append(np.load(cond_file))
                else:
                    # If condition file doesn't exist, use zeros
                    self.conditions.append(np.zeros(condition_dim))
            
            self.conditions = np.stack(self.conditions)
        
        assert len(self.texts) == len(self.conditions), "Number of texts and conditions must match"
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Get text and tokenize
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text)
        
        # Truncate or pad tokens to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Get conditioning vector
        condition = self.conditions[idx].astype(np.float32)
        
        return {
            'tokens': torch.tensor(tokens, dtype=torch.long),
            'condition': torch.tensor(condition, dtype=torch.float32),
        }

class BlindMultimodalDataset(Dataset):
    """Dataset for training a multimodal LLM with paired text and fixed condition vectors"""

    def __init__(
        self, 
        text_path: str,
        tokenizer_name: str = "robollama",
        max_length: int = 2048,
        condition_dim: int = 1024,
    ):
        self.max_length = max_length
        self.condition_dim = condition_dim

        # Load tokenizer
        self.tokenizer = get_encoding(tokenizer_name)
        self.pad_token_id = getattr(self.tokenizer, "pad_token_id", 0)

        # Load text data
        self.texts = []
        with open(text_path, 'r', encoding='utf-8') as f:
            self.texts=json.load(f)
        for t in self.texts:
            assert len(t) > 0

        # Fixed zero conditioning vector
        self.fixed_condition = torch.ones(self.condition_dim, dtype=torch.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text)
        # Pad or truncate tokens
        #if len(tokens) > self.max_length:
        #    tokens = tokens[:self.max_length]
        return {
            'tokens': torch.tensor(tokens, dtype=torch.long),
            'condition': self.fixed_condition.clone(),  # clone to prevent shared ref
        }


def create_dataloader(
    text_path: str,
    condition_path: str,
    batch_size: int,
    max_length: int = 2048,
    tokenizer_name: str = "robollama",
    condition_dim: int = 1024,  # RGB values
    shuffle: bool = True,
    num_workers: int = 4,
):
    """Create a DataLoader for multimodal training data"""
    
    dataset = BlindMultimodalDataset(
        text_path=text_path,
        #condition_path=condition_path,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        condition_dim=condition_dim,
    )
    
    def collate_fn(batch):
        # Get max sequence length in this batch
        max_len = max([item['tokens'].size(0) for item in batch])
        
        # Prepare tensors
        tokens = torch.zeros((len(batch), max_len), dtype=torch.long)
        # Stack conditions and ensure they are 2D [batch_size, condition_dim]
        conditions = torch.stack([item['condition'] for item in batch])
        if conditions.dim() == 3:  # If conditions are [batch_size, seq_len, condition_dim]
            conditions = conditions.mean(dim=1)  # Average over sequence dimension
        
        # Create targets (shifted tokens) and fill tokens
        targets = torch.zeros((len(batch), max_len), dtype=torch.long).fill_(-1)  # -1 is ignored in loss
        
        for i, item in enumerate(batch):
            seq_len = item['tokens'].size(0)
            tokens[i, :seq_len] = item['tokens']
            
            # Target is the next token (shifted by 1)
            if seq_len > 1:
                targets[i, :seq_len-1] = item['tokens'][1:]
        
        return {
            'tokens': tokens,
            'conditions': conditions,
            'targets': targets,
        }

    def collate_fn(batch):
        # Get max sequence length in this batch
        max_len = max([item['tokens'].size(0) for item in batch])
        
        # Prepare tensors with correct pad token
        tokens = torch.full((len(batch), max_len), 0, dtype=torch.long)
        targets = torch.full((len(batch), max_len), -1, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        
        # Stack conditions and ensure they are 2D
        conditions = torch.stack([item['condition'] for item in batch])
        if conditions.dim() == 3:
            conditions = conditions.mean(dim=1)
        
        for i, item in enumerate(batch):
            seq_len = item['tokens'].size(0)
            tokens[i, :seq_len] = item['tokens']
            attention_mask[i, :seq_len] = 1  # Mark non-padded positions
            
            # Create targets (next token prediction)
            if seq_len > 1:
                targets[i, :seq_len - 1] = item['tokens'][1:]
        
        return {
            'tokens': tokens,
            'conditions': conditions,
            'targets': targets,
            'attention_mask': attention_mask,
        }
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


# Pre-processing functions

def preprocess_text_file(
    input_file: str,
    output_jsonl: str,
    chunk_size: int = 2048,
    overlap: int = 0,
    tokenizer_name: str = "sp_model",
):
    """
    Preprocess a raw text file into jsonl format with chunking
    
    Args:
        input_file: Path to input text file
        output_jsonl: Path to output jsonl file
        chunk_size: Maximum chunk size in tokens
        overlap: Number of tokens to overlap between chunks
        tokenizer_name: Name of the SentencePiece model to use
    """
    # Load tokenizer
    tokenizer = get_encoding(tokenizer_name)
    
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Tokenize entire text
    tokens = tokenizer.encode(text)
    
    # Chunk into sections
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i:i + chunk_size]
        if len(chunk_tokens) < 10:  # Skip very small chunks at the end
            continue
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append({"text": chunk_text})
    
    # Write to jsonl
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + '\n')
    
    print(f"Processed {len(tokens)} tokens into {len(chunks)} chunks")


def create_dummy_conditions(
    jsonl_file: str,
    output_file: str,
    condition_dim: int = 3,  # RGB values
    random_init: bool = False,
):
    """
    Create dummy condition vectors for testing
    
    Args:
        jsonl_file: Path to jsonl file with text data
        output_file: Path to output numpy file for conditions
        condition_dim: Dimension of condition vectors (3 for RGB)
        random_init: If True, initialize with random RGB values; otherwise with zeros
    """
    # Count lines in jsonl file
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        count = sum(1 for _ in f)
    
    # Create condition vectors
    if random_init:
        # Generate random RGB values between 0 and 1
        conditions = np.random.uniform(0, 1, (count, condition_dim)).astype(np.float32)
    else:
        conditions = np.zeros((count, condition_dim), dtype=np.float32)
    
    # Save to file
    np.save(output_file, conditions)
    print(f"Created {count} RGB condition vectors")
