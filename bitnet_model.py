import math
import struct
import inspect
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from bitnet import BitLinear, BitTransformerBlock

@dataclass
class BitModelArgs:
    # default hyperparameters for the BitNet model
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000
    hidden_dim: Optional[int] = None
    multiple_of: int = 256  # MLP hidden layer size will be multiple of
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0
    # Multimodal parameters
    condition_dim: int = 768  # Dimension of the conditioning vector (e.g., CLIP embedding)
    condition_proj_dim: int = 4096  # Projection dimension to match model dim

class BitTransformer(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: BitModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        # Multimodal conditioning - project condition vectors to model dimension
        self.condition_projector = BitLinear(
            in_features=params.condition_dim,
            out_features=params.condition_proj_dim,
            bias=True
        )
        
        # Token embeddings
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        
        # Transformer blocks
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(BitTransformerBlock(
                dim=params.dim,
                n_heads=params.n_heads,
                hidden_dim=params.hidden_dim or (4 * params.dim),
                dropout=params.dropout
            ))
        
        # Final layer norm
        self.norm = nn.LayerNorm(params.dim, eps=params.norm_eps)
        
        # Output projection
        self.output = BitLinear(params.dim, params.vocab_size, bias=False)
        
        # Share the unembedding parameters with the embedding parameters
        self.tok_embeddings.weight = self.output.weight  # Weight tying
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Initialize attribute for the loss of the last forward call
        self.last_loss = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, condition: Optional[torch.Tensor] = None, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        
        # Add multimodal conditioning if provided
        if condition is not None:
            # Project condition to the model dimension and add batch dimension if needed
            if condition.dim() == 2:
                condition_proj = self.condition_projector(condition)
            else:
                condition_proj = self.condition_projector(condition.unsqueeze(1))
            
            # Add sequence dimension if needed
            if condition_proj.dim() == 2:
                condition_proj = condition_proj.unsqueeze(1)
            
            # Prepend the condition as the first token in the sequence
            h = torch.cat([condition_proj, h], dim=1)
            # Adjust sequence length for positional embeddings
            seqlen += 1
        
        # Forward through transformer blocks
        for layer in self.layers:
            h = layer(h)
        
        h = self.norm(h)

        if targets is not None:
            # For training, calculate loss on all positions
            # If condition is present, we need to shift prediction targets 
            # to align with the token positions (ignore the condition token)
            logits = self.output(h)
            
            if condition is not None:
                # Only compute loss on the text portion (excluding the condition token)
                # Shift the logits and targets accordingly
                logits = logits[:, 1:, :]  # Remove the first position (condition token)
                
            self.last_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        else:
            # For inference, only forward the output on the last position
            logits = self.output(h[:, [-1], :])  # Note: using list [-1] to preserve the time dim
            self.last_loss = None

        return logits

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, condition=None, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Also accepts a condition tensor for multimodal generation.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond, condition=condition)
            logits = logits[:, -1, :] # crop to just the final time step
            if temperature == 0.0:
                # "sample" the single most likely index
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # pluck the logits at the final step and scale by desired temperature
                logits = logits / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx 