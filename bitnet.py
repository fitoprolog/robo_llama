import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def quantize_weights(weights: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Quantize weights to ternary values {-1, 0, 1} using a threshold.
    
    Args:
        weights: Input weights tensor
        scale: Scaling factor for the weights
    Returns:
        Quantized weights tensor
    """
    # Scale the weights
    weights = weights * scale
    
    # Calculate threshold based on weight statistics
    threshold = 0.7 * torch.mean(torch.abs(weights))
    
    # Quantize to ternary values
    quantized = torch.zeros_like(weights)
    quantized[weights > threshold] = 1.0
    quantized[weights < -threshold] = -1.0
    
    return quantized

def quantize_activations(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """
    Quantize activations to a specified number of bits.
    
    Args:
        x: Input activation tensor
        bits: Number of bits for quantization
    Returns:
        Quantized activation tensor
    """
    # Calculate scale factor
    scale = (2 ** bits - 1) / (torch.max(x) - torch.min(x))
    
    # Quantize
    x_quantized = torch.round(x * scale) / scale
    
    return x_quantized

class BitLinear(nn.Module):
    """
    BitNet linear layer with ternary weights and quantized activations.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        
        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize scale factor
        self.scale = nn.Parameter(torch.ones(1))
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize weights with normal distribution
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize weights
        weight_quantized = quantize_weights(self.weight, self.scale)
        
        # Linear transformation
        output = F.linear(x, weight_quantized, self.bias)
        
        # Quantize activations
        output = quantize_activations(output)
        
        return output

class BitAttention(nn.Module):
    """
    BitNet attention layer with ternary weights and quantized activations.
    """
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        # Query, Key, Value projections
        self.q_proj = BitLinear(dim, dim)
        self.k_proj = BitLinear(dim, dim)
        self.v_proj = BitLinear(dim, dim)
        
        # Output projection
        self.out_proj = BitLinear(dim, dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scale factor for attention scores
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, values
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Compute output
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        
        # Final projection
        output = self.out_proj(output)
        
        return output

class BitFeedForward(nn.Module):
    """
    BitNet feed-forward network with ternary weights and quantized activations.
    """
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = BitLinear(dim, hidden_dim)
        self.w2 = BitLinear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w1(x)
        x = F.gelu(x)  # GELU activation
        x = self.dropout(x)
        x = self.w2(x)
        return x

class BitTransformerBlock(nn.Module):
    """
    BitNet transformer block with ternary weights and quantized activations.
    """
    def __init__(self, dim: int, n_heads: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attention = BitAttention(dim, n_heads, dropout)
        self.feed_forward = BitFeedForward(dim, hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Attention block
        attn_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward block
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x 