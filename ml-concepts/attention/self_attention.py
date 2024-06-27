import torch
import torch.nn as nn

import numpy as np

def self_attention(queries, keys, values, mask=None):
    """
    Compute self-attention
    
    Args:
    - queries: numpy array of shape (batch_size, seq_len, d_model)
    - keys: numpy array of shape (batch_size, seq_len, d_model)
    - values: numpy array of shape (batch_size, seq_len, d_model)
    - mask: optional numpy array of shape (batch_size, seq_len, seq_len)
    
    Returns:
    - output: numpy array of shape (batch_size, seq_len, d_model)
    - attention_weights: numpy array of shape (batch_size, seq_len, seq_len)
    """
    
    # Get dimensions
    batch_size, seq_len, d_model = queries.shape
    
    # Compute attention scores
    attention_scores = np.matmul(queries, keys.transpose(0, 2, 1))
    
    # Scale attention scores
    attention_scores = attention_scores / np.sqrt(d_model)
    
    # Apply mask if provided
    if mask is not None:
        attention_scores = np.where(mask == 0, -1e9, attention_scores)
    
    # Compute attention weights
    attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis=-1, keepdims=True)
    
    # Compute output
    output = np.matmul(attention_weights, values)
    
    return output, attention_weights

# Example usage
batch_size = 2
seq_len = 4
d_model = 8

queries = np.random.randn(batch_size, seq_len, d_model)
keys = np.random.randn(batch_size, seq_len, d_model)
values = np.random.randn(batch_size, seq_len, d_model)

output, attention_weights = self_attention(queries, keys, values)

print("Output shape:", output.shape)
print("Attention weights shape:", attention_weights.shape)