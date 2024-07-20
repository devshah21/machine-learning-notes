import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create a matrix of shape (max_len, d_model) containing the positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super(TransformerEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model, max_len)

    def forward(self, x):
        token_embeddings = self.token_embedding(x)
        embeddings = self.position_encoding(token_embeddings)
        return embeddings

# Example usage
vocab_size = 30522  # vocabulary size
d_model = 512       # Embedding size
max_len = 100       # Maximum sequence length

embedding_layer = TransformerEmbedding(vocab_size, d_model, max_len)
input_ids = torch.tensor([[101, 19204, 2135, 1567, 2003, 2019, 2590, 3350, 1012, 102]])  # Example input
embeddings = embedding_layer(input_ids)

print(embeddings.shape)  # Output shape should be (sequence_length, batch_size, d_model)
