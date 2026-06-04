from torch import nn

from model.embeddings import InputEmbeddings
from model.encoder import EncoderLayer
from model.positional_encoding import PositionalEncoding

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, hidden_dim, dropout, max_seq_length):
        super().__init__()
        self.embedding = InputEmbeddings(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, hidden_dim, dropout) for _ in range(num_layers)])

    def forward(self, x, src_mask):
        x = self.embedding(x)
        x = self.dropout(self.positional_encoding(x))
        for layer in self.layers:
            x = layer(x, src_mask)
        return x