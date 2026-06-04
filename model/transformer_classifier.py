from torch import nn

class TransformerClassifier(nn.Module):
    def __init__(self, encoder, d_model):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x, attention_mask):
        x = self.encoder(x, attention_mask)

        # Expand mask
        mask = attention_mask.unsqueeze(-1)

        # Remove padding influence
        masked_x = x * mask

        # Sum embeddings
        summed = masked_x.sum(dim=1)

        # Count valid tokens
        counts = mask.sum(dim=1)

        # Mean pooling
        pooled = summed / counts.clamp(min=1)

        logits = self.classifier(pooled)

        return logits.squeeze(-1)