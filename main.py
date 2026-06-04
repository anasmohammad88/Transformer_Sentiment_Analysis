import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from data.dataset import IMDBDataset
from data.preprocessing import TextPreprocessor, clean_text
from data.tokenizer import Tokenizer
from data.vocabulary import Vocabulary
from sklearn.model_selection import train_test_split
from model.transformer import TransformerEncoder
from model.transformer_classifier import TransformerClassifier
from training.config import Config
import torch.optim as optim
from training.trainer import Trainer


def main():
    """
    Main training pipeline.
    """

    # 1. Load dataset
    df = pd.read_csv("data/IMDB Dataset.csv")
    text = df["review"].tolist()
    label = df["sentiment"].tolist()
    X_train, X_valid, y_train, y_valid = train_test_split(text, label, test_size=0.2, random_state=42)
    
    # 2. Initialize tokenizer
    tokenizer = Tokenizer()
    
    # 3. Build vocabulary
    vocab = Vocabulary(Config.VOCAB_SIZE, Config.MIN_FREQUENCY)
    vocab.build_vocabulary(X_train, tokenizer)
    
    
    sample = X_train[0]
    print("\nRAW:")
    print(sample)

    cleaned = clean_text(sample)
    print("\nCLEANED:")
    print(cleaned)

    tokens = tokenizer.tokenize(cleaned)
    print("\nTOKENS:")
    print(tokens)

    encoded = vocab.encode(tokens)
    print("\nENCODED:")
    print(encoded)

    decoded = vocab.decode(encoded)
    print("\nDECODED:")
    print(decoded)
    print("\nVOCAB SIZE:")
    print(len(vocab))

    # 4. Create preprocessor
    preprocessor = TextPreprocessor(tokenizer, vocab)
    
    # 5. Create datasets
    train_data = IMDBDataset(X_train, y_train, preprocessor, Config.MAX_SEQUENCE_LENGTH)
    valid_data = IMDBDataset(X_valid, y_valid, preprocessor, Config.MAX_SEQUENCE_LENGTH)
    
    # 6. Create dataloaders
    train_loader = DataLoader(dataset=train_data, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)
    valid_loader = DataLoader(dataset=valid_data, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)
    
    # 7. Initialize model
    transformer_encoder = TransformerEncoder(len(vocab), Config.D_MODEL,Config.NUM_LAYERS,Config.NUM_HEADS,Config.HIDDEN_DIM,Config.DROPOUT, Config.MAX_SEQUENCE_LENGTH)
    model = TransformerClassifier(encoder=transformer_encoder, d_model=Config.D_MODEL)
    
    # 8. Initialize optimizer and loss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(params=model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=Config.PATIENCE)
    
    # 9. Initialize trainer
    trainer = Trainer(model, train_loader, valid_loader, optimizer, criterion, scheduler)
    
    # 10. Start training
    trainer.train(Config.EPOCHS)
    
if __name__ == "__main__":
    main()