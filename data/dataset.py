from torch.utils.data import Dataset
import torch

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, preprocessor, max_length=256):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        """
        Return:
            processed_text,
            label
        """
        # Get sample
        text = self.texts[index]
        label = self.labels[index]
        
        # Preprocess text
        text = self.preprocessor.process(text)
        text = self.preprocessor.pad_sequence(sequence=text, max_length=self.max_length)
        
        # Convert label to numeric
        label = 1 if label == 'positive' else 0
        
        # Attention mask: 1 for real tokens, 0 for padding
        attention_mask = [1 if token != 0 else 0 for token in text]
        
        text = torch.tensor(data=text, dtype=torch.long)
        label = torch.tensor(data=label, dtype=torch.float32)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        return text, attention_mask, label