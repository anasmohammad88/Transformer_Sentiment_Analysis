import torch
import torchmetrics as met
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, train_loader, valid_loader, optimizer, criterion, scheduler):
        self.device =  torch.device("cuda" if torch.cuda.is_available()  else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.writer = SummaryWriter(log_dir="logs")
        self.train_accuracy = met.Accuracy(task="binary").to(self.device)
        self.valid_accuracy = met.Accuracy( task="binary").to(self.device)
        
    def train(self, epochs):
        """
        Full training loop.
        """
        best_valid_accuracy = 0.0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            self.train_accuracy.reset()
            
            for text, attention_mask, label in self.train_loader:
                text = text.to(self.device)
                label = label.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                # Reset gradients
                self.optimizer.zero_grad()

                # Forward pass
                logits = self.model(text, attention_mask)

                # Compute loss
                loss = self.criterion(logits,label)

                # Backpropagation
                loss.backward()

                # Update weights
                self.optimizer.step()

                # Accumulate loss
                train_loss += (loss.item() * text.size(0))

                # Update accuracy
                self.train_accuracy.update(torch.sigmoid(logits),label.int())

            # Average train loss
            train_loss /= len(self.train_loader.dataset)
            train_acc = (self.train_accuracy.compute().item())
            
            
            self.model.eval()
            valid_loss = 0.0
            self.valid_accuracy.reset()

            with torch.no_grad():

                for text, attention_mask, label in self.valid_loader:
                    text = text.to(self.device)
                    label = label.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    
                    # Forward pass
                    logits = self.model(text, attention_mask)

                    # Compute loss
                    loss = self.criterion(logits,label)

                    # Accumulate loss
                    valid_loss += (loss.item() * text.size(0))

                    # Update accuracy
                    self.valid_accuracy.update(torch.sigmoid(logits), label.int())

            # Average validation loss
            valid_loss /= len(self.valid_loader.dataset)
            valid_acc = (self.valid_accuracy.compute().item())
            
            self.writer.add_scalars("Loss", {"Train": train_loss,"Eval": valid_loss}, epoch)
            self.writer.add_scalars("Accuracy", {"Train": train_acc,"Eval": valid_acc}, epoch)
            
            self.scheduler.step(valid_acc)
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            print(
                f"Epoch: {epoch + 1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Valid Loss: {valid_loss:.4f} | "
                f"Train Accuracy: {train_acc:.4f} | "
                f"Valid Accuracy: {valid_acc:.4f} | " 
                f"LR: {current_lr:.6f}" 
            )
            
            # Save best model
            if valid_acc > best_valid_accuracy:
                best_valid_accuracy = valid_acc
                torch.save(self.model.state_dict(), "transformer_sentiment_analysis.pth")

        print(f"\nBest Validation Accuracy: {best_valid_accuracy:.4f}")