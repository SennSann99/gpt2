import torch
import torch.nn.functional as F
import numpy as np

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.config = config

    def _calculate_loss(self, input_ids, targets):
        input_ids, targets = input_ids.to(self.device), targets.to(self.device)
        logits = self.model(input_ids)
        # Flatten batches and sequence lengths for CrossEntropy
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten())

    def evaluate(self):
        self.model.eval()
        val_losses = []
        with torch.no_grad():
            for i, (input_ids, targets) in enumerate(self.val_loader):
                if i >= self.config.val_iters:
                    break
                loss = self._calculate_loss(input_ids, targets)
                val_losses.append(loss.item())
                
        self.model.train()
        return np.mean(val_losses) if val_losses else 0.0

    def train(self):
        train_history, val_history = [], []
        self.model.train()

        for epoch in range(self.config.num_epochs):
            losses = []
            for input_ids, targets in self.train_loader:
                self.optimizer.zero_grad()
                loss = self._calculate_loss(input_ids, targets)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            
            avg_train_loss = np.mean(losses)
            avg_val_loss = self.evaluate()
            
            train_history.append(avg_train_loss)
            val_history.append(avg_val_loss)
            
            print(f"Epoch {epoch+1}/{self.config.num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        return train_history, val_history