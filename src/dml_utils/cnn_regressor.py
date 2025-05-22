import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin

class CNNRegressor(BaseEstimator, RegressorMixin):
    """
    Wrapper for a PyTorch CNN model to make it compatible with scikit-learn's fit/predict interface.
    """
    def __init__(
        self,
        model,
        epochs=10,
        batch_size=32,
        learning_rate=0.001,
        is_classifier=False,
        patience=3,
        val_fraction=0.2,
        verbose=False,        
    ):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.is_classifier = is_classifier
        self.patience = patience              
        self.val_fraction = val_fraction      
        self.verbose = verbose                 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def fit(self, X, y):
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss() if self.is_classifier else nn.MSELoss()

        # Convert data to tensors
        X_tensor = torch.from_numpy(X).float().to(self.device)
        y_tensor = torch.from_numpy(y).float().view(-1, 1).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)

        # Split into training and validation sets
        total = len(dataset)
        val_size = int(total * self.val_fraction)
        train_size = total - val_size
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False
        )

        best_loss = float('inf')
        patience_counter = 0
        best_state = None

        # Training loop 
        for epoch in range(1, self.epochs + 1):
            # Training 
            epoch_train_loss = 0.0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
            epoch_train_loss /= len(train_loader)

            # Validation
            self.model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    epoch_val_loss += loss.item()
            epoch_val_loss /= len(val_loader)

            # Switch back to training mode
            self.model.train()

            if self.verbose:
                print(
                    f"Epoch {epoch}/{self.epochs} - "
                    f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}"
                )

            # Early stopping check
            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                best_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if self.verbose:
                    print(f"No improvement in epoch {epoch}. Patience: {patience_counter}/{self.patience}")
                if patience_counter >= self.patience:
                    if self.verbose:
                        print("Early stopping triggered.")
                    break

        # Load best model state
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor).cpu().numpy().reshape(-1)

        # Convert logits to binary predictions (0/1)
        if self.is_classifier:
            return (outputs > 0.5).astype(int)  
    
        return outputs