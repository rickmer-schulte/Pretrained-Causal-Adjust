import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin

class CNNRegressor(BaseEstimator, RegressorMixin):
    """
    Wrapper for a PyTorch CNN model to make it compatible with scikit-learn's fit/predict interface.
    """
    def __init__(self, model, epochs=10, batch_size=32, learning_rate=0.001, is_classifier=False):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.is_classifier = is_classifier
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
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(self.epochs):
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        
        return self

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor).cpu().numpy()
        # Flatten to (batch_size,)
        outputs = outputs.reshape(-1) 
        
        # Convert logits to binary predictions (0/1)
        if self.is_classifier:
            return (outputs > 0.5).astype(int)  
    
        return outputs