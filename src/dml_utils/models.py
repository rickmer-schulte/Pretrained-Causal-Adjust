import torch
import torch.nn as nn

class CNN_2(nn.Module):
    """
    A CNN model with 2 convolutional layers followed by fully connected layers.
    The model is designed for regression tasks by default, but can be used for classification
    by setting the is_classifier flag to True.
    """
    def __init__(self, output_dim=1, is_classifier=False):
        super(CNN_2, self).__init__()
        self.is_classifier = is_classifier
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 56 * 56, 128), 
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        x = x.view(-1, 1) 
        if self.is_classifier:
            x = torch.sigmoid(x) 
        return x

class CNN_5(nn.Module):
    """
    A CNN model with 5 convolutional layers followed by fully connected layers.
    The model is designed for regression tasks by default, but can be used for classification
    by setting the is_classifier flag to True.
    """
    def __init__(self, output_dim=1, is_classifier=False):
        super(CNN_5, self).__init__()
        self.is_classifier = is_classifier
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)), 
            nn.Flatten()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        x = x.view(-1, 1)
        if self.is_classifier:
            x = torch.sigmoid(x)
        return x