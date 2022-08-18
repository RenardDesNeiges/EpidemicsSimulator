import torch
import torch.nn as nn

class DQN(nn.Module):
    """Fully connected classifier network for classfiying feature vectors."""
    
    def __init__(self, dropout=0):
        super().__init__()

        # Dropout modules
        dropout = nn.Dropout(dropout)

        #### Fully connected classification layers --------------------
        
        lin_layers = []
        lin1 = nn.Linear(in_features=39, out_features=64)
        relu1 = nn.ReLU()
        lin_layers += [self.lin1, self.relu1]
        lin2 = nn.Linear(in_features=64, out_features=32)
        relu2 = nn.ReLU()
        lin_layers += [self.lin2, self.relu2]
        lin3 = nn.Linear(in_features=32, out_features=16)
        relu3 = nn.ReLU()
        lin_layers += [self.lin3, self.relu3]
        lin4 = nn.Linear(in_features=16, out_features=7)
        lin_layers += [self.lin4]
        
        self.fcn = nn.Sequential(*lin_layers)
        
 
    def forward(self, x):
        x = self.fcn(x)
        x = self.dropout(x)
        return x

