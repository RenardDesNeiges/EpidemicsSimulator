import torch
import torch.nn as nn

class DQN(nn.Module):
    """Fully connected classifier network for classfiying feature vectors."""
    
    def __init__(self, in_dim = 126, out_dim = 16, dropout=0):
        super().__init__()

        # Dropout modules
        self.dropout = nn.Dropout(dropout)

        #### Fully connected classification layers --------------------
        
        lin_layers = []
        lin1 = nn.Linear(in_features=in_dim, out_features=64)
        relu1 = nn.ReLU()
        lin_layers += [lin1, relu1]
        lin2 = nn.Linear(in_features=64, out_features=32)
        relu2 = nn.ReLU()
        lin_layers += [lin2, relu2]
        lin3 = nn.Linear(in_features=32, out_features=16)
        relu3 = nn.ReLU()
        lin_layers += [lin3, relu3]
        lin4 = nn.Linear(in_features=16, out_features=out_dim)
        lin_layers += [lin4]
        
        self.fcn = nn.Sequential(*lin_layers)
        
 
    def forward(self, x):
        x = self.fcn(x.view(x.size(0),-1))

        x = self.dropout(x)
        return x

