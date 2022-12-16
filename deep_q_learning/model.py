"""Torch modules for deep-q-learning."""
import torch
import torch.nn as nn
from torch._tensor import Tensor
from typing import Tuple


class DQN(nn.Module):
    """Fully connected Q-Value estimator network."""

    def __init__(self, in_dim:int=126, out_dim:int=16, dropout:float=0, small:bool=False)-> nn.Module:
        """
        ![](https://miro.medium.com/max/1400/1*Gh5PS4R_A5drl5ebd_gNrg@2x.webp)
        
        Args:
            in_dim (int, optional): input dimensionality. Defaults to 126.
            out_dim (int, optional): output dimensionality. Defaults to 16.
            dropout (float, optional): dropout rate. Defaults to 0.
            small (bool, optional): if true, the network has 2 layers, else it has 4. Defaults to False.

        Returns:
            nn.Module: The Deep-Q-Network.
        """
        super().__init__()
        # Dropout modules
        self.dropout = nn.Dropout(dropout)

        # Fully connected classification layers --------------------

        lin_layers = []
        lin1 = nn.Linear(in_features=in_dim, out_features=64)
        relu1 = nn.ReLU()
        lin_layers += [lin1, relu1]
        lin2 = nn.Linear(in_features=64, out_features=32)
        relu2 = nn.ReLU()
        lin_layers += [lin2, relu2]

        if small:
            lin3 = nn.Linear(in_features=32, out_features=out_dim)
            lin_layers += [lin3]
        else:
            lin3 = nn.Linear(in_features=32, out_features=16)
            relu3 = nn.ReLU()
            lin_layers += [lin3, relu3]
            lin4 = nn.Linear(in_features=16, out_features=out_dim)
            lin_layers += [lin4]

        self.fcn = nn.Sequential(*lin_layers)

    def forward(self, x:Tensor) -> Tensor:
        """Forward pass through the deep-q-network module

        Args:
            x (Tensor): input

        Returns:
            Tensor: output
        """
        x = self.fcn(x.view(x.size(0), -1))
        return x



class FactoredDQN(nn.Module):
    """Fully connected Factored Q-Value estimator network."""

    def __init__(self, in_dim:int=126, out_dim:int=16, dropout:float=0, small:bool=False)-> nn.Module:
        """
        ![](https://miro.medium.com/max/1400/1*Gh5PS4R_A5drl5ebd_gNrg@2x.webp)
        
        Args:
            in_dim (int, optional): input dimensionality. Defaults to 126.
            out_dim (int, optional): output dimensionality. Defaults to 16.
            dropout (float, optional): dropout rate. Defaults to 0.
            small (bool, optional): if true, the network has 2 layers, else it has 4. Defaults to False.

        Returns:
            nn.Module: The Deep-Q-Network.
        """
        super().__init__()
        # DQN module
        self.out_dim = out_dim
        self.dqn = DQN(in_dim,out_dim,dropout,small)

    def forward(self, x:Tensor) -> Tuple[Tensor,Tensor]:
        """Forward pass through the deep-q-network module

        Args:
            x (Tensor): input

        Returns:
            Tuple[Tensor,Tensor]: (actions, q_values)
        """
        y = torch.reshape(self.dqn(x), (x.shape[0],2,self.out_dim//2))
        qmax = torch.max(y,axis=1)
        return qmax.indices, y, torch.sum(qmax.values)

