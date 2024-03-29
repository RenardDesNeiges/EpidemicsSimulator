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

class DQ_CNN(nn.Module):
    """CNN classifier network."""

    def __init__(self, in_dim=126, out_dim=16,  classifier_dropout=0.4, conv_dropout=0.25):
        """
        ![](https://miro.medium.com/max/1184/1*V6Y8FF2qfw_ztNbs1AHXNg.webp)
        
        Args:
            classifier_dropout (float, optional): Dropout rate for the classifier. Defaults to 0.4.
            conv_dropout (float, optional): Dropout rate for the convolutional module. Defaults to 0.25.
        """
        super().__init__()

        conv_layers = []
        self.dropout = nn.Dropout(classifier_dropout)
        self.dropout2d = nn.Dropout2d(conv_dropout)

        #  Convolutional layers --------------------

        conv1 = nn.Conv2d(2, 16, kernel_size=(
            4, 4), stride=(1, 1), padding=(2, 2))
        relu1 = nn.ReLU()
        bn1 = nn.BatchNorm2d(16)
        nn.init.kaiming_normal_(conv1.weight, a=0.1)
        conv1.bias.data.zero_()
        mp1 = nn.MaxPool2d((2, 2), stride=(2, 2))
        conv_layers += [conv1, relu1, bn1, mp1]

        conv2 = nn.Conv2d(16, 32, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        relu2 = nn.ReLU()
        bn2 = nn.BatchNorm2d(32)
        nn.init.kaiming_normal_(conv2.weight, a=0.1)
        conv2.bias.data.zero_()
        conv_layers += [conv2, relu2, bn2]

        conv3 = nn.Conv2d(32, 64, kernel_size=(
            2, 2), stride=(1, 1), padding=(1, 1))
        relu3 = nn.ReLU()
        bn3 = nn.BatchNorm2d(64)
        nn.init.kaiming_normal_(conv3.weight, a=0.1)
        conv3.bias.data.zero_()
        conv_layers += [conv3, relu3, bn3]

        conv4 = nn.Conv2d(64, 64, kernel_size=(
            3, 3), stride=(2, 2), padding=(1, 1))
        relu4 = nn.ReLU()
        bn4 = nn.BatchNorm2d(64)
        nn.init.kaiming_normal_(conv4.weight, a=0.1)
        conv4.bias.data.zero_()
        conv_layers += [conv4, relu4, bn4]

        # Fully connected classification layers --------------------

        self.ap = nn.AdaptiveAvgPool2d(output_size=10)
        lin1 = nn.Linear(in_features=6400, out_features=40)
        relu4 = nn.ReLU()
        lin2 = nn.Linear(in_features=40, out_features=out_dim)
        lin_layers = [lin1, relu4, lin2]

        # Wrap the convolutional and classification layers
        self.conv = nn.Sequential(*conv_layers)
        self.classifier = nn.Sequential(*lin_layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout2d(x)

        # Adaptive pool and flatten for input classification layers
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.classifier(x)
        x = self.dropout(x)

        return x
