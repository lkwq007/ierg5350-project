import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, hidden_layer=1, init_weight=True):
        super(DQN, self).__init__()
        lst = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for i in range(hidden_layer):
            lst.append(nn.Linear(hidden_dim, hidden_dim))
            lst.append(nn.ReLU())
        lst.append(nn.Linear(hidden_dim, 1))
        self.backbone = nn.Sequential(*lst)

        if init_weight:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.backbone(x)
