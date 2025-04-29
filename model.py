import torch
import torch.nn as nn


class FlexiblePINN(nn.Module):
    def __init__(self, num_buses):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2 * num_buses, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 2 * num_buses),
        )
        self.num_buses = num_buses

    def forward(self, S):
        x = torch.cat([S.real, S.imag], dim=1)
        out = self.model(x)
        V_mag = out[:, : self.num_buses]
        V_ang = out[:, self.num_buses :]
        return V_mag, V_ang
