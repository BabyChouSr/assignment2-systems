import torch
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        # print(f"Output of first ffn: {x.dtype}")
        x = self.relu(x)
        x = self.ln(x)
        # print(f"Output of layer norm: {x.dtype}")
        x = self.fc2(x)
        return x