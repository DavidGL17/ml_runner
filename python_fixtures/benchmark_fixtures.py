import torch.nn as nn
import random

INPUT_DIMS = 800


class HugeLinearModel(nn.Module):
    def __init__(self):
        super(HugeLinearModel, self).__init__()
        self.linear1 = nn.Linear(INPUT_DIMS, 1800)
        self.linear2 = nn.Linear(1800, 1500)
        self.linear3 = nn.Linear(1500, 2000)

    def forward(self, x):
        output = self.linear1(x)
        output = self.linear2(output)
        output = self.linear3(output)
        return output

    def get_input_dims(self) -> int:
        return INPUT_DIMS


class LongLinearModel(nn.Module):
    def __init__(self):
        super(LongLinearModel, self).__init__()

        # Generate 30 output sizes between 100 and 1000
        # We create a list of dimensions starting with the input_dim
        # followed by 30 random sizes.
        self.layer_sizes = [random.randint(100, 1000) for _ in range(30)]
        dims = [INPUT_DIMS] + self.layer_sizes

        # Use nn.ModuleList to register the 30 layers
        # Each layer i connects dims[i] to dims[i+1]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(30)])

    def forward(self, x):
        # Iterate through the layers sequentially
        for layer in self.layers:
            x = layer(x)
        return x

    def get_input_dims(self) -> int:
        return INPUT_DIMS
