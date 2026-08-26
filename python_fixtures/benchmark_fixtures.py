import torch.nn as nn

INPUT_DIMS = 800


class HugeLinearModel(nn.Module):
    def __init__(self):
        super(HugeLinearModel, self).__init__()
        self.layer_sizes = [INPUT_DIMS, 1800, 1500, 2000, 2500, 1900]
        self.layers = nn.ModuleList([nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]) for i in range(len(self.layer_sizes) - 1)])

    def forward(self, x):
        # Iterate through the layers sequentially
        for layer in self.layers:
            x = layer(x)
        return x

    def get_input_dims(self) -> int:
        return INPUT_DIMS


class LongLinearModel(nn.Module):
    def __init__(self, layer_sizes: list[int]):
        super(LongLinearModel, self).__init__()

        # Generate 30 output sizes between 100 and 1000
        # We create a list of dimensions starting with the input_dim
        # followed by 30 random sizes.
        self.layer_sizes = layer_sizes
        dims = [INPUT_DIMS] + self.layer_sizes

        # Use nn.ModuleList to register the 30 layers
        # Each layer i connects dims[i] to dims[i+1]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    def forward(self, x):
        # Iterate through the layers sequentially
        for layer in self.layers:
            x = layer(x)
        return x

    def get_input_dims(self) -> int:
        return INPUT_DIMS
