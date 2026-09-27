import torch
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


class LeNet300100(nn.Module):
    """Classic fully-connected baseline: three Linear layers over a
    flattened 28x28 input (784 -> 300 -> 100 -> 10). This is the dense-only
    reference architecture the original LeNet-5 paper benchmarks its
    convolutional design against."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_input_dims(self) -> int:
        return 28 * 28


class LeNet5(nn.Module):
    """Classic convolutional architecture: two Conv2d+MaxPool2d stages over
    a 32x32 single-channel input, feeding into three Linear layers down to
    10 classes."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),  # 32x32 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            nn.Conv2d(6, 16, kernel_size=5),  # 14x14 -> 10x10
            nn.ReLU(),
            nn.MaxPool2d(2),  # 10x10 -> 5x5
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def get_input_dims(self) -> tuple[int, int, int]:
        return (1, 32, 32)
