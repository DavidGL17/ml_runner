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
    """Classic fully-connected baseline (LeCun et al. 1998): three Linear
    layers over a flattened 28x28 input (784 -> 300 -> 100 -> 10). The
    dense-only reference architecture the original LeNet-5 paper benchmarks
    its convolutional design against."""

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
    """Classic convolutional architecture (LeCun et al. 1998): two
    Conv2d+MaxPool2d stages over a 32x32 single-channel input, feeding into
    three Linear layers down to 10 classes."""

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


class _BasicBlock(nn.Module):
    """One ResNet BasicBlock: two 3x3 conv+BN, a residual add, and a 1x1
    conv+BN shortcut whenever the stride or channel count changes."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.shortcut is None else self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class ResNet18Cifar(nn.Module):
    """ResNet-18 (He et al. 2015): the same 8-BasicBlock structure as the
    original (2 blocks per stage x 4 stages, channels 64/128/256/512), with
    a CIFAR-style plain 3x3 conv stem instead of the ImageNet variant's
    initial 7x7 conv + maxpool, so it stays fast to benchmark over a 32x32
    input. The final global-average-pool is replaced with a kernel-matched
    Conv2d (4x4 feature map -> 1x1) so the whole model only ever uses
    Conv2d, BatchNorm2d, ReLU, elementwise add and Linear - the ops
    confirmed as supported - rather than AdaptiveAvgPool2d, which wasn't."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        # 32x32 -> (three stride-2 stages) -> 4x4; collapse to 1x1 with a
        # kernel-matched conv instead of a pooling op.
        self.pool_conv = nn.Conv2d(512, 512, kernel_size=4, bias=False)
        self.bn_pool = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(512, 10)

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [_BasicBlock(in_channels, out_channels, stride)]
        for _ in range(blocks - 1):
            layers.append(_BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.relu(self.bn_pool(self.pool_conv(x)))
        x = torch.flatten(x, 1)
        return self.fc(x)

    def get_input_dims(self) -> tuple[int, int, int]:
        return (3, 32, 32)


class LSTMClassifier(nn.Module):
    """A 2-layer LSTM (Hochreiter & Schmidhuber, 1997) over a short
    synthetic sequence, followed by a Linear head on the final hidden
    state - the classic recurrent-classifier shape, using only nn.LSTM and
    nn.Linear (no embedding lookup - it consumes continuous features
    directly)."""

    SEQ_LEN = 20
    INPUT_SIZE = 64
    HIDDEN_SIZE = 128

    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=self.INPUT_SIZE, hidden_size=self.HIDDEN_SIZE, num_layers=2)
        self.fc = nn.Linear(self.HIDDEN_SIZE, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch, input_size) - nn.LSTM's default layout, which
        # is exactly what benchmark.py's resolve_batched_shape() builds for
        # a 2-tuple get_input_dims().
        output, _ = self.lstm(x)
        return self.fc(output[-1])

    def get_input_dims(self) -> tuple[int, int]:
        return (self.SEQ_LEN, self.INPUT_SIZE)


class AlexNetLite(nn.Module):
    """AlexNet (Krizhevsky et al. 2012)-style: 5 conv layers (MaxPool2d
    after the 1st, 2nd and 5th) feeding into 3 Linear layers. Scaled down
    to a 64x64 input (vs. the original 224x224) and narrower channels so
    1000 forward passes stay quick, especially on something like a
    Raspberry Pi. LocalResponseNorm - used in the original, rarely
    implemented in minimal inference engines - is dropped, the same way
    most modern reimplementations of AlexNet treat it."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=2),  # 64x64 -> 31x31
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # -> 15x15
            nn.Conv2d(32, 96, kernel_size=5, padding=2),  # -> 15x15
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # -> 7x7
            nn.Conv2d(96, 128, kernel_size=3, padding=1),  # -> 7x7
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # -> 7x7
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 96, kernel_size=3, padding=1),  # -> 7x7
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # -> 3x3
        )
        self.classifier = nn.Sequential(
            nn.Linear(96 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def get_input_dims(self) -> tuple[int, int, int]:
        return (3, 64, 64)
