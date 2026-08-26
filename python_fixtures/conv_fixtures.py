import torch.nn as nn


# Two conv layers stacked directly, no activation or flatten in between -
# isolates conv-to-conv chaining (D3 -> D3 -> D3) on its own.
class SimpleConvOnlyModel(nn.Module):
    def __init__(self):
        super(SimpleConvOnlyModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# Conv -> ReLU (D3 activation) -> Flatten, with no dense layer after -
# isolates the D3 -> Flat transition on its own.
class ConvFlattenModel(nn.Module):
    def __init__(self):
        super(ConvFlattenModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.act_relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.act_relu(x)
        x = self.flatten(x)
        return x


# The full pipeline: Conv2d -> ReLU -> Conv2d -> ReLU -> Flatten -> Linear ->
# Sigmoid -> Linear -> Tanh -> Linear -> Softmax. Exercises every layer type
# and activation together in one model.
class FullConvModel(nn.Module):
    def __init__(self, output_size):
        super(FullConvModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.act1_relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.act2_relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(8 * 4 * 4, 20)
        self.act3_sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(20, 15)
        self.act4_tanh = nn.Tanh()
        self.linear3 = nn.Linear(15, output_size)
        self.act5_softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1_relu(x)
        x = self.conv2(x)
        x = self.act2_relu(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.act3_sigmoid(x)
        x = self.linear2(x)
        x = self.act4_tanh(x)
        x = self.linear3(x)
        x = self.act5_softmax(x)
        return x
