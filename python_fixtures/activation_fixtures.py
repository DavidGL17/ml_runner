import torch.nn as nn


# Dense layers with every activation type chained in between (relu, sigmoid,
# tanh, softmax). Linear/identity activation is intentionally excluded since
# it has no corresponding ONNX node to export from.
class ActivationModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ActivationModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 8)
        self.act1_relu = nn.ReLU()
        self.linear2 = nn.Linear(8, 12)
        self.act2_sigmoid = nn.Sigmoid()
        self.linear3 = nn.Linear(12, 10)
        self.act3_tanh = nn.Tanh()
        self.linear4 = nn.Linear(10, output_size)
        self.act4_softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act1_relu(x)
        x = self.linear2(x)
        x = self.act2_sigmoid(x)
        x = self.linear3(x)
        x = self.act3_tanh(x)
        x = self.linear4(x)
        x = self.act4_softmax(x)
        return x
