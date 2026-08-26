import torch.nn as nn
from random import randint


def get_output_size() -> int:
    return randint(2, 10) * 5


class SimpleLinearModel(nn.Module):
    def __init__(self, input_size, output_size, layer_num):
        super(SimpleLinearModel, self).__init__()
        self.layers = nn.ModuleList()
        if layer_num == 1:
            self.layers.append(nn.Linear(input_size, output_size))
        else:
            inter_output_size = get_output_size()
            self.layers.append(nn.Linear(input_size, inter_output_size))
            inter_input_size = inter_output_size
            for i in range(layer_num - 2):
                inter_output_size = get_output_size()
                self.layers.append(nn.Linear(inter_input_size, inter_output_size))
                inter_input_size = inter_output_size
            self.layers.append(nn.Linear(inter_input_size, output_size))

    def forward(self, x):
        # Pass input through the linear layer
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
