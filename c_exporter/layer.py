from abc import ABC, abstractmethod


class LayerParser(ABC):
    def __init__(self, c_layer_name: str, enum_name: str, layer_num: int):
        self.c_layer_name = c_layer_name
        self.enum_name = enum_name
        self.layer_num = layer_num

    @abstractmethod
    def parse_layer(self) -> str:
        pass


class LinearLayerParser(LayerParser):

    def __init__(self, layer_num: int, input_size: int, output_size: int, weights: list, bias: list):
        super().__init__("DenseLayer", "Dense", layer_num)
        self.input_size = input_size
        self.output_size = output_size
        self.weights = weights
        self.bias = bias

    def parse_layer(self) -> str:
        return_string = f"""
#define LAYER_{self.layer_num}_INPUT_SIZE {self.input_size}
#define LAYER_{self.layer_num}_OUTPUT_SIZE {self.output_size}

static const float layer_{self.layer_num}_weights[LAYER_{self.layer_num}_INPUT_SIZE][LAYER_{self.layer_num}_OUTPUT_SIZE] = {{
"""
        for i in range(len(self.weights)):
            return_string += "{"
            for j in range(len(self.weights[i])):
                return_string += f"{self.weights[i][j]:.6f}f"
                if j < len(self.weights[i]) - 1:
                    return_string += ", "
            return_string += "}"
            if i < len(self.weights) - 1:
                return_string += ",\n"
        return_string += "\n};\n\n"

        return_string += f"static const float layer_{self.layer_num}_biases[LAYER_{self.layer_num}_OUTPUT_SIZE] = {{\n"
        for i in range(len(self.bias)):
            return_string += f"{self.bias[i]:.6f}f"
            if i < len(self.bias) - 1:
                return_string += ",\n"
        return_string += "\n};\n"

        return_string += f"""
static const {self.c_layer_name} layer_{self.layer_num} = {{.input_size = LAYER_{self.layer_num}_INPUT_SIZE,
                                    .output_size = LAYER_{self.layer_num}_OUTPUT_SIZE,
                                    .weights = (float *)layer_{self.layer_num}_weights,
                                    .bias = (float *)layer_{self.layer_num}_biases}};
    """

        return return_string


class Conv1DLayerParser(LayerParser):
    """
    Generates C definitions for a 1‑D convolutional layer.
    The ONNX weight tensor has shape (out_channels, in_channels, kernel_size).
    """

    def __init__(
        self,
        layer_num: int,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        weights: list,
        bias: list,
    ):
        super().__init__("Conv1DLayer", "Conv", layer_num)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = weights  # 1‑D flattened list
        self.bias = bias

    def parse_layer(self) -> str:
        # C pre‑processor macros for layer meta‑data
        macros = f"""
#define LAYER_{self.layer_num}_INPUT_CHANNELS  {self.input_channels}
#define LAYER_{self.layer_num}_OUTPUT_CHANNELS {self.output_channels}
#define LAYER_{self.layer_num}_KERNEL_SIZE     {self.kernel_size}
#define LAYER_{self.layer_num}_STRIDE          {self.stride}
#define LAYER_{self.layer_num}_PADDING        {self.padding}\n
"""

        # Flatten the weight matrix into a 1‑D array
        weight_array = f"static const float layer_{self.layer_num}_weights["
        weight_array += f"{self.output_channels * self.input_channels * self.kernel_size}] = {{\n"
        for i, val in enumerate(self.weights):
            weight_array += f"{val:.6f}f"
            if i < len(self.weights) - 1:
                weight_array += ", "
        weight_array += "\n};\n\n"

        # Bias vector
        bias_array = f"static const float layer_{self.layer_num}_biases[{self.output_channels}] = {{\n"
        for i, val in enumerate(self.bias):
            bias_array += f"{val:.6f}f"
            if i < len(self.bias) - 1:
                bias_array += ", "
        bias_array += "\n};\n\n"

        # ConvLayer instance
        layer_decl = f"""static const ConvLayer layer_{self.layer_num} = {{
    .input_channels  = LAYER_{self.layer_num}_INPUT_CHANNELS,
    .output_channels = LAYER_{self.layer_num}_OUTPUT_CHANNELS,
    .kernel_size     = LAYER_{self.layer_num}_KERNEL_SIZE,
    .stride          = LAYER_{self.layer_num}_STRIDE,
    .padding         = LAYER_{self.layer_num}_PADDING,
    .weights         = (float *)layer_{self.layer_num}_weights,
    .bias            = (float *)layer_{self.layer_num}_biases
}};\n"""

        return macros + weight_array + bias_array + layer_decl
