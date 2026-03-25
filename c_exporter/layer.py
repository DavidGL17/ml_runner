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
