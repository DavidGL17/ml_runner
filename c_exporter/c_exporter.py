from .utils import Layer
from .const import MODEL_TEMPLATE_INIT, MODEL_TEMPLATE_END


def export_layer(layer: Layer) -> str:
    return_string = f"""
#define LAYER_{layer.layer_num}_INPUT_SIZE {layer.input_size}
#define LAYER_{layer.layer_num}_OUTPUT_SIZE {layer.output_size}

static const float layer_{layer.layer_num}_weights[LAYER_{layer.layer_num}_INPUT_SIZE][LAYER_{layer.layer_num}_OUTPUT_SIZE] = {{
"""
    for i in range(len(layer.weights)):
        return_string += "{"
        for j in range(len(layer.weights[i])):
            return_string += f"{layer.weights[i][j]:.6f}f"
            if j < len(layer.weights[i]) - 1:
                return_string += ", "
        return_string += "}"
        if i < len(layer.weights) - 1:
            return_string += ",\n"
    return_string += "\n};\n\n"

    return_string += f"static const float layer_{layer.layer_num}_biases[LAYER_{layer.layer_num}_OUTPUT_SIZE] = {{\n"
    for i in range(len(layer.bias)):
        return_string += f"{layer.bias[i]:.6f}f"
        if i < len(layer.bias) - 1:
            return_string += ",\n"
    return_string += "\n};\n"

    return_string += f"""
static const {layer.layerType.layer_name} layer_{layer.layer_num} = {{.input_size = LAYER_{layer.layer_num}_INPUT_SIZE,
                                       .output_size = LAYER_{layer.layer_num}_OUTPUT_SIZE,
                                       .weights = (float *)layer_{layer.layer_num}_weights,
                                       .bias = (float *)layer_{layer.layer_num}_biases}};
"""

    return return_string


def export_model(model_layers: list[Layer], input_size: int, output_size: int) -> str:
    return_string = ""
    for layer in model_layers:
        return_string += export_layer(layer)
    return_string += f"""
#define MODEL_INPUT_SIZE {input_size}
#define MODEL_OUTPUT_SIZE {output_size}
"""
    return_string += "static Model model = {.layer_types = (LayerTypes[]){"
    for i, layer in enumerate(model_layers):
        return_string += f"{layer.layerType.enum_name}"
        if i < len(model_layers) - 1:
            return_string += ", "
    return_string += "},\n"
    return_string += f"                      .num_layers = {len(model_layers)},\n"
    return_string += "                      .layers = (void *[]){"
    for i, layer in enumerate(model_layers):
        return_string += f"(void *)&layer_{layer.layer_num}"
        if i < len(model_layers) - 1:
            return_string += ", "
    return_string += "},\n"
    return_string += """                      .output = NULL,
                      .input = NULL,
                      .input_size = MODEL_INPUT_SIZE,
                      .output_size = MODEL_OUTPUT_SIZE};
"""

    return_string = MODEL_TEMPLATE_INIT + return_string + MODEL_TEMPLATE_END

    return return_string
