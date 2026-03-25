from .layer import LayerParser
from .const import MODEL_TEMPLATE_INIT, MODEL_TEMPLATE_END


def export_model(model_layers: list[LayerParser], input_size: int, output_size: int) -> str:
    return_string = ""
    for layer in model_layers:
        return_string += layer.parse_layer()
    return_string += f"""
#define MODEL_INPUT_SIZE {input_size}
#define MODEL_OUTPUT_SIZE {output_size}
"""
    return_string += "static Model model = {.layer_types = (LayerTypes[]){"
    for i, layer in enumerate(model_layers):
        return_string += f"{layer.enum_name}"
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
