#ifndef MODEL_WEIGHTS_H
#define MODEL_WEIGHTS_H

#include "layers/conv.h"
#include "layers/dense.h"
#include "model.h"

#define LAYER_0_INPUT_CHANNELS 1
#define LAYER_0_OUTPUT_CHANNELS 8
#define LAYER_0_KERNEL_SIZE 3
#define LAYER_0_STRIDE 1
#define LAYER_0_PADDING 0

static const float layer_0_weights[24] = {
    0.312895f,  0.121777f,  -0.273029f, -0.572958f, -0.422403f, 0.199023f,  0.560386f, -0.293596f,
    -0.392227f, -0.544646f, -0.310812f, -0.382111f, 0.522043f,  -0.145729f, 0.389817f, -0.556246f,
    -0.137970f, -0.176023f, 0.474092f,  -0.176053f, 0.192697f,  0.517026f,  0.066675f, 0.118278f};

static const float layer_0_biases[8] = {-0.346965f, -0.424987f, 0.129061f, 0.284051f,
                                        -0.305683f, 0.367364f,  0.487681f, -0.009804f};

static const Conv1DLayer layer_0 = {.input_channels = LAYER_0_INPUT_CHANNELS,
                                    .output_channels = LAYER_0_OUTPUT_CHANNELS,
                                    .kernel_size = LAYER_0_KERNEL_SIZE,
                                    .stride = LAYER_0_STRIDE,
                                    .padding = LAYER_0_PADDING,
                                    .output_size = 8,
                                    .weights = (float *)layer_0_weights,
                                    .bias = (float *)layer_0_biases};

#define MODEL_INPUT_SIZE 1
#define MODEL_OUTPUT_SIZE 8
static Model model = {.layer_types = (LayerTypes[]){Conv1D},
                      .num_layers = 1,
                      .layers = (void *[]){(void *)&layer_0},
                      .output = NULL,
                      .input = NULL,
                      .input_size = MODEL_INPUT_SIZE,
                      .output_size = MODEL_OUTPUT_SIZE};

#endif /* MODEL_WEIGHTS_H */
