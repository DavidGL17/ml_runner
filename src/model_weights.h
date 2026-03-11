#ifndef MODEL_WEIGHTS_H
#define MODEL_WEIGHTS_H

#include "layers/dense.h"
#include "model.h"

// Model dimensions
#define MODEL_INPUT_SIZE 4
#define MODEL_OUTPUT_SIZE 2

// Hardcoded weights matrix (INPUT_SIZE x OUTPUT_SIZE)
static const float model_weights[MODEL_INPUT_SIZE][MODEL_OUTPUT_SIZE] = {
    {0.5f, -0.3f}, // Input 0 connections to neurons 0 and 1
    {-0.2f, 0.9f}, // Input 1 connections to neurons 0 and 1
    {0.8f, -0.4f}, // Input 2 connections to neurons 0 and 1
    {-0.1f, 0.2f}  // Input 3 connections to neurons 0 and 1
};

// Hardcoded biases array (MODEL_OUTPUT_SIZE)
static const float model_biases[MODEL_OUTPUT_SIZE] = {
    0.7f, // Bias for neuron 0
    -1.5f // Bias for neuron 1
};

static const DenseLayer dense_layer = {.input_size = MODEL_INPUT_SIZE,
                                       .output_size = MODEL_OUTPUT_SIZE,
                                       .weights = (float *)model_weights,
                                       .bias = (float *)model_biases};

static Model model = {.layer_types = (LayerTypes[]){Dense},
                      .num_layers = 1,
                      .layers = (void *[]){(void *)&dense_layer},
                      .output = NULL,
                      .input = NULL,
                      .input_size = MODEL_INPUT_SIZE,
                      .output_size = MODEL_OUTPUT_SIZE};

#endif /* MODEL_WEIGHTS_H */