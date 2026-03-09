#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include <stdlib.h>

typedef struct DenseLayer {
   int input_size;
   int output_size;
   float *weights; // Pointer to weight matrix data
   float *bias;    // Pointer to bias vector data
} DenseLayer;

// Initialize a dense layer with given weights and biases
DenseLayer *dense_layer_init(int input_size, int output_size, float *weights, float *bias);

// Forward pass function for dense layer
void forward_dense(DenseLayer *layer, float *input, float *output);

#endif // DENSE_LAYER_H