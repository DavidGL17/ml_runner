#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include <stdlib.h>

typedef struct Conv1DLayer {
   int input_channels;
   int output_channels;
   int kernel_size;
   int stride;
   int padding;
   float *weights; // Pointer to convolutional kernels weights
   float *bias;    // Pointer to bias vector data
} Conv1DLayer;

// Forward pass function for convolutional layer
void forward_conv_1d(Conv1DLayer *layer, float *input, float *output);
#endif // CONVOLUTIONAL_LAYER_H