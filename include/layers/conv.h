#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include <stdlib.h>

typedef struct ConvLayer {
   int input_channels;
   int output_channels;
   int kernel_size;
   int stride;
   int padding;
   float *weights; // Pointer to convolutional kernels weights
   float *bias;    // Pointer to bias vector data
} ConvLayer;

// Initialize a convolutional layer with given parameters
ConvLayer *conv_layer_init(int input_channels, int output_channels, int kernel_size, int stride,
                           int padding, float *weights, float *bias);

// Forward pass function for convolutional layer
void forward_conv(ConvLayer *layer, float *input, float *output, int input_height, int input_width);
#endif // CONVOLUTIONAL_LAYER_H