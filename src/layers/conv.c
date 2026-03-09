#include "layers/conv.h"

ConvLayer *conv_layer_init(int input_channels, int output_channels, int kernel_size, int stride,
                           int padding, float *weights, float *bias) {
   ConvLayer *layer = (ConvLayer *)malloc(sizeof(ConvLayer));
   layer->input_channels = input_channels;
   layer->output_channels = output_channels;
   layer->kernel_size = kernel_size;
   layer->stride = stride;
   layer->padding = padding;

   // Allocate memory for weights and biases if needed
   if (weights == NULL) {
      layer->weights =
          malloc((input_channels * kernel_size * kernel_size * output_channels) * sizeof(float));
      // Initialize weights randomly or with zeros
      // Add your initialization code here
   } else {
      layer->weights = weights;
   }

   if (bias == NULL) {
      layer->bias = malloc(output_channels * sizeof(float));
      // Initialize biases
      // Add your initialization code here
   } else {
      layer->bias = bias;
   }

   return layer;
}

void forward_conv(ConvLayer *layer, float *input, float *output, int input_height,
                  int input_width) {
   // TODO: Implement convolutional layer forward pass
   // This will involve sliding the kernel over the input and performing
   // element-wise multiplications and sums to produce the output features
}