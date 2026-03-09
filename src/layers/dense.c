#include "layers/dense.h"

DenseLayer *dense_layer_init(int input_size, int output_size, float *weights, float *bias) {
   DenseLayer *layer = (DenseLayer *)malloc(sizeof(DenseLayer));
   layer->input_size = input_size;
   layer->output_size = output_size;

   // Allocate memory for weights and biases if needed
   if (weights == NULL) {
      layer->weights = malloc(input_size * output_size * sizeof(float));
      // Initialize weights randomly or with zeros
      // Add your initialization code here
   } else {
      layer->weights = weights;
   }

   if (bias == NULL) {
      layer->bias = malloc(output_size * sizeof(float));
      // Initialize biases
      // Add your initialization code here
   } else {
      layer->bias = bias;
   }

   return layer;
}

void forward_dense(DenseLayer *layer, float *input, float *output) {
   // TODO: Implement dense layer forward pass
   // This will involve matrix multiplication and addition of biases
}