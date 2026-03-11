#include "layers/dense.h"

void forward_dense(DenseLayer *layer, float *input, float *output) {
   for (int i = 0; i < layer->output_size; ++i) {
      output[i] = layer->bias[i];
      for (int j = 0; j < layer->input_size; ++j) {
         output[i] += input[j] * layer->weights[j * layer->output_size + i];
      }
   }
}