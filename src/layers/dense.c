#include "layers/dense.h"

void forward_dense(DenseLayer *layer, float *input, float *output) {
   for (int i = 0; i < layer->output_size; ++i) {
      float temp_output = 0;
      for (int j = 0; j < layer->input_size; ++j) {
         temp_output += input[j] * layer->weights[i * layer->input_size + j];
      }
      output[i] = temp_output + layer->bias[i]; // Add bias and store in output array
   }
}