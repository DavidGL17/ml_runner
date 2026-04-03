#include "model.h"
#include "layer.h"
#include "layers/conv.h"
#include "layers/dense.h"
#include <stdlib.h>
#include <string.h>

void model_init(Model *model) {
   (*model).input = malloc(sizeof(float) * (*model).input_size);
   (*model).output = malloc(sizeof(float) * (*model).output_size);
}

void model_forward(Model *model) {
   // Forward pass through all layers
   float *input = (*model).input;
   float *output = (*model).output;

   float *temporary_input = malloc(sizeof(float) * (*model).input_size);
   memcpy(temporary_input, input, sizeof(float) * (*model).input_size);
   float *temporary_output = NULL;
   for (int i = 0; i < (*model).num_layers; i++) {
      switch ((LayerTypes)(*model).layer_types[i]) {
      case Dense:
         temporary_output = malloc(sizeof(float) * (*(DenseLayer *)(*model).layers[i]).output_size);
         forward_dense((DenseLayer *)(*model).layers[i], temporary_input, temporary_output);
         break;
      case Conv1D:
         temporary_output = NULL;
         // forward_conv(&(*model).layers[i].layer.conv);
         break;
      default:
         break;
      }
      free(temporary_input);
      temporary_input = temporary_output;
   }
   memcpy(output, temporary_output, sizeof(float) * (*model).output_size);
}

void model_free(Model *model) {
   free((*model).input);
   free((*model).output);
}