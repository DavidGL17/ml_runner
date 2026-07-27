#include "model.h"
#include "model_weights.h"
#include <stdio.h>

int main() {
   model_init(&model);
   for (int i = 0; i < model.input_size; ++i) {
      model.input[i] = (float)(i + 1);
   }

   model_forward(&model);

   // print input and output
   printf("Input: ");
   for (int i = 0; i < model.input_size; i++) {
      printf("%f ", model.input[i]);
   }
   printf("\nOutput: ");
   for (int i = 0; i < model.output_size; i++) {
      printf("%f ", model.output[i]);
   }
   printf("\n");

   model_free(&model);
}