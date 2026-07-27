#ifndef MODEL_H
#define MODEL_H

#include "layer.h"

/**
 * @brief Structure that represents a neural network model.
 *
 * This structure encapsulates all components necessary to define and work with a neural network
 * model. It includes information about the layers, their types, input and output dimensions, as
 * well as pointers to layer instances and the model's output.
 */
typedef struct Model {
   LayerTypes
       *layer_types; // Pointer to an array of layer type enums representing each layer in the model
   int num_layers;   // Total number of layers in the model
   void **layers;    // Array of pointers to layer instances
   float *output;    // Output data produced by the model after processing
   float *input;     // Input data fed into the model for processing
   int input_size;   // Number of features or elements in the input data
   int output_size;  // Number of features or elements in the output data
} Model;

void model_init(Model *model);

void model_forward(Model *model);

void model_free(Model *model);

#endif // MODEL_H