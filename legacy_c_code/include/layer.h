#ifndef LAYER_H
#define LAYER_H

/**
 * @brief Enum that represents different types of layers used in neural networks.
 */
typedef enum LayerTypes {
   Dense = 0,  /**< Represents a dense (fully connected) layer */
   Conv1D = 1, /**< Represents a convolutional 1D layer */
} LayerTypes;

#endif // LAYER_H