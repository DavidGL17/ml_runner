#ifndef LAYER_H
#define LAYER_H

/**
 * @brief Enum that represents different types of layers used in neural networks.
 */
typedef enum LayerTypes {
   Conv = 0, /**< Represents a convolutional layer */
   Dense = 1 /**< Represents a dense (fully connected) layer */
} LayerTypes;

#endif // LAYER_H