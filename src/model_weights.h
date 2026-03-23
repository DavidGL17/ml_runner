#ifndef MODEL_WEIGHTS_H
#define MODEL_WEIGHTS_H

#include "layers/dense.h"
#include "model.h"

#define LAYER_0_INPUT_SIZE 10
#define LAYER_0_OUTPUT_SIZE 10

static const float layer_0_weights[LAYER_0_INPUT_SIZE][LAYER_0_OUTPUT_SIZE] = {
{-0.230055f, -0.171888f, -0.124529f, 0.222677f, -0.149048f, -0.129030f, 0.311440f, -0.091044f, 0.155361f, -0.038560f},
{-0.119174f, 0.165989f, -0.255760f, 0.244097f, -0.102381f, 0.274477f, 0.001855f, 0.278313f, -0.165693f, 0.221434f},
{0.271509f, -0.060761f, 0.284897f, -0.262335f, -0.237607f, -0.314511f, 0.190442f, 0.002151f, 0.156293f, 0.034089f},
{0.144372f, -0.314916f, 0.127901f, -0.230625f, 0.090468f, 0.292282f, 0.270734f, 0.083269f, -0.067189f, -0.075442f},
{-0.304626f, 0.218615f, -0.197609f, -0.297480f, 0.308716f, 0.099331f, 0.053666f, 0.095540f, -0.119100f, -0.282084f},
{0.278978f, 0.194724f, 0.040073f, -0.307938f, 0.186814f, 0.274673f, 0.106697f, -0.140253f, -0.227062f, 0.217784f},
{-0.119727f, -0.091374f, 0.240300f, -0.000577f, 0.086229f, 0.116556f, 0.144762f, -0.147206f, 0.118433f, -0.279119f},
{-0.090602f, -0.175165f, 0.160833f, -0.303776f, 0.273112f, 0.125609f, -0.178089f, -0.192517f, 0.032454f, -0.142049f},
{0.257188f, 0.252255f, -0.156506f, -0.294707f, 0.285487f, 0.143364f, 0.134089f, -0.284126f, 0.085227f, 0.078621f},
{-0.309126f, 0.252002f, -0.059940f, -0.074358f, -0.213814f, -0.062420f, -0.144138f, -0.001252f, 0.004351f, 0.170473f}
};

static const float layer_0_biases[LAYER_0_OUTPUT_SIZE] = {
0.079469f,
0.315751f,
0.046654f,
0.243662f,
-0.016878f,
0.044009f,
-0.196135f,
-0.228204f,
-0.212724f,
0.185348f
};

static const DenseLayer layer_0 = {.input_size = LAYER_0_INPUT_SIZE,
                                       .output_size = LAYER_0_OUTPUT_SIZE,
                                       .weights = (float *)layer_0_weights,
                                       .bias = (float *)layer_0_biases};

#define LAYER_1_INPUT_SIZE 10
#define LAYER_1_OUTPUT_SIZE 5

static const float layer_1_weights[LAYER_1_INPUT_SIZE][LAYER_1_OUTPUT_SIZE] = {
{-0.043744f, 0.308052f, -0.127571f, -0.103912f, -0.116939f},
{0.111478f, 0.312682f, -0.044257f, 0.120255f, -0.177784f},
{-0.161954f, -0.017807f, 0.315028f, 0.120968f, 0.046539f},
{0.131096f, 0.091994f, 0.297350f, 0.206870f, -0.002975f},
{-0.290873f, 0.140051f, -0.260933f, -0.217914f, -0.198780f},
{-0.219400f, -0.162104f, -0.243344f, 0.015889f, -0.013549f},
{-0.154690f, -0.216459f, 0.201738f, -0.203423f, 0.301636f},
{0.065188f, 0.124731f, -0.253541f, 0.014256f, 0.273802f},
{0.156387f, -0.204459f, 0.162914f, 0.314302f, -0.308797f},
{-0.221447f, 0.286499f, 0.007882f, -0.179439f, 0.070895f}
};

static const float layer_1_biases[LAYER_1_OUTPUT_SIZE] = {
-0.219932f,
-0.058277f,
0.015077f,
0.091651f,
-0.187937f
};

static const DenseLayer layer_1 = {.input_size = LAYER_1_INPUT_SIZE,
                                       .output_size = LAYER_1_OUTPUT_SIZE,
                                       .weights = (float *)layer_1_weights,
                                       .bias = (float *)layer_1_biases};

#define MODEL_INPUT_SIZE 10
#define MODEL_OUTPUT_SIZE 5
static Model model = {.layer_types = (LayerTypes[]){Dense, Dense},
                      .num_layers = 2,
                      .layers = (void *[]){(void *)&layer_0, (void *)&layer_1},
                      .output = NULL,
                      .input = NULL,
                      .input_size = MODEL_INPUT_SIZE,
                      .output_size = MODEL_OUTPUT_SIZE};

#endif /* MODEL_WEIGHTS_H */
