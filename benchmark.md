# Benchmark results

Results obtained by running `poetry run python benchmark.py`. It runs a benchmark based on what is configured in [the config file](benchmark_config.json). 

It will run the following models. All of them are small classifiers with 10 output classes, picked to exercise different operator mixes (dense, convolution, residual/batch-norm, recurrent) at very different compute budgets.

|                        Model                         |     Family      |         Input          | Parameters | Operators exercised                                             |
| :--------------------------------------------------: | :-------------: | :--------------------: | :--------: | :-------------------------------------------------------------- |
|    [LeNet300100](#results-for-model--lenet300100)    | Fully connected | 784 (flattened 28x28)  |   ~267 K   | Linear, ReLU                                                    |
|         [LeNet5](#results-for-model--lenet5)         |    Small CNN    |        1x32x32         |   ~62 K    | Conv2d (5x5), MaxPool2d, Linear, ReLU                           |
|  [ResNet18Cifar](#results-for-model--resnet18cifar)  |  Residual CNN   |        3x32x32         |  ~15.4 M   | Conv2d (3x3, 1x1, 4x4), BatchNorm2d, residual add, ReLU, Linear |
| [LSTMClassifier](#results-for-model--lstmclassifier) |    Recurrent    | 20 steps x 64 features |   ~233 K   | 2-layer LSTM, Linear, multiple onnx layers                      |
|    [AlexNetLite](#results-for-model--alexnetlite)    |  Mid-size CNN   |        3x64x64         |  ~1.03 M   | Conv2d (7x7, 5x5, 3x3), MaxPool2d, Linear, ReLU                 |

For more details on the models, see [the model chapter](#model-descriptions).

## Results

Detailed results per version can be found in the [benchmark_results](benchmark_results/) folder. Results for latest unpublished version are still work in progress.

### Overall results

![benchmark results](img/benchmark_result_times.png)

### Improvements on latest version vs base python

![rust speedup](img/benchmark_rust_speedup.png)


---

## Model descriptions

### LeNet300100
The classic fully connected baseline from LeCun et al. (1998): three Linear layers over a flattened 28x28 image (784 -> 300 -> 100 -> 10) with ReLU in between. It is the dense-only reference the original LeNet-5 paper compared its convolutional design against. It is the cheapest model here and mostly measures raw matrix-vector/matrix-multiply speed and per-call overhead.

### LeNet5
The convolutional counterpart from the same paper: two Conv2d (5x5) + ReLU + MaxPool2d (2x2) stages over a 32x32 single-channel input (32x32 -> 28x28 -> 14x14 -> 10x10 -> 5x5, with 6 then 16 channels), followed by three Linear layers (400 -> 120 -> 84 -> 10). It is a small, convolution-dominated workload with tiny channel counts.

### ResNet18Cifar
ResNet-18 (He et al., 2015) keeping the original structure of 8 BasicBlocks (2 per stage, 4 stages, channels 64/128/256/512). Each block has two 3x3 conv + BatchNorm layers and a residual add, with a 1x1 conv + BatchNorm shortcut when the stride or channel count changes. Two adaptations are made for benchmarking:
- The ImageNet 7x7 stem + max-pool is replaced by a plain 3x3 conv stem, so it stays practical on 32x32 CIFAR-style inputs.
- The final global average pool is replaced by a kernel-matched 4x4 Conv2d (4x4 feature map -> 1x1), so the whole model only uses Conv2d, BatchNorm2d, ReLU, elementwise add and Linear, which are the operators the Rust runner supports.

It is by far the heaviest model here and stresses convolution throughput, memory traffic and batch-norm handling.

### LSTMClassifier
A 2-layer LSTM (Hochreiter & Schmidhuber, 1997) with hidden size 128, run over a synthetic sequence of 20 steps with 64 continuous features per step (no embedding lookup), followed by a Linear head (128 -> 10) on the final time step's output. The input layout is `(seq_len, batch, input_size)`, the default of `nn.LSTM`. It tests recurrent execution: many small, sequentially dependent matrix multiplications plus gate activations.

### AlexNetLite
An AlexNet-style (Krizhevsky et al., 2012) network: 5 Conv2d + ReLU layers (7x7 stride 2, 5x5, then three 3x3), with MaxPool2d (3x3, stride 2) after the 1st, 2nd and 5th, feeding 3 Linear layers (864 -> 512 -> 256 -> 10). It is scaled down to a 64x64 input (instead of 224x224) with narrower channels so that repeated forward passes stay quick, especially on a Raspberry Pi. Local Response Normalization from the original is dropped, as in most modern reimplementations, since minimal inference engines rarely support it. It sits between LeNet5 and ResNet18Cifar in compute and uses larger kernels and strided convolutions.

> Parameter counts are for the learnable weights and biases (BatchNorm running statistics excluded) and are rounded.
