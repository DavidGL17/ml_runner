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


## Results

### Results for model : LeNet300100
|   Runner   |               Backend               | Runs  |    Min     |    Max     |    Mean    |   Median   |  StdDev   | Errors |
| :--------: | :---------------------------------: | :---: | :--------: | :--------: | :--------: | :--------: | :-------: | :----: |
|   python   | PyTorch 2.13.0+cpu (CPU, 6 threads) |  90   | 91.896 µs  | 254.490 µs | 98.498 µs  | 93.746 µs  | 19.302 µs |   0    |
|    rust    |      ndarray (matrixmultiply)       |  90   | 49.497 µs  | 120.293 µs | 50.942 µs  | 49.697 µs  | 7.708 µs  |   0    |
|    rust    |             simd (wide)             |  90   | 37.899 µs  | 56.599 µs  | 38.973 µs  | 38.099 µs  | 3.058 µs  |   0    |
|    rust    |           ndarray (BLAS)            |  90   | 15.199 µs  | 120.298 µs | 17.292 µs  | 15.500 µs  | 11.811 µs |   0    |
| python@pi4 | PyTorch 2.13.0+cpu (CPU, 4 threads) |  90   |  2.676 ms  |  3.022 ms  |  2.792 ms  |  2.780 ms  | 68.336 µs |   0    |
|  rust@pi4  |      ndarray (matrixmultiply)       |  90   | 194.628 µs | 263.629 µs | 201.892 µs | 198.406 µs | 11.821 µs |   0    |
|  rust@pi4  |             simd (wide)             |  90   | 209.332 µs | 474.202 µs | 235.135 µs | 214.111 µs | 42.661 µs |   0    |
|  rust@pi4  |           ndarray (BLAS)            |  90   | 136.740 µs | 407.924 µs | 175.923 µs | 172.888 µs | 42.764 µs |   0    |


### Results for model : LeNet5
|   Runner   |               Backend               | Runs  |    Min     |    Max     |    Mean    |   Median   |   StdDev   | Errors |
| :--------: | :---------------------------------: | :---: | :--------: | :--------: | :--------: | :--------: | :--------: | :----: |
|   python   | PyTorch 2.13.0+cpu (CPU, 6 threads) |  90   | 599.899 µs |  1.163 ms  | 744.490 µs | 717.450 µs | 103.960 µs |   0    |
|    rust    |      ndarray (matrixmultiply)       |  90   |  1.050 ms  |  1.219 ms  |  1.064 ms  |  1.061 ms  | 21.132 µs  |   0    |
|    rust    |             simd (wide)             |  90   | 147.500 µs | 194.600 µs | 150.967 µs | 147.900 µs |  8.961 µs  |   0    |
|    rust    |           ndarray (BLAS)            |  90   |  1.048 ms  |  6.422 ms  |  2.029 ms  |  1.914 ms  | 622.109 µs |   0    |
| python@pi4 | PyTorch 2.13.0+cpu (CPU, 4 threads) |  90   |  3.372 ms  |  7.290 ms  |  3.607 ms  |  3.497 ms  | 523.462 µs |   0    |
|  rust@pi4  |      ndarray (matrixmultiply)       |  90   |  3.495 ms  |  3.781 ms  |  3.540 ms  |  3.512 ms  | 63.278 µs  |   0    |
|  rust@pi4  |             simd (wide)             |  90   | 682.682 µs | 873.718 µs | 688.967 µs | 683.451 µs | 21.530 µs  |   0    |
|  rust@pi4  |           ndarray (BLAS)            |  90   |  3.641 ms  |  8.432 ms  |  4.016 ms  |  3.772 ms  | 715.717 µs |   0    |


### Results for model : ResNet18Cifar
|   Runner   |               Backend               | Runs  |     Min     |     Max     |    Mean     |   Median    |   StdDev   | Errors |
| :--------: | :---------------------------------: | :---: | :---------: | :---------: | :---------: | :---------: | :--------: | :----: |
|   python   | PyTorch 2.13.0+cpu (CPU, 6 threads) |  90   |  11.216 ms  |  13.599 ms  |  12.305 ms  |  12.293 ms  | 624.581 µs |   0    |
|    rust    |      ndarray (matrixmultiply)       |  90   | 1718.186 ms | 2095.916 ms | 1774.921 ms | 1751.964 ms | 61.342 ms  |   0    |
|    rust    |             simd (wide)             |  90   | 116.151 ms  | 136.158 ms  | 120.161 ms  | 119.236 ms  |  3.375 ms  |   0    |
|    rust    |           ndarray (BLAS)            |  90   | 1716.017 ms | 2207.893 ms | 1766.499 ms | 1735.467 ms | 86.572 ms  |   0    |
| python@pi4 | PyTorch 2.13.0+cpu (CPU, 4 threads) |  90   | 115.224 ms  | 217.947 ms  | 130.071 ms  | 122.797 ms  | 18.866 ms  |   0    |
|  rust@pi4  |      ndarray (matrixmultiply)       |  90   | 5911.441 ms | 6358.975 ms | 5935.305 ms | 5916.225 ms | 67.651 ms  |   0    |
|  rust@pi4  |             simd (wide)             |  90   | 584.094 ms  | 639.562 ms  | 588.680 ms  | 587.118 ms  |  6.923 ms  |   0    |
|  rust@pi4  |           ndarray (BLAS)            |  90   | 5926.182 ms | 9536.005 ms | 5973.480 ms | 5932.462 ms | 377.677 ms |   0    |


### Results for model : LSTMClassifier
|   Runner   |               Backend               | Runs  |    Min     |    Max     |    Mean    |   Median   |   StdDev   | Errors |
| :--------: | :---------------------------------: | :---: | :--------: | :--------: | :--------: | :--------: | :--------: | :----: |
|   python   | PyTorch 2.13.0+cpu (CPU, 6 threads) |  90   | 452.400 µs |  2.612 ms  | 669.788 µs | 591.450 µs | 303.512 µs |   0    |
|    rust    |      ndarray (matrixmultiply)       |  90   |  1.107 ms  |  1.514 ms  |  1.153 ms  |  1.139 ms  | 57.725 µs  |   0    |
|    rust    |             simd (wide)             |  90   | 815.800 µs | 973.500 µs | 833.559 µs | 819.350 µs | 29.408 µs  |   0    |
|    rust    |           ndarray (BLAS)            |  90   | 812.200 µs |  1.772 ms  |  1.261 ms  |  1.343 ms  | 219.139 µs |   0    |
| python@pi4 | PyTorch 2.13.0+cpu (CPU, 4 threads) |  90   |  5.856 ms  |  8.289 ms  |  6.133 ms  |  6.038 ms  | 403.953 µs |   0    |
|  rust@pi4  |      ndarray (matrixmultiply)       |  90   |  3.776 ms  |  4.384 ms  |  3.880 ms  |  3.847 ms  | 97.515 µs  |   0    |
|  rust@pi4  |             simd (wide)             |  90   |  3.681 ms  |  5.109 ms  |  3.848 ms  |  3.785 ms  | 220.169 µs |   0    |
|  rust@pi4  |           ndarray (BLAS)            |  90   |  2.633 ms  |  3.080 ms  |  2.779 ms  |  2.756 ms  | 111.023 µs |   0    |


### Results for model : AlexNetLite
|   Runner   |               Backend               | Runs  |    Min     |    Max     |    Mean    |   Median   |   StdDev   | Errors |
| :--------: | :---------------------------------: | :---: | :--------: | :--------: | :--------: | :--------: | :--------: | :----: |
|   python   | PyTorch 2.13.0+cpu (CPU, 6 threads) |  90   | 875.400 µs |  1.745 ms  |  1.184 ms  |  1.179 ms  | 176.547 µs |   0    |
|    rust    |      ndarray (matrixmultiply)       |  90   | 105.468 ms | 128.976 ms | 109.770 ms | 109.177 ms |  3.337 ms  |   0    |
|    rust    |             simd (wide)             |  90   | 15.001 ms  | 17.005 ms  | 15.523 ms  | 15.447 ms  | 306.171 µs |   0    |
|    rust    |           ndarray (BLAS)            |  90   | 149.402 ms | 186.177 ms | 154.248 ms | 152.428 ms |  5.196 ms  |   0    |
| python@pi4 | PyTorch 2.13.0+cpu (CPU, 4 threads) |  90   | 12.357 ms  | 36.246 ms  | 14.286 ms  | 12.797 ms  |  4.802 ms  |   0    |
|  rust@pi4  |      ndarray (matrixmultiply)       |  90   | 355.300 ms | 363.338 ms | 356.673 ms | 356.149 ms |  1.477 ms  |   0    |
|  rust@pi4  |             simd (wide)             |  90   | 69.085 ms  | 75.760 ms  | 70.166 ms  | 69.712 ms  |  1.235 ms  |   0    |
|  rust@pi4  |           ndarray (BLAS)            |  90   | 361.759 ms | 369.024 ms | 362.892 ms | 362.431 ms |  1.486 ms  |   0    |

