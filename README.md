# Neural Network Library with CUDA/C++

## Goal

I plan on creating a very rudimentary neural network library that should allow the programmer to instantiate a `NeuralNet` object, by specifying the number of layers, the number of hidden units per layer, as well as the activation functions (ReLU/Sigmoid/Softmax) for each layer. The model object should have an API for functions like `train` and `predict`. For now, I will only focus on a barebones implementation of a NN, without diving into improvements like regularization like L2 or dropout, or Adam optimization.

First, I will implement a neural network using plain C++ (without worrying about parallelization) in `/cpu`, while trying to keep it as modular and compact as possible. I will verify that the implementation is functional using the benchmarks listed on [Yann LeCun's website](https://yann.lecun.com/exdb/mnist/). Specifically, I will try to hit a 3-5% test error rate (to account for random initialization and variability in number of epochs) using a 3-layer network with 300+100 hidden units as a sanity check.

Once that works, I will move on to parallelizing the library with CUDA for efficiency, in `/gpu`. Once I run similar tests to verify the implementation, I will provide comparisons of runtimes for training and inference using a CPU vs a GPU.

## Description of Dataset

The benchmark dataset I will use for this project is from the classic MNIST handwritten digit recognition problem, with $28 * 28$ pixel images as input into the NN as an array of 784 float values between -1 and 1.

The `/data_MNIST` directory contains 60,000 training examples and 10,000 testing examples along with their labels, taken from [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) (originally distributed by [Yann LeCun](https://yann.lecun.com/exdb/mnist/), but I think the original website does not allow downloads anymore).
