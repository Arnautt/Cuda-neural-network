# Cuda-neural-network


One layer neural network class in C++ from scratch, for illustration purpose only. It show how to improve algorithmic efficiency just by doing matrix multiplication on GPU. We provide tree different versions : one in C++, another one in CUDA C++ with naive implementation of matrix multiplication and the last one with better use of shared memory.

- Command line to compile the code :

```console nvcc nn.cu main.cu -o nn.exe ```
