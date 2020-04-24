# Cuda-neural-network


One layer neural network class in C++ from scratch, for illustration purpose only. It show how to improve algorithmic efficiency just by doing some matrix multiplication on GPU. We provide three different versions : one in C++, another one in CUDA C++ with naive implementation of matrix multiplication and the last one with better use of shared memory. Just by doing these operations, we can improve the efficiency by a factor 30-40 for big enough dataset. The command line to compile the code is :

```console
nvcc nn.cu main.cu -o nn.exe
```
