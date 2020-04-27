#include <cstdio>
#include <cmath>
#include <random>
#include <ctime>
#include <iostream>
#include "nn.hpp"


int main(void){


  // Constant initialization
  const int n = 100000; // nb of training examples
  const int d = 50; // dimension of variables
  const int dim_h = 150; // nb of hidden units
  const int niter = 100; // number of iteration for gradient descent
  const double lr = .05; // learning rate .01
  const int freq_show = 10; // frequence to show iteration and precision



  // Train a neural network with GPU acceleration
  nnet nnet_gpu(n, d, dim_h, niter, lr, freq_show);
  nnet_gpu.random_init();
  nnet_gpu.train_on_gpu();
  nnet_gpu.~nnet();



  // Train a neural network only on CPU
  nnet nnet_cpu(n, d, dim_h, niter, lr, freq_show);
  nnet_cpu.random_init();
  nnet_cpu.train_on_cpu();
  nnet_cpu.~nnet();



  return 0;
}
