#include <cstdio>
#include <cmath>
#include <random>
#include <ctime>
#include <iostream>
#include "nn.hpp"
#define block_size 32 // 32*32 = 1024 = maximal number of threads
#define tile_size 32

#define CHECK_CUDA_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}




// Matrix multiplication on GPU with shared memory
__global__
void d_matrix_mul_tile(double *d_m1, double *d_m2, double *d_res, const int h1, const int w1, const int h2, const int w2, const int nb_loop) {

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x*blockDim.x + tx;
  int row = blockIdx.y*blockDim.y + ty;
  __shared__ double m1_tile[tile_size][tile_size];
  __shared__ double m2_tile[tile_size][tile_size];


  double trans=0; // value to feed in d_res[row,col]



  for (int i = 0 ; i < nb_loop ; i++) {
    // Step 1 : fill values in shared Memory

    // condition to avoid out-of-bounds global memory accesses
    if ((i*tile_size+tx >= w1) || row >= h1) {
      m1_tile[ty][tx] = 0;
    }
    else {
      m1_tile[ty][tx] = d_m1[(row*w1)+ (i*tile_size + tx)];
    }


    if ((i*tile_size+ty >= h2) || col >= w2) {
      m2_tile[ty][tx] = 0;
    }
    else {
      m2_tile[ty][tx] = d_m2[i*tile_size*w2 + ty*w2 + col];
    }

    // wait for all threads
    __syncthreads();


    // Step 2 : Augment trans values
    for (int k = 0 ; k < tile_size ; k++) trans += m1_tile[ty][k]*m2_tile[k][tx];
    __syncthreads();
  }

  // if condition because of ceil in host function (last values)
  if ( (row < h1) && (col < w2) ) {d_res[row*w2+col] = trans;}

}




// Naive implementation of matrix multiplication on GPU
__global__
void d_matrix_mul(double *d_m1, double *d_m2, double *d_res, const int h1, const int w1, const int h2, const int w2) {

  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;

  // If thread is in the range h1*w2, then it will feed one value of the result
  if (ix < w2 && iy < h1) {
    double trans = 0.;
    for (int k = 0 ; k < w1 ; k++) {
      trans += d_m1[iy*w1+k]*d_m2[k*w2+ix];
    }
    d_res[iy*w2+ix] = trans;
    __syncthreads();
  }


}



// Multiply 2 matrix h_m1 and h_m2 on host, with calculation on device
void nnet::matrix_mul_gpu(double *h_m1, double *h_m2, double *h_res, const int h1, const int w1, const int h2, const int w2) {

  // GPU memory allocation
  double *d_m1, *d_m2, *d_res;
  CHECK_CUDA_ERROR(cudaMalloc(&d_m1, h1*w1*sizeof(double)));
  CHECK_CUDA_ERROR(cudaMalloc(&d_m2, h2*w2*sizeof(double)));
  CHECK_CUDA_ERROR(cudaMalloc(&d_res, h1*w2*sizeof(double)));

  CHECK_CUDA_ERROR(cudaMemcpy(d_m1, h_m1, h1*w1*sizeof(double), cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(d_m2, h_m2, h2*w2*sizeof(double), cudaMemcpyHostToDevice));


  // Matrix multiplication on GPU
  // We need ceil function to deal with any matrix dimension (not only factor of block_size)
  // nb_loop variable is the number of tiled matrix we need to get the final result
  dim3 nb_threads(block_size,block_size,1);
  dim3 nb_blocs((int)ceil((float)w2/block_size),(int)ceil((float)h1/block_size),1);
  const int nb_loop = ceil((float)w1/tile_size);
  d_matrix_mul_tile<<<nb_blocs, nb_threads>>>(d_m1, d_m2, d_res, h1, w1, h2, w2, nb_loop);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());


  // Transfer GPU to CPU and free memory
  CHECK_CUDA_ERROR(cudaMemcpy(h_res, d_res, h1*w2*sizeof(double), cudaMemcpyDeviceToHost));
  cudaFree(d_m1); cudaFree(d_m2); cudaFree(d_res);

}





// Constructor for the class : allocation for dataset and weights
nnet::nnet(const int n, const int d, const int dim_h, const int niter, const double lr, const int freq_show): freq_show(freq_show), n(n), d(d), dim_h(dim_h), niter(niter), lr(lr) {
  X = new double[n*d];
  y = new int[n];
  w1 = new double[d*dim_h];
  w2 = new double[dim_h];
  return;
}


// Destructor of the class
nnet::~nnet() {
  delete [] X;
  delete [] y;
  delete [] w1;
  delete [] w2;
}




// Vector and real-value sigmoid functions
void nnet::sigmoid(double * in, double * out, const int size) {
  for (int i = 0; i < size ; i++) {
    out[i] = 1/(1+exp(-in[i]));
  }
}

double nnet::sigmoid1(double inp) {
  return 1/(1+exp(-inp));
}




// Functions to calculate the delta's errors in the backward step
void nnet::calculate_last_delta(double *ypred, int *y, double *z2, double *res, const int n){
  double norm_const = -1./n;
  double sig_z2;
  for (int i = 0 ; i < n ; i++) {
    sig_z2 = sigmoid1(z2[i]);
    res[i] = norm_const*(y[i]/ypred[i] - (1-y[i])/(1-ypred[i])) * sig_z2*(1-sig_z2);
  }
}


void nnet::calculate_delta1(double *z1, double *res, double *trans, const int h, const int w){
  // delta1 of size n*dim_h
  double sig_z1;
  for (int i = 0 ; i < h ; i++) {
    for (int j = 0 ; j < w ; j++) {
      // calculate delta_ij
      sig_z1 = sigmoid1(z1[i*w+j]);
      res[i*w+j] = trans[i*w+j] * sig_z1*(1-sig_z1);
    }
  }
}




// Random initialization of weights
void nnet::_random_init_weight(double *w1, double *w2, const int d, const int dim_h) {
  std::mt19937 G(time(NULL));
  std::normal_distribution<double> normal(0,1);

  // Fill w1 randomly
  for (int i = 0 ; i < d ; i++) {
    for (int j = 0 ; j < dim_h ; j++) {
      w1[i*dim_h+j] = normal(G);
    }
  }

  // Fill w2 randomly
  for (int i = 0 ; i < dim_h ; i++) {
    w2[i] = normal(G);
  }
}



// Random generation of dataset for classification
void nnet::random_init() {
  std::mt19937 G(time(NULL));
  std::normal_distribution<double> normal(0,1);


  // 1. Random dataset generation
  float nb_one = 0.;
  float threshold = 0.;
  for (int i = 0 ; i < n ; i++) {
    threshold = 0.;
    // every entry of X is a normal realization
    for (int j = 0 ; j < d ; j++) {
      X[i*d+j] = normal(G);
      threshold += X[i*d+j];
    }
    y[i] = 0;
    if ( (threshold > d/15) ) {y[i] = 1; nb_one++;}
  }
  std::cout << "Percentage of 1 in our dataset : " << nb_one/n << std::endl;




  // 1. Another dataset generation
  /*
  std::normal_distribution<double> normal0(0,1);
  std::normal_distribution<double> normal1(2,2);
  int middle = (int)(n/2);

  for (int i = 0 ; i < middle ; i++) {
    for (int j = 0 ; j < d ; j++) X[i*d+j] = normal0(G);
    y[i] = 0;
  }

  for (int i = middle ; i < n ; i++) {
    for (int j = 0 ; j < d ; j++) X[i*d+j] = normal1(G);
    y[i] = 1;
  }
  */

  // 2. Random weights initialization
  _random_init_weight(w1, w2, d, dim_h);
}




// Batch gradient descent algorithm to train the NN on CPU
void nnet::train_on_cpu() {
  std::cout << "----------- Start training on CPU --------------" << std::endl;
  // Memory allocation
  double *X_transpose, *ypred;
  double *w1_transpose;
  double *grad_w1, *grad_w2;
  double *a1, *a2;
  double *z1, *z2;
  double *delta2, *delta1;

  X_transpose = new double[d*n];
  ypred = new double[n];
  w1_transpose = new double[dim_h*d];
  grad_w1 = new double[d*dim_h] ; grad_w2 = new double[dim_h];
  z1 = new double[n*dim_h]; z2 = new double[n];
  a1 = new double[n*dim_h]; a2 = new double[n];
  delta2 = new double[n]; delta1 = new double[n*dim_h];

  // Transpose matrices X and w1
  transpose(X, X_transpose, n, d);
  transpose(w1, w1_transpose, d, dim_h);


  // Timer for CPU algorithm
  double duration=0, duration_cpu=0;
  std::clock_t start_cpu, end_cpu;
  start_cpu = std::clock();


  // Algorithm
  double prec = 0.; // precision of our NN
  for (int iter = 0 ; iter < niter ; iter++) {
    // 1. Forward pass

    // First layer : linear operation + sigmoid
    matrix_mul(X, w1, z1, n, d, d, dim_h);
    sigmoid(z1, a1, n*dim_h);

    // Second layer : linear operation + sigmoid
    matrix_mul(a1, w2, z2, n, dim_h, dim_h, 1);
    sigmoid(z2, ypred, n);

    // Precision of our predictions
    prec = 0.;
    for (int i = 0 ; i < n ; i++) {
      if ( (ypred[i] >= 0.5) == y[i] ) prec++;
    }
    prec /= n;

    // 2. Backward pass (backprop algorithm)
    // We start with the second (=last) layer and back propagate error

    // Error and gradient calculation
    calculate_last_delta(ypred, y, z2, delta2, n);
    matrix_mul(delta2, a1, grad_w2, 1, n, n, dim_h);



    // Then we can calculate the second gradient
    // trans will contain transition values (weight transpose time delta2) in order to calculate delta1
    double *trans;
    trans = new double[n*dim_h];
    matrix_mul(delta2, w2, trans, n, 1, 1, dim_h);
    calculate_delta1(z1, delta1, trans, n, dim_h);
    matrix_mul(X_transpose, delta1, grad_w1, d, n, n, dim_h);
    delete [] trans;

    // Batch gradient descent step
    gd_step(w1, grad_w1, d*dim_h, lr);
    gd_step(w2, grad_w2, dim_h, lr);


    if (iter % freq_show == 0) {
      std::cout << "Epoch no. " << iter << ", precision = " << prec << std::endl;
    }

  }

  end_cpu = std::clock();
  duration = end_cpu - start_cpu;
  duration_cpu = (float)duration/(CLOCKS_PER_SEC/1000);
  std::cout << "Time of execution on CPU : " << duration_cpu << " ms" << std::endl;


  // Free memory
  delete [] X_transpose; delete [] ypred;
  delete [] w1_transpose;
  delete [] a1; delete [] a2;
  delete [] z1; delete [] z2;
  delete [] delta1; delete [] delta2;
  delete [] grad_w1; delete [] grad_w2;


}




// Matrix multiplication on CPU
void nnet::matrix_mul(double *m1, double *m2, double *res, const int h1, const int w1, const int h2, const int w2) {
  double res_trans;
  for (int i = 0 ; i < h1 ; i++) {
    for (int j = 0 ; j < w2 ; j++) {
      // Compute (i,j) coordinate result of matrix multiplication
      res_trans = 0.;
      for (int k = 0 ; k < w1 ; k++) {
        res_trans += m1[i*w1+k]*m2[k*w2+j];
      }
      res[i*w2+j] = res_trans;
    }
  }
}



// Gradient of BCE loss
void nnet::grad_loss(int *y, double *ypred, double *res, const int n) {
  double norm_const = -1./n;
  for (int i = 0 ; i < n ; i++) {
    res[i] = norm_const*(y[i]/ypred[i] - (1-y[i])/(1-ypred[i]));
  }
}


// Transpose matrix mat of size h*w in res
void nnet::transpose(double *mat, double *res, const int h, const int w) {
  for (int i = 0 ; i < w ; i++) {
    for (int j = 0 ; j < h ; j++) {
      res[i*h+j] = mat[j*w+i];
    }
  }
}


// Batch gradient descent step with learning rate lr
void nnet::gd_step(double *weight, double *grad, const int size, const double lr) {
  for (int i = 0 ; i < size ; i++) {
    weight[i] -= lr*grad[i];
  }
}


// Batch gradient descent algorithm to train the NN on GPU
void nnet::train_on_gpu() {
  std::cout << "----------- Start training with GPU acceleration --------------" << std::endl;

  // Memory allocation
  double *X_transpose, *ypred;
  double *w1_transpose;
  double *grad_w1, *grad_w2;
  double *a1, *a2;
  double *z1, *z2;
  double *delta2, *delta1;

  X_transpose = new double[d*n];
  ypred = new double[n];
  w1_transpose = new double[dim_h*d];
  grad_w1 = new double[d*dim_h] ; grad_w2 = new double[dim_h];
  z1 = new double[n*dim_h]; z2 = new double[n];
  a1 = new double[n*dim_h]; a2 = new double[n];
  delta2 = new double[n]; delta1 = new double[n*dim_h];

  // Transpose matrices X and w1
  transpose(X, X_transpose, n, d);
  transpose(w1, w1_transpose, d, dim_h);


  // Timer for algorithm with GPU acceleration
  double duration=0, duration_gpu=0;
  std::clock_t start_gpu, end_gpu;
  start_gpu = std::clock();


  // Algorithm
  double prec = 0.;
  for (int iter = 0 ; iter < niter ; iter++) {
    // 1. Forward pass

    // First layer : linear operation and sigmoid
    matrix_mul_gpu(X, w1, z1, n, d, d, dim_h);
    sigmoid(z1, a1, n*dim_h);

    // Second layer : linear operation and sigmoid
    matrix_mul_gpu(a1, w2, z2, n, dim_h, dim_h, 1);
    sigmoid(z2, ypred, n);

    // Precision of our predictions
    prec = 0.;
    for (int i = 0 ; i < n ; i++) {
      if ( (ypred[i] >= 0.5) == y[i] ) prec++;
    }


    // 2. Backward pass (backprop algorithm)

    // We start with the second (=last) layer and back propagate error
    // Error and gradient calculation
    calculate_last_delta(ypred, y, z2, delta2, n);
    matrix_mul_gpu(delta2, a1, grad_w2, 1, n, n, dim_h);


    // Then we can calculate the second gradient
    // trans will contain transition values (wieght transpose time delta2) in order to calculate delta1
    double *trans;
    trans = new double[n*dim_h];
    matrix_mul_gpu(delta2, w2, trans, n, 1, 1, dim_h);
    calculate_delta1(z1, delta1, trans, n, dim_h);
    matrix_mul_gpu(X_transpose, delta1, grad_w1, d, n, n, dim_h);
    delete [] trans;

    // Batch gradient descent step
    gd_step(w1, grad_w1, d*dim_h, lr);
    gd_step(w2, grad_w2, dim_h, lr);


    if (iter % freq_show == 0) {
      std::cout << "Epoch no. " << iter << ", precision = " << prec/n << std::endl;
    }

  }

  end_gpu = std::clock();
  duration = end_gpu - start_gpu;
  duration_gpu = (float)duration/(CLOCKS_PER_SEC/1000);
  std::cout << "Time of execution on GPU : " << duration_gpu << " ms" << std::endl;



  // Free memory
  delete [] X_transpose;
  delete [] ypred;

  delete [] w1_transpose;
  delete [] a1; delete [] a2;
  delete [] z1; delete [] z2;
  delete [] delta1; delete [] delta2;
  delete [] grad_w1; delete [] grad_w2;


}
