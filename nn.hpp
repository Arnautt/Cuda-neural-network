// Matrix multiplication on GPU, with and without shared memory
__global__ void d_matrix_mul(double *d_m1, double *d_m2, double *d_res, const int h1, const int w1, const int h2, const int w2);
__global__ void d_matrix_mul_tile(double *d_m1, double *d_m2, double *d_res, const int h1, const int w1, const int h2, const int w2, const int nb_loop);


// Neural network class
class nnet {
private:
  // dataset
  double *X;
  int *y;

  // functions for training algorithm
  void _random_init_weight(double *w1, double *w2, const int d, const int dim_h);
  void calculate_last_delta(double *ypred, int *y, double *z2, double *res, const int n);
  void calculate_delta1(double *z1, double *res, double *trans, const int h, const int w);
  void sigmoid(double * in, double * out, const int size);
  void grad_loss(int *y, double *ypred, double *res, const int n);
  void transpose(double *mat, double *res, const int h, const int w);
  void gd_step(double *weight, double *grad, const int size, const double lr);
  void matrix_mul(double *m1, double *m2, double *res, const int h1, const int w1, const int h2, const int w2);
  void matrix_mul_gpu(double *h_m1, double *h_m2, double *h_res, const int h1, const int w1, const int h2, const int w2);
  void matrix_mul_gpu2(double *d_m1, double *d_m2, double *d_res, double *h_res, const int h1, const int w1, const int h2, const int w2);

public:
  const int n; // nb of training examples
  const int d; // dimension of variables
  const int dim_h; // nb of hidden units
  const int niter; // number of iteration for gradient descent
  const double lr; // learning rate
  const int freq_show; // frequence to show iteration and precision
  double *w1; // weights of the first layer
  double *w2; // weights of the second layer

  // Constructor, destructor and functions to train the NN
  nnet(const int n, const int d, const int dim_h, const int niter=1000, const double lr=0.01, const int freq_show=1000);
  ~nnet();
  void train_on_cpu();
  void train_on_gpu();
  void random_init();

};
