#include "tests.hpp"
#include "../neural_network.hpp"
#include <iostream>

using namespace std;
using namespace NN;

bool linearForward() {
  Generator<float> G(0);
  LinearLayer<float> LL(10, 10, G);
  TanH<float> A;
  MSE<float> L;

  Tensor<float> X(10, 2, G);
  Tensor<float> Logits = LL.forward(X);
  Tensor<float> Activations = A.forward(Logits);
  // Tensor<float> Loss = L.forward(Activations);

  MatrixXd MW = toMatrix(LL.w());
  MatrixXd MX = toMatrix(X);
  VectorXd MB = toVector(LL.b());

  cout << LL;
  cout << "logits: ";
  cout << Logits;
  cout << "\nactivations: ";
  cout << Activations;
  cout << "\nmean squared error: ";
  // cout << Loss;

  MatrixXd RLogits = ((MW * MX).colwise() + MB);
  MatrixXd RActivations = RLogits.array().tanh();
  cout << "\n----------------------" << endl;
  cout << "logits (reference): \n" << RLogits.format(EIGEN_STYLE) << endl;
  cout << "\ntanh (reference): \n" << RActivations.format(EIGEN_STYLE) << endl;
  cout << "\n----------------------" << endl;

  // TODO: function to convert MatrixXd & VectorXd to Tensors & compare
  // return result of comparison
  // only print one quantity
  return true;
}

int main() { linearForward(); }
