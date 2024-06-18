#include "tests.hpp"
#include <iostream>

using namespace std;
using namespace NN;

bool linearForward() {
  Generator<float> G(0);
  LinearLayer<float> L(10, 10, G);
  Tensor<float> X(10, 2, G);
  Tensor<float> Logits = L.forward(X);

  MatrixXd MW = toMatrix(L.w());
  MatrixXd MX = toMatrix(X);
  VectorXd MB = toVector(L.b());

  cout << L;
  cout << "logits: " << endl;
  cout << Logits;

  cout << "\n----------------------" << endl;
  cout << "reference implementation: " << endl;
  cout << "\nlogits: \n"
       << ((MW * MX).colwise() + MB).format(EIGEN_STYLE) << endl;
  cout << "\n----------------------" << endl;

  // TODO: function to convert MatrixXd & VectorXd to Tensors & compare
  // return result of comparison
  // only print one quantity
  return true;
}

int main() { linearForward(); }
