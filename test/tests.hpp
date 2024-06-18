#include "../neural_network.hpp"
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;
const Eigen::IOFormat EIGEN_STYLE(4);
using namespace NN;

template <typename D> MatrixXd toMatrix(Tensor<D>& T) {
  MatrixXd M(T.rows(), T.cols());
  for (int R = 0; R < T.rows(); R++) {
    for (int C = 0; C < T.cols(); C++) {
      M(R, C) = T[R * T.cols() + C];
    }
  }
  return M;
};

template <typename D> VectorXd toVector(Tensor<D>& T) {
  VectorXd V(T.rows(), T.cols());
  for (int R = 0; R < T.rows(); R++) {
    for (int C = 0; C < T.cols(); C++) {
      V(R, C) = T[R * T.cols() + C];
    }
  }
  return V;
};
