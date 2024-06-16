#include <Eigen/Dense>
#include <format>
#include <iostream>
#include <stdexcept>
#include <stdlib.h>
#include <typeinfo>
#include <unordered_map>
#include <vector>

using namespace std;
using Eigen::MatrixXd;

template <typename T> class Generator {
public:
  Generator(int seed) { srand(seed); }

  // TODO: imprve random number generation quality
  void normal(T l, T r, vector<T>& data) {
    if ((long long)(l - r) - (l - r) != 0) {
      throw invalid_argument(
          "the range must have an integer length, e.g. 1.5 - .5 = 1");
    }

    for (int i = 0; i < data.size(); i++) {
      data[i] = l + (rand() % (int)(r - l)) + (rand() / (T)RAND_MAX);
    }
  }
};

template <typename D> class Tensor {
public:
  Tensor(int rows, int cols, Generator<D>& g, int N = 1) {
    ncols = cols;
    nrows = rows;
    dim3 = N;
    length = cols * rows * N;
    tsize = length * sizeof(D);
    data = new vector<D>(length);
    g.normal(-1, 1, *data);
  }

  Tensor(float* d, int rows, int cols, int N = 1) {
    nrows = rows;
    ncols = cols;
    dim3 = N;
    length = cols * rows * N;
    tsize = length * sizeof(D);
    data = new vector<D>(d, d + length);
  }

  Tensor(int rows, int cols, int N = 1) {
    nrows = rows;
    ncols = cols;
    dim3 = N;
    length = cols * rows * N;
    tsize = length * sizeof(D);
    data = new vector<D>(length);
  }

  int len() { return length; }
  int size() { return tsize; }
  int rows() { return nrows; }
  int cols() { return ncols; }

  // adapt to handle 3D tensors (later ndim tensors)
  friend Tensor<D>* operator*(Tensor<D>& L, Tensor<D>& R) {
    Tensor<D>* O = new Tensor<D>(L.nrows, R.ncols);
    L.mmul(R, *O);
    return O;
  }

  void mmul(Tensor<D>& R, Tensor<D>& O) {
    Tensor<D> L = *this;
    if (L.ncols != R.nrows)
      throw invalid_argument(
          "Inner dimension of input tensors does not match.");
    if (O.nrows != L.nrows && O.ncols != R.ncols)
      throw invalid_argument("Provided output tensor is not the correct size.");

    for (int r = 0; r < L.nrows; r++) {
      for (int c = 0; c < R.ncols; c++) {
        for (int i = 0; i < L.ncols; i++) {
          O[r * R.ncols + c] += L[r * L.ncols + i] * R[c + R.ncols * i];
        }
      }
    }
  }

  Tensor<D> T() {
    Tensor<D> t = *this;
    Tensor<D> transpose(ncols, nrows);
    for (int r = 0; r < nrows; r++) {
      for (int c = 0; c < ncols; c++) {
        transpose[c * ncols + r] = t[c + r * ncols];
      }
    }
    return transpose;
  }

  D& operator[](int index) {
    if (index >= length) {
      throw invalid_argument("Tensor array index out of bounds.");
    }
    return (*data)[index];
  }

  const vector<D> Data() { return vector<D>(*data); }

  friend ostream& operator<<(ostream& stream, Tensor& t) {
    stream << "\n";
    for (int i = 0; i < t.len(); i++) {
      stream << format("{:.4f}", (*t.data)[i]) << ' ';
      if (i % t.ncols == t.ncols - 1) {
        stream << "\n";
      }
    }
    return stream;
  }

private:
  vector<D>* data;
  int ncols;
  int nrows;
  int dim3;
  int length;
  size_t tsize;
};

template <typename T> MatrixXd to_matrix(Tensor<T> t) {
  MatrixXd m(t.rows(), t.cols());
  for (int r = 0; r < t.rows(); r++) {
    for (int c = 0; c < t.cols(); c++) {
      m(r, c) = t[r * t.cols() + c];
    }
  }
  return m;
};

template <typename T> class IComponent {
public:
  virtual Tensor<T>* forward(Tensor<T>&) = 0;
  virtual void backward() = 0;
};

template <typename T> class Component : public IComponent<T> {
public:
  Component() {}
  string name() { return typeid(*this).name(); }
};

template <typename T> class NeuralNetwork : public Component<T> {
public:
  NeuralNetwork() {}

protected:
  unordered_map<string, Component<T>*> layers;
};

template <typename T> class Loss : public Component<T> {
public:
  Loss() {}
  Tensor<T>* forward(Tensor<T>& A) {
    cout << this->name() << ": forward" << endl;
    return &A;
  }
  void backward() { cout << this->name() << ": backward" << endl; }
};

template <typename T> class LinearLayer : public Component<T> {
public:
  LinearLayer(int sIn, int sOut, Generator<T>& g) {
    w = new Tensor<T>(sIn, sOut, g);
    b = new Tensor<T>(sOut, 1, g);
  }

  LinearLayer(Component<T> comp) {}

  Tensor<T>* forward(Tensor<T>& X) {
    // TODO: add in the biases
    return *w * X;
  }

  void backward() { cout << this->name() << ": backward" << endl; }

  Tensor<T>* W() { return w; };
  Tensor<T>* B() { return b; };

protected:
  Tensor<T>* w;
  Tensor<T>* b;
};

template <typename T> class Activation : public Component<T> {
public:
  Activation() {}
  Tensor<T>* forward(Tensor<T>& X) {
    cout << this->name() << ": forward" << endl;
    return &X;
  }

  void backward() { cout << this->name() << ": backward" << endl; }

  // TODO: every component should have a print operator override.
};

template <typename T> class SimpleNet : public NeuralNetwork<T> {
public:
  SimpleNet(Generator<T>& g) {
    this->layers.emplace("l0", new LinearLayer<T>(10, 2, g));
    this->layers.emplace("activation", new Activation<T>());
    this->layers.emplace("loss", new Loss<T>());
  };

  Tensor<T>* forward(Tensor<T>& X) {
    LinearLayer<T> l0 = *(LinearLayer<T>*)this->layers["l0"];
    cout << "-----------------------" << endl;
    cout << this->name() << ": forward\n" << endl;
    cout << "input: " << X << endl;
    cout << "weights: " << *l0.W() << endl;
    cout << "bias: " << *l0.B() << endl;

    MatrixXd mW = to_matrix(*l0.W());
    MatrixXd mX = to_matrix(X);
    Eigen::IOFormat mstyle(4);
    cout << "reference logits: \n" << (mW * mX).format(mstyle) << endl;

    Tensor<T>* logits = l0.forward(X);
    cout << "\nlogits: " << *logits << endl;
    cout << "-----------------------" << endl;

    this->layers["activation"]->forward(X);
    this->layers["loss"]->forward(X);
    return logits;
  };

  void backward() {
    this->layers["loss"]->backward();
    this->layers["activation"]->backward();
    this->layers["l0"]->backward();
  };
};

int main() {
  Generator<float> g(0);
  SimpleNet<float> nn(g);
  Tensor<float> X(2, 10, g);
  nn.forward(X);
  nn.backward();
}
