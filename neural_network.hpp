#include <format>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <stdlib.h>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <vector>

using namespace std;

namespace NN {
template <typename T> class Generator {
public:
  Generator(int Seed);
  void normal(T L, T R, vector<T>& Data);
};

template <typename D> class Tensor {
public:
  Tensor(int Rows, int Cols, Generator<D>& G, int N = 1);
  Tensor(float* Data, int Rows, int Cols, int N = 1);
  Tensor(int Rows, int Cols, int N = 1);
  Tensor(Tensor<D>& T);

  int len();
  int size();
  int rows();
  int cols();

  void mmul(Tensor<D>& R, Tensor<D>& O);
  friend Tensor<D>& operator*(Tensor<D>& L, Tensor<D>& R) {
    Tensor<D>* O = new Tensor<D>(L.NRows, R.NCols);
    L.mmul(R, *O);
    return *O;
  };

  Tensor<D> t();
  D& operator[](int Index);
  Tensor<D>& operator+=(Tensor<D>& R);
  Tensor<D> operator+(Tensor<D>& R);
  const vector<D> data();

  ostream& streamOut(ostream& OStream);

  friend ostream& operator<<(ostream& OStream, Tensor<D>& T) {
    return T.streamOut(OStream);
  }

private:
  vector<D>* Data;
  int NCols;
  int NRows;
  int Dim3;
  int Length;
  size_t TSize;
};

template <typename T> class IComponent {
public:
  virtual Tensor<T>& forward(Tensor<T>&) = 0;
  virtual void backward() = 0;
};

template <typename T> class Component : public IComponent<T> {
public:
  Component() {}
  string name();
  // TODO: every component should have a print operator override.
};

template <typename T> class Layer : public Component<T> {
public:
  Layer() {}
  virtual Tensor<T>& w();
  virtual Tensor<T>& b();

  ostream& streamOut(ostream& OStream);

  friend ostream& operator<<(ostream& OStream, Layer<T>& L) {
    return L.streamOut(OStream);
  }

protected:
  Tensor<T>* W;
  Tensor<T>* B;
};

template <typename T> class NeuralNetwork : public Component<T> {
public:
  NeuralNetwork() {}

  Layer<T>& layer(string LayerName);

protected:
  unordered_map<string, Layer<T>*> Layers;
};

template <typename T> class Loss : public Layer<T> {
public:
  Loss() {}
  Tensor<T>& forward(Tensor<T>& A);
  void backward();
};

template <typename T> class LinearLayer : public Layer<T> {
public:
  LinearLayer(int SIn, int SOut, Generator<T>& G);
  LinearLayer(NN::Layer<T> Comp);

  Tensor<T>& forward(Tensor<T>& X);
  void backward();
};

template <typename T> class TanH : public Layer<T> {
public:
  TanH();
  Tensor<T>& forward(Tensor<T>& X);
  void backward();
};

template <typename T> Generator<T>::Generator(int Seed) { srand(Seed); }

// TODO: improve random number generation quality
template <typename T> void Generator<T>::normal(T L, T R, vector<T>& Data) {
  if ((long long)(L - R) - (L - R) != 0) {
    throw invalid_argument(
        "the range must have an integer length, e.g. 1.5 - .5 = 1");
  }
  for (int I = 0; I < Data.size(); I++) {
    Data[I] = L + (rand() % (int)(R - L)) + (rand() / (T)RAND_MAX);
  }
}

template <typename D>
Tensor<D>::Tensor(int Rows, int Cols, Generator<D>& G, int N) {
  NCols = Cols;
  NRows = Rows;
  Dim3 = N;
  Length = Cols * Rows * N;
  TSize = Length * sizeof(D);
  Data = new vector<D>(Length);
  G.normal(-1, 1, *Data);
}

template <typename D>
Tensor<D>::Tensor(float* Data, int Rows, int Cols, int N) {
  NRows = Rows;
  NCols = Cols;
  Dim3 = N;
  Length = Cols * Rows * N;
  TSize = Length * sizeof(D);
  Data = new vector<D>(Data, Data + Length);
}

template <typename D> Tensor<D>::Tensor(int Rows, int Cols, int N) {
  NRows = Rows;
  NCols = Cols;
  Dim3 = N;
  Length = Cols * Rows * N;
  TSize = Length * sizeof(D);
  Data = new vector<D>(Length);
}

template <typename D> Tensor<D>::Tensor(Tensor<D>& T) {
  NRows = T.rows();
  NCols = T.cols();
  Dim3 = T.Dim3;
  Length = T.cols() * T.rows() * T.Dim3;
  TSize = T.Length * sizeof(D);
  Data = new vector<D>(T.data());
}

template <typename T> int Tensor<T>::len() { return Length; }
template <typename T> int Tensor<T>::size() { return TSize; }
template <typename T> int Tensor<T>::rows() { return NRows; }
template <typename T> int Tensor<T>::cols() { return NCols; }

// adapt to handle 3D tensors (later ndim tensors)
template <typename D> void Tensor<D>::mmul(Tensor<D>& RS, Tensor<D>& O) {
  Tensor<D> LS = *this;
  if (LS.cols() != RS.rows())
    throw invalid_argument("Inner dimension of input tensors does not match.");
  if (O.rows() != LS.rows() && O.cols() != RS.cols())
    throw invalid_argument("Provided output tensor is not the correct size.");

  for (int R = 0; R < LS.rows(); R++) {
    for (int C = 0; C < RS.cols(); C++) {
      for (int I = 0; I < LS.cols(); I++) {
        O[R * RS.cols() + C] += LS[R * LS.cols() + I] * RS[C + RS.cols() * I];
      }
    }
  }
}

template <typename D> Tensor<D> Tensor<D>::t() {
  Tensor<D> T = *this;
  Tensor<D> Transpose(NCols, NRows);
  for (int R = 0; R < NRows; R++) {
    for (int C = 0; C < NCols; C++) {
      Transpose[C * NCols + R] = T[C + R * NCols];
    }
  }
  return Transpose;
}

template <typename D> D& Tensor<D>::operator[](int Index) {
  if (Index >= Length) {
    throw invalid_argument("Tensor array index out of bounds.");
  }
  return (*Data)[Index];
}

template <typename D> Tensor<D> Tensor<D>::operator+(Tensor<D>& R) {
  Tensor<D>& LS(*this);
  LS += R;
  return LS;
};

// TODO: potentially implement broadcasting more generally
template <typename D> Tensor<D>& Tensor<D>::operator+=(Tensor<D>& RS) {
  Tensor<D>& LS = *this;
  if (LS.rows() == RS.rows() && LS.cols() == RS.cols()) {
    for (int R = 0; R < LS.rows(); R++) {
      for (int C = 0; C < LS.cols(); C++) {
        LS[R * LS.cols() + C] += RS[R * LS.cols() + C];
      }
    }
  } else if (LS.cols() == RS.cols()) {
    for (int R = 0; R < LS.rows(); R++) {
      for (int C = 0; C < LS.cols(); C++) {
        LS[R * LS.cols() + C] += RS[C];
      }
    }
  } else if (LS.rows() == RS.rows()) {
    for (int R = 0; R < LS.rows(); R++) {
      for (int C = 0; C < LS.cols(); C++) {
        LS[R * LS.cols() + C] += RS[R];
      }
    }
  } else {
    // TODO: improve all error messages to include detailed information
    throw invalid_argument(
        "Dimensions of input tensors do not match & are not broadcastable.");
  }
  return LS;
}

template <typename D> const vector<D> Tensor<D>::data() {
  return vector<D>(*Data);
}

template <typename D> ostream& Tensor<D>::streamOut(ostream& OStream) {
  Tensor<D>& T = *this;
  OStream << "\n";
  for (int I = 0; I < T.len(); I++) {
    OStream << setw(8) << format("{:.4f}", T.data()[I]) << ' ';
    if (I % T.cols() == T.cols() - 1) {
      OStream << "\n";
    }
  }
  return OStream;
}

template <typename T> string Component<T>::name() {
  return typeid(*this).name();
}

template <typename T> Tensor<T>& Layer<T>::w() { return *W; }
template <typename T> Tensor<T>& Layer<T>::b() { return *B; }
template <typename D> ostream& Layer<D>::streamOut(ostream& OStream) {
  Layer<D>& L = *this;
  OStream << "-----------------------" << endl;
  OStream << L.name() << ": forward\n" << endl;
  OStream << "weights: " << L.w() << endl;
  OStream << "bias: " << L.b() << endl;
  OStream << "-----------------------" << endl;

  return OStream;
}

template <typename T> Layer<T>& NeuralNetwork<T>::layer(string LayerName) {
  return *this->layers[LayerName];
}

template <typename T> Tensor<T>& Loss<T>::forward(Tensor<T>& A) {
  cout << this->name() << ": forward" << endl;
  return A;
}

template <typename T> void Loss<T>::backward() {
  cout << this->name() << ": backward" << endl;
};

template <typename T>
LinearLayer<T>::LinearLayer(int SIn, int SOut, Generator<T>& G) {
  this->W = new Tensor<T>(SOut, SIn, G);
  this->B = new Tensor<T>(SOut, 1, G);
}

template <typename T> Tensor<T>& LinearLayer<T>::forward(Tensor<T>& X) {
  Tensor<T>& Logits = this->w() * X;
  Logits += this->b();
  return Logits;
}

template <typename T> void LinearLayer<T>::backward() {
  cout << this->name() << ": backward" << endl;
}

template <typename T> Tensor<T>& TanH<T>::forward(Tensor<T>& X) {
  cout << this->name() << ": forward" << endl;
  return X;
}

template <typename T> void TanH<T>::backward() {
  cout << this->name() << ": backward" << endl;
}

} // namespace NN
