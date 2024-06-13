#include <iostream>
#include <stdlib.h>
#include <typeinfo>
#include <unordered_map>
#include <vector>

using namespace std;

template <typename T> class Tensor {
public:
  Tensor(int rows, int columns, int N = 1) {
    data = new vector<T>(rows * columns * N);
    srand(0);
    for (auto e : *data) {
      e = -1 + (rand() % 2) + (rand() / (float)RAND_MAX);
    }
  }

  friend ostream &operator<<(ostream &os, Tensor &t) {
    os << "\n";
    for (T e : *t.data) {
      os << e << ' ';
    }
    os << "\n";
    return os;
  }

private:
  vector<T> *data;
};

class IComponent {
public:
  virtual void forward() = 0;
  virtual void backward() = 0;
};

class Component : public IComponent {
public:
  Component() {}
  string name() { return typeid(*this).name(); }
};

class NeuralNetwork : public Component {
public:
  NeuralNetwork() {}

protected:
  unordered_map<string, Component *> layers;
};

class Loss : public Component {
public:
  Loss() {}
  void forward() { cout << name() << ": forward" << endl; }
  void backward() { cout << name() << ": backward" << endl; }
};

class LinearLayer : public Component {
public:
  LinearLayer(int sIn, int sOut) {
    w = new Tensor<float>(sIn, sOut);
    b = new Tensor<float>(sOut, 1);
  }

  void forward() {
    cout << "-----------------------" << endl;
    cout << name() << ": forward" << endl;
    cout << "weights: " << *w << endl;
    cout << "bias: " << *b << endl;
    cout << "-----------------------" << endl;
  }
  void backward() { cout << name() << ": backward" << endl; }

private:
  Tensor<float> *w;
  Tensor<float> *b;
};

class Activation : public Component {
public:
  Activation() {}
  void forward() { cout << name() << ": forward" << endl; }
  void backward() { cout << name() << ": backward" << endl; }
};

class SimpleNet : public NeuralNetwork {
public:
  SimpleNet() {
    layers.emplace("l0", new LinearLayer(100, 10));
    layers.emplace("activation", new Activation());
    layers.emplace("loss", new Loss());
  };

  void forward() {
    layers["l0"]->forward();
    layers["activation"]->forward();
    layers["loss"]->forward();
  };

  void backward() {
    layers["loss"]->backward();
    layers["activation"]->backward();
    layers["l0"]->backward();
  };
};

int main() {
  SimpleNet nn;
  nn.forward();
  nn.backward();
}
