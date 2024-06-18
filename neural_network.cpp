#include "neural_network.hpp"

using namespace std;
using namespace NN;

template <typename T> class SimpleNet : public NeuralNetwork<T> {
public:
  SimpleNet(Generator<T>& G) {
    this->layers.emplace("l0", new LinearLayer<T>(10, 2, G));
    this->layers.emplace("activation", new TanH<T>());
    this->layers.emplace("loss", new Loss<T>());
  };

  Tensor<T>& forward(Tensor<T>& X) {
    Layer<T>& L0 = this->Layer("l0");
    Tensor<T>& Logits = L0.forward(X);
    Tensor<T>& Activations = this->layers("activation")->forward(Logits);
    Tensor<T>& Loss = this->layers("loss")->forward(Activations);

    cout << "-----------------------" << endl;
    cout << this->name() << ": forward\n" << endl;
    cout << "input: " << X << endl;
    cout << "weights: " << L0.W() << endl;
    cout << "bias: " << L0.B() << endl;
    cout << "logits: " << Logits << endl;
    cout << "activations: " << Activations << endl;
    cout << "loss: " << Loss << endl;
    cout << "-----------------------" << endl;

    return Loss;
  };

  void backward() {
    this->layers("loss")->backward();
    this->layers("activation")->backward();
    this->layers("l0")->backward();
  };
};
