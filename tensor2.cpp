#include <cstddef>
#include <cstdlib>
#include <format>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdlib.h>

using namespace std;

// TODO: compile with flag to turn off exceptions
#define INVALID_ARGUMENT(MSG)                                                  \
  cout << MSG << endl;                                                         \
  exit(1);

#define HEAP_MEMORY_SIZE 2 << 16
#define STACK_MEMORY_SIZE 2 << 10
#define SEED 0

// TODO: could we line this up with L1 line sizes?
// To look at the program memory map compile with -Wl,-map,output.map
// Why do this? We want our fast data structure to have its pointers on the
// stack & it's potentially large tensors on the heap. We want those tensors to
// be in contiguous memory & preferably in the L1 cache which will save us 100s
// of cycles per cache miss.
struct Memory {
  size_t HeapLen; // sizeof preallocated heap buffer
  char* T;        // beginning of memory holding tensors
  char* TP;       // beginning of free heap memory

  // The only way this works is if the Memory object is in main fn scope, i.e.
  // as soon as it goes out of scope, our stack pointers are destroyed.
  size_t StackLen; // sizeof preallocated stack buffer
  char* S;         // beginning of stack memory holding pointers to tensors
  char* SP;        // beginning of free stack memory
};

Memory& initMemory(Memory& M, size_t HeapMemorySize, size_t StackMemorySize,
                   char* StackBuffer) {
  M.T = (char*)malloc(HeapMemorySize);
  M.TP = M.T;
  M.HeapLen = HeapMemorySize;

  M.S = StackBuffer;
  M.SP = M.S;
  M.StackLen = StackMemorySize;
  return M;
};

template <typename D> struct Tensor;
void cleanMemory(Memory& M) { free(M.T); }

static Memory ProgramMemory;
template <typename D> struct Tensor {
  size_t Rows;
  size_t Cols;
  size_t Len;
  D* Data;
  Tensor<D>* Grad = NULL;
  void (*BackProp)(Tensor<D>&) = NULL;
  vector<Tensor<D>*> Parents;
  vector<Tensor<D>*> Children;

  D& operator[](int Index) {
    if (Index >= Len) {
      stringstream SS;
      SS << "Tensor array index out of bounds. IDX: " << Index
         << " LEN: " << Len << endl;
      INVALID_ARGUMENT(SS.str());
    }
    return *(Data + Index);
  };

  friend ostream& operator<<(ostream& OStream, Tensor<D>& O) {
    OStream << "\n";
    OStream << "RowsS       ColsS       LenS        DataS       DataH" << endl;
    OStream << &O.Rows << ' ' << &O.Cols << ' ' << &O.Len << ' ' << &O.Data
            << ' ' << O.Data << endl;
    for (int I = 0; I < O.Len; I++) {
      OStream << setw(8) << format("{:.4f}", O[I]) << ' ';
      if (I % O.Cols == O.Cols - 1) {
        OStream << "\n";
      }
    }
    return OStream;
  };

  void backward() { this->BackProp(*this); }

  Tensor<D>& t() const {
    Tensor<D> I = *this;
    Tensor<D>& Trns = createTensor(I.Cols, I.Rows);
    for (int R = 0; R < I.Rows; R++) {
      for (int C = 0; C < I.Cols; C++) {
        Trns[C * I.Rows + R] = I[C + R * I.Cols];
      }
    }
    return Trns;
  }

  static void mmBackward(Tensor<D>& Logits) {
    Tensor<D>& L = *Logits.Parents[0];
    Tensor<D>& R = *Logits.Parents[1];

    *L.Grad += Logits.Grad->mm(R.t());
    *R.Grad += L.t().mm(*Logits.Grad);

    for (auto P : Logits.Parents) {
      if (P->BackProp != NULL)
        P->BackProp(*P);
    }
  }

  static void tanHBackward(Tensor<D>& A) {
    Tensor<D>& One = createTensor(A.Rows, A.Cols, ones);
    Tensor<D>& Diff = One - A.squares();
    Tensor<D>& Logits = *A.Parents[0];
    *Logits.Grad += Diff * *A.Grad;
    // TODO: this will break down-> use reverse topological order
    for (auto P : A.Parents) {
      if (P->BackProp != NULL)
        P->BackProp(*P);
    }
  };

  Tensor<D>& tanH() {
    Tensor<D>& I = *this;
    vector<Tensor<D>*> Parents = {&I};
    Tensor<D>& O = createTensor(I.Rows, I.Cols, Parents, tanHBackward, true);

    for (int R = 0; R < I.Rows; R++) {
      for (int C = 0; C < I.Cols; C++) {
        O[R * I.Cols + C] = tanh(I[R * I.Cols + C]);
      }
    }
    return O;
  };

  Tensor<D>& squares() {
    Tensor<D>& I = *this;
    vector<Tensor<D>*> Parents = {&I};
    // TODO: squares backprop?
    Tensor<D>& S = createTensor(I.Rows, I.Cols, Parents, nullptr, false);
    for (int R = 0; R < I.Rows; R++) {
      for (int C = 0; C < I.Cols; C++) {
        S[R * I.Cols + C] += I[R * I.Cols + C] * I[R * I.Cols + C];
      }
    }
    return S;
  }

  friend Tensor<D>& operator*(D Data, Tensor<D>& RS) { return RS * Data; };
  friend Tensor<D>& operator*(Tensor<D>& LS, D Data) {
    Tensor<D>& O = createTensor(LS.Rows, LS.Cols);
    for (int R = 0; R < LS.Rows; R++) {
      for (int C = 0; C < LS.Cols; C++) {
        O[R * LS.Cols + C] = LS[R * LS.Cols + C] * Data;
      }
    }
    return O;
  };

  friend Tensor<D>& operator*(Tensor<D>& LS, Tensor<D>& RS) {
    // TODO: elementwize multiplication backprop?
    // TODO: broadcasting?
    if (LS.Rows != RS.Rows || LS.Cols != RS.Cols) {
      stringstream SS;
      SS << "Dimensions of input tensors do not match: (" << LS.Rows << ", "
         << LS.Cols << ") RS (" << RS.Rows << ", " << RS.Cols << ")";
      INVALID_ARGUMENT(SS.str());
    }
    Tensor<D>& O = createTensor(LS.Rows, LS.Cols);
    for (int R = 0; R < LS.Rows; R++) {
      for (int C = 0; C < LS.Cols; C++) {
        O[R * LS.Cols + C] += LS[R * LS.Cols + C] * RS[R * LS.Cols + C];
      }
    }
    return O;
  };

  Tensor<D>& operator/(D RS) {
    Tensor<D>& LS = *this;
    Tensor<D>& O = createTensor(LS.Rows, LS.Cols);
    for (int R = 0; R < LS.Rows; R++) {
      for (int C = 0; C < LS.Cols; C++) {
        O[R * LS.Cols + C] = LS[R * LS.Cols + C] / RS;
      }
    }
    return O;
  };

  Tensor<D>& operator/(Tensor<D>& RS) {
    Tensor<D>& LS = *this;
    Tensor<D>& O = createTensor(LS.Rows, RS.Rows);

    // TODO: adapt to handle 3D tensors (later ndim tensors)
    // tensor to be broadcast must be on the RHS, should that change?
    if (RS.Cols == 1 && RS.Cols == 1) {
      for (int R = 0; R < LS.Rows; R++) {
        for (int C = 0; C < LS.Cols; C++) {
          O[R * LS.Cols + C] = LS[R * LS.Cols + C] / RS[0];
        }
      }
    } else if (LS.Rows == RS.Rows && LS.Cols == RS.Cols) {
      for (int R = 0; R < LS.Rows; R++) {
        for (int C = 0; C < LS.Cols; C++) {
          O[R * LS.Cols + C] = LS[R * LS.Cols + C] / RS[R * LS.Cols + C];
        }
      }
    } else if (LS.Rows == RS.Rows && RS.Cols == 1) {
      for (int R = 0; R < LS.Rows; R++) {
        for (int C = 0; C < LS.Cols; C++) {
          O[R * LS.Cols + C] = LS[R * LS.Cols + C] / RS[R];
        }
      }
    } else if (LS.Cols == RS.Cols && RS.Rows == 1) {
      for (int R = 0; R < LS.Rows; R++) {
        for (int C = 0; C < LS.Cols; C++) {
          O[R * LS.Cols + C] = LS[R * LS.Cols + C] / RS[C];
        }
      }
    } else {
      INVALID_ARGUMENT("The dimension to be broadcast must be of length 1.");
    }

    return O;
  };

  Tensor<D>& sum() {
    Tensor<D>& T = *this;
    D S = 0;
    for (int R = 0; R < T.Rows; R++) {
      for (int C = 0; C < T.Cols; C++) {
        S += T[R * T.Cols + C];
      }
    }
    return createTensor(S);
  };

  static void lossBackward(Tensor<D>& L) {
    L.Grad = &createTensor(L.Rows, L.Cols, ones);
    Tensor<D>& A = *L.Parents[0];
    for (auto P : L.Parents) {
      if (P->BackProp != NULL)
        P->BackProp(*P);
    }
  }

  Tensor<D>& mse(Tensor<D>& Targets) {
    Tensor<D>& A = *this;
    Tensor<D>& Diff = (Targets - A) / A.Len;
    Tensor<D>& L = Diff.squares().sum();
    *A.Grad -= 2.f * Diff;
    L.Parents.push_back(this);
    L.BackProp = lossBackward;
    return L;
  };

  // tensor to be broadcast must be on the RHS, should that change?
  friend Tensor<D>& operator-=(Tensor<D>& LS, Tensor<D>& RS) {
    if (LS.Rows == RS.Rows && LS.Cols == RS.Cols) {
      for (int R = 0; R < LS.Rows; R++) {
        for (int C = 0; C < LS.Cols; C++) {
          LS[R * LS.Cols + C] -= RS[R * LS.Cols + C];
        }
      }
    } else if (LS.Cols == RS.Cols && RS.Rows == 1) {
      for (int R = 0; R < LS.Rows; R++) {
        for (int C = 0; C < LS.Cols; C++) {
          LS[R * LS.Cols + C] -= RS[C];
        }
      }
    } else if (LS.Rows == RS.Rows && RS.Cols == 1) {
      for (int R = 0; R < LS.Rows; R++) {
        for (int C = 0; C < LS.Cols; C++) {
          LS[R * LS.Cols + C] -= RS[R];
        }
      }
    }
    // TODO: only possible for operator-, not operator-= -> rethink - & -=
    // operator overloads
    //  else if (LS.cols() == RS.cols() && LS.rows() == 1) {
    //    for (int R = 0; R < RS.rows(); R++) {
    //      for (int C = 0; C < LS.cols(); C++) {
    //        LS[C] -= RS[R * LS.cols() + C];
    //      }
    //    }
    //  } else if (LS.rows() == RS.rows() && LS.cols() == 1) {
    //    for (int R = 0; R < LS.rows(); R++) {
    //      for (int C = 0; C < RS.cols(); C++) {
    //        LS[R] -= RS[R * LS.cols() + C];
    //      }
    //    }
    //  }
    else {
      stringstream SS;
      SS << "Dimensions of input tensors do not match & are "
            "not broadcastable. The "
            "dimension to be broadcast must be of length 1. LS ("
         << LS.Rows << ", " << LS.Cols << ") RS (" << RS.Rows << ", " << RS.Cols
         << ")";
      throw invalid_argument(SS.str());
    }
    return LS;
  }

  friend Tensor<D>& operator-(Tensor<D>& LS, Tensor<D>& RS) {
    vector<Tensor<D>*> Parents = {&LS, &RS};
    // TODO: subtraction backprop?
    Tensor<D>& O = createTensor(LS);
    O -= RS;
    return O;
  };

  // tensor to be broadcast must be on the RHS, should that change?
  Tensor<D>& operator+=(Tensor<D>& RS) {
    Tensor<D>& LS = *this;
    if (LS.Rows == RS.Rows && LS.Cols == RS.Cols) {
      for (int R = 0; R < LS.Rows; R++) {
        for (int C = 0; C < LS.Cols; C++) {
          LS[R * LS.Cols + C] += RS[R * LS.Cols + C];
        }
      }
    } else if (LS.Cols == RS.Cols && RS.Rows == 1) {
      for (int R = 0; R < LS.Rows; R++) {
        for (int C = 0; C < LS.Cols; C++) {
          LS[R * LS.Cols + C] += RS[C];
        }
      }
    } else if (LS.Rows == RS.Rows && RS.Cols == 1) {
      for (int R = 0; R < LS.Rows; R++) {
        for (int C = 0; C < LS.Cols; C++) {
          LS[R * LS.Cols + C] += RS[R];
        }
      }
    } else {
      stringstream SS;
      SS << "Dimensions of input tensors do not match & are "
            "not broadcastable. The "
            "dimension to be broadcast must be of length 1. LS ("
         << LS.Rows << ", " << LS.Cols << ") RS (" << RS.Rows << ", " << RS.Cols
         << ")";
      INVALID_ARGUMENT(SS.str());
    }
    return LS;
  }

  Tensor<D>& mm(Tensor<D>& R) {
    Tensor<D>& L = *this;
    if (L.Cols != R.Rows) {
      stringstream SS;
      SS << "Inner dimension of input tensors does not match. (" << L.Rows
         << "," << L.Cols << ") (" << R.Rows << "," << R.Cols << ")" << endl;
      INVALID_ARGUMENT(SS.str());
    }

    vector<Tensor<D>*> Parents = {&L, &R};
    Tensor<D>& O =
        createTensor(L.Rows, R.Cols, Parents, Tensor<D>::mmBackward, true);
    for (int RW = 0; RW < L.Rows; RW++) {
      for (int C = 0; C < R.Cols; C++) {
        for (int I = 0; I < L.Cols; I++) {
          O[RW * R.Cols + C] += L[RW * L.Cols + I] * R[C + R.Cols * I];
        }
      }
    }
    return O;
  };

  static Tensor<D>& createTensor(D Data) {
    Tensor<D>& T = createTensor(1, 1);
    T[0] = Data;
    return T;
  };

  static Tensor<D>& createTensor(Tensor<D>& C) {
    Memory& M = ProgramMemory;
    Tensor<D>* TP = new (M.SP) Tensor<D>{};
    M.SP += sizeof(*TP);
    Tensor<D>& T = *TP;

    T.Rows = C.Rows;
    T.Cols = C.Cols;
    T.Len = C.Rows * C.Cols;

    T.Data = new (M.TP) float[T.Len];
    M.TP += sizeof(D) * T.Len;

    for (int I = 0; I < C.Len; I++) {
      T[I] = C[I];
    }

    if (C.Grad != NULL) {
      T.Grad = &createTensor(*C.Grad);
    } else {
      T.Grad = NULL;
    }

    if (C.BackProp != NULL) {
      T.BackProp = C.BackProp;
    }
    return T;
  };

  static Tensor<D>& createTensor(size_t R, size_t C,
                                 Tensor<D>& (*Generator)(Tensor<D>&) = NULL,
                                 bool Gradient = false) {
    Memory& M = ProgramMemory;
    Tensor<D>* TP = new (M.SP) Tensor<D>{};
    M.SP += sizeof(*TP);
    Tensor<D>& T = *TP;

    T.Rows = R;
    T.Cols = C;
    T.Len = R * C;

    T.Data = new (M.TP) float[T.Len];
    M.TP += sizeof(D) * T.Len;

    if (Gradient) {
      T.Grad = &createTensor(R, C, vector<Tensor<D>*>(), zeros);
    } else {
      T.Grad = NULL;
    }

    if (Generator != NULL) {
      T = Generator(T);
    }
    return T;
  };

  static Tensor<D>& createTensor(size_t R, size_t C, vector<Tensor<D>*> Parents,
                                 void (*BackProp)(Tensor<D>&) = NULL,
                                 Tensor<D>& (*Generator)(Tensor<D>&) = NULL,
                                 bool Gradient = false) {
    // TODO: handle allocation of too much memory
    Memory& M = ProgramMemory;
    Tensor<D>* TP = new (M.SP) Tensor<D>{};
    M.SP += sizeof(*TP);
    Tensor<D>& T = *TP;

    T.Rows = R;
    T.Cols = C;
    T.Len = R * C;

    size_t DataSize = sizeof(D) * T.Len;
    T.Data = new (M.TP) float[T.Len];
    M.TP += DataSize;

    T.Parents = Parents;
    T.BackProp = BackProp;

    if (Gradient) {
      T.Grad = &createTensor(R, C, vector<Tensor<D>*>(), zeros);
    } else {
      T.Grad = NULL;
    }

    if (Generator != NULL) {
      T = Generator(T);
    }

    return T;
  };

  static Tensor<D>& createTensor(size_t R, size_t C, vector<Tensor<D>*> Parents,
                                 void (*BackProp)(Tensor<D>&) = NULL,
                                 bool Gradient = false) {
    // TODO: handle allocation of too much memory
    Memory& M = ProgramMemory;
    Tensor<D>* TP = new (M.SP) Tensor<D>{};
    M.SP += sizeof(*TP);
    Tensor<D>& T = *TP;

    T.Rows = R;
    T.Cols = C;
    T.Len = R * C;

    size_t DataSize = sizeof(D) * T.Len;
    T.Data = new (M.TP) float[T.Len];
    M.TP += DataSize;

    T.Parents = Parents;
    T.BackProp = BackProp;

    if (Gradient) {
      T.Grad = &createTensor(R, C, vector<Tensor<D>*>(), zeros);
    } else {
      T.Grad = NULL;
    }

    return T;
  };

  static Tensor<D>& createTensor(size_t R, size_t C, vector<Tensor<D>*> Parents,
                                 Tensor<D>& (*Generator)(Tensor<D>&) = NULL,
                                 void (*BackProp)(Tensor<D>&) = NULL,
                                 bool Gradient = false) {
    // TODO: handle allocation of too much memory
    Memory& M = ProgramMemory;
    Tensor<D>* TP = new (M.SP) Tensor<D>{};
    M.SP += sizeof(*TP);
    Tensor<D>& T = *TP;

    T.Rows = R;
    T.Cols = C;
    T.Len = R * C;

    size_t DataSize = sizeof(D) * T.Len;
    T.Data = new (M.TP) float[T.Len];
    M.TP += DataSize;

    T.Parents = Parents;
    T.BackProp = BackProp;

    if (Gradient) {
      T.Grad = &createTensor(R, C, vector<Tensor<D>*>(), zeros);
    } else {
      T.Grad = NULL;
    }
    if (Generator != NULL) {
      T = Generator(T);
    }

    return T;
  };

  // TODO: improve random number generation quality
  static Tensor<D>& normal(Tensor<D>& T) {
    float L = -1.0f;
    float R = 1.0f;
    if ((long long)(L - R) - (L - R) != 0) {
      throw invalid_argument(
          "the range must have an integer length, e.g. 1.5 - .5 = 1");
    }
    for (int I = 0; I < T.Len; I++) {
      T[I] = L + (rand() % (int)(R - L)) + (rand() / (D)RAND_MAX);
    }

    return T;
  }

  static Tensor<D>& ones(Tensor<D>& T) {
    for (int I = 0; I < T.Len; I++) {
      T[I] = 1.0f;
    }
    return T;
  }

  static Tensor<D>& zeros(Tensor<D>& T) {
    for (int I = 0; I < T.Len; I++) {
      T[I] = 0.0f;
    }
    return T;
  }
};

int main() {
  int Seed = SEED;
  size_t HeapMemorySize = HEAP_MEMORY_SIZE;
  size_t StackMemorySize = STACK_MEMORY_SIZE;
  char StackBuffer[STACK_MEMORY_SIZE];

  srand(Seed);
  ProgramMemory =
      initMemory(ProgramMemory, HeapMemorySize, StackMemorySize, StackBuffer);

  Tensor<float> X;
  Tensor<float> W;
  Tensor<float> Logits;
  Tensor<float> A;
  Tensor<float> Targets;
  Tensor<float> Loss;

  X = Tensor<float>::createTensor(10, 2, Tensor<float>::normal, true);
  W = Tensor<float>::createTensor(2, 10, Tensor<float>::normal, true);
  Targets = Tensor<float>::createTensor(10, 10, Tensor<float>::ones, false);

  Logits = X.mm(W);
  A = Logits.tanH();
  Loss = A.mse(Targets);

  cout << "X:" << X << "\nW:" << W;
  cout << "\nLogits:" << Logits;
  cout << "\nActivations:" << A;

  cout << "\nLoss:" << Loss << endl;
  Loss.backward();

  cout << "\nLoss.Grad:" << *Loss.Grad;
  cout << "\nActivations.Grad:" << *A.Grad;
  cout << "\nLogits.Grad:" << *Logits.Grad;
  cout << "\nX.Grad:" << *X.Grad << "\nW.Grad:" << *W.Grad;

  cleanMemory(ProgramMemory);
}
