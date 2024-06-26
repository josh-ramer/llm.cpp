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

#define HEAP_MEMORY_SIZE 2 << 12
#define STACK_MEMORY_SIZE 2 << 8
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
  void (*Backward)(Tensor<D>&) = NULL;
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

  static void mmBackward(Tensor<D>& T) {
    Tensor<D>& L = *T.Parents[0];
    Tensor<D>& R = *T.Parents[1];
    Tensor<D>& O = *T.Children[0];
    *L.Grad += *R.Grad * *O.Grad;
    *R.Grad += *L.Grad * *O.Grad;

    for (auto P : O.Parents) {
      if (P->Backward != NULL)
        P->Backward(T);
    }
  }

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
      // TODO: improve all error messages to include detailed information
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

  friend Tensor<D>& operator*(Tensor<D>& L, Tensor<D>& R) {
    if (L.Cols != R.Rows) {
      stringstream SS;
      SS << "Inner dimension of input tensors does not match. (" << L.Rows
         << "," << L.Cols << ") (" << R.Rows << "," << R.Cols << ")" << endl;
      INVALID_ARGUMENT(SS.str());
    }

    Tensor<D> Gradient =
        createTensor(L.Rows, R.Cols, vector<Tensor<D>*>(), ones);
    vector<Tensor<D>*> Parents = {&L, &R};
    Tensor<D>& Out =
        createTensor(L.Rows, R.Cols, Parents, Tensor<D>::mmBackward, &Gradient);
    for (int RW = 0; RW < L.Rows; RW++) {
      for (int C = 0; C < R.Cols; C++) {
        for (int I = 0; I < L.Cols; I++) {
          Out[RW * R.Cols + C] += L[RW * L.Cols + I] * R[C + R.Cols * I];
        }
      }
    }
    return Out;
  };

  static Tensor<D>& createTensor(size_t R, size_t C,
                                 Tensor<D>& (*Generator)(Tensor<D>&) = NULL,
                                 Tensor<D>* Gradient = NULL) {
    Memory& M = ProgramMemory;
    Tensor<D>* TP = new (M.SP) Tensor<D>{};
    M.SP += sizeof(*TP);
    Tensor<D>& T = *TP;

    T.Rows = R;
    T.Cols = C;
    T.Len = R * C;

    T.Data = new (M.TP) float[T.Len];
    M.TP += sizeof(D) * T.Len;

    T.Grad = Gradient;
    if (Generator != NULL) {
      T = Generator(T);
    }
    return T;
  };

  static Tensor<D>& createTensor(size_t R, size_t C, vector<Tensor<D>*> Parents,
                                 void (*Backward)(Tensor<D>&) = NULL,
                                 Tensor<D>* Gradient = NULL,
                                 Tensor<D>& (*Generator)(Tensor<D>&) = NULL) {
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

    T.Grad = Gradient;
    T.Parents = Parents;
    T.Backward = Backward;

    if (Generator != NULL) {
      T = Generator(T);
    }

    return T;
  };

  static Tensor<D>& createTensor(size_t R, size_t C, vector<Tensor<D>*> Parents,
                                 Tensor<D>& (*Generator)(Tensor<D>&) = NULL,
                                 void (*Backward)(Tensor<D>&) = NULL,
                                 Tensor<D>* Gradient = NULL) {
    // TODO: handle allocation of too much memory
    Memory& M = ProgramMemory;
    Tensor<D>* TP = new (M.SP) Tensor<D>{};
    M.SP += sizeof(TP);
    Tensor<D>& T = *TP;

    T.Rows = R;
    T.Cols = C;
    T.Len = R * C;

    size_t DataSize = sizeof(D) * T.Len;
    T.Data = new (M.TP) float[T.Len];
    M.TP += DataSize;

    T.Grad = Gradient;
    T.Parents = Parents;
    T.Backward = Backward;

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

  Tensor<float> Dx;
  Tensor<float> Dw;
  Tensor<float> DLogits;

  X = Tensor<float>::createTensor(10, 2, Tensor<float>::normal);
  W = Tensor<float>::createTensor(10, 10, Tensor<float>::normal);
  Dx = Tensor<float>::createTensor(10, 2, Tensor<float>::ones);
  Dw = Tensor<float>::createTensor(10, 10, Tensor<float>::ones);

  cout << X << Dx << W << Dw;
  Logits = W * X;
  // cout << Logits << *Logits.Grad;

  cleanMemory(ProgramMemory);
}
