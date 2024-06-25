#include <cstddef>
#include <cstdlib>
#include <format>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdlib.h>

using namespace std;

// TODO: compile with flag to turn off exceptions
#define INVALID_ARGUMENT(IDX, LEN)                                             \
  cout << "Tensor array index out of bounds. IDX: " << IDX << " LEN: " << LEN  \
       << endl;                                                                \
  exit(1);

#define HEAP_MEMORY_SIZE 2 << 16
#define STACK_MEMORY_SIZE 2 << 8
#define SEED 0

// TODO: could we line this up with L1 line sizes?
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

template <typename D> struct Tensor {
  size_t Rows;
  size_t Cols;
  size_t Len;
  D* Data;

  D& operator[](int Index) {
    if (Index >= Len) {
      stringstream SS;
      INVALID_ARGUMENT(Index, Len);
    }
    return *(Data + Index);
  };

  friend ostream& operator<<(ostream& OStream, Tensor<D>& O) {
    OStream << "\n";
    OStream << &O.Data << ' ' << O.Data << ' ' << &O.Rows << ' ' << &O.Cols
            << ' ' << O.Len << endl;
    for (int I = 0; I < O.Len; I++) {
      OStream << setw(8) << format("{:.4f}", O[I]) << ' ';
      if (I % O.Cols == O.Cols - 1) {
        OStream << "\n";
      }
    }
    return OStream;
  };
};

// TODO: improve random number generation quality
template <typename D> Tensor<D>& normal(Tensor<D>& T) {
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

template <typename D> Tensor<D>& ones(Tensor<D>& T) {
  for (int I = 0; I < T.Len; I++) {
    T[I] = 1.0f;
  }
  return T;
}

template <typename D> Tensor<D>& zeros(Tensor<D>& T) {
  for (int I = 0; I < T.Len; I++) {
    T[I] = 0.0f;
  }
  return T;
}

template <typename D>
Tensor<D>* createTensor(size_t R, size_t C, Memory& M,
                        Tensor<D>& (*Generator)(Tensor<D>&)) {
  Tensor<D>* TP = new (M.SP) Tensor<D>{};
  M.SP += sizeof(Tensor<D>);
  Tensor<D>& T = *TP;

  T.Rows = R;
  T.Cols = C;
  T.Len = R * C;

  T.Data = new (M.TP) float[T.Len];
  M.TP += sizeof(D) * T.Len;

  T = Generator(T);
  return TP;
};

int main() {
  int Seed = SEED;
  size_t HeapMemorySize = HEAP_MEMORY_SIZE;
  size_t StackMemorySize = STACK_MEMORY_SIZE;
  Memory ProgramMemory;
  char StackBuffer[STACK_MEMORY_SIZE];

  srand(Seed);
  ProgramMemory =
      initMemory(ProgramMemory, HeapMemorySize, StackMemorySize, StackBuffer);

  Tensor<float>* X;
  Tensor<float>* Dx;

  X = createTensor<float>(10, 2, ProgramMemory, normal);
  Dx = createTensor<float>(10, 2, ProgramMemory, ones);
  cout << *X << *Dx;
}
