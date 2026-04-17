#include "mat.hpp"
#include <cstddef>

int main() {
  std::mt19937 rng{15};
  nn::Matrix_ A(2, 3), B(3, 2);
  A.rand_fill(2, 6, rng);
  B.rand_fill(2, 8, rng);

  A.print();
  B.print();

  nn::Matrix_ C(A.rows(),B.cols());
  nn::Matrix_::matmul(C, A, B);



  C.print("C",3);
}