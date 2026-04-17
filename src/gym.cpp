#include "mat.hpp"
#include <iostream>

int main() {
  std::mt19937 rng{15};
  nn::Mat A(2, 3), B(3, 2);
  A.rand_fill(2, 6, rng);
  B.rand_fill(2, 8, rng);

  A.print();
  B.print();

  nn::Mat C(A.rows(),B.cols());
  nn::Mat::matmul(C, A, B);

  C.print("C",3);
}