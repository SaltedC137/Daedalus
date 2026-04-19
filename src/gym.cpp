#include "set.hpp"
#include <iostream>
#include <vector>

int main() {
  std::mt19937 rng{15};
  nn::Mat A(2, 3), B(3, 2);
  A.rand_fill(2, 6, rng);
  B.rand_fill(2, 8, rng);

  A.print();
  B.print();

  nn::Mat C(A.rows(), B.cols());
  nn::Mat::matmul(C, A, B);

  C.print("C", 3);

  std::vector<nn::Mat> mnist_img =
      nn::load_mnist_img("assets/data/t10k-images.idx3-ubyte");

  std::vector<int> mnist_lab =
      nn ::load_mnist_lab("assets/data/t10k-labels.idx1-ubyte");

  if (!mnist_lab.empty()) {
    std::cout << "Success load it. \n";
    std::cout << "size :" << mnist_img.size() << std::endl;
  } else {
    std::cout << "load fail!!";
  }
}
