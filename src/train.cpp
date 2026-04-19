#include "mat.hpp"
#include "net.hpp"
#include "set.hpp"
#include <cstddef>
#include <iostream>
#include <random>
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

  if (mnist_img.empty() || mnist_lab.empty() ||
      mnist_img.size() != mnist_lab.size()) {
    std::cout << " Data loading failed!!!!!" << std::endl;
    return 1;
  } else {
    std::cout << " Data loading success!!!!!" << std::endl;
  }

  size_t count = mnist_img.size();
  size_t input_size = mnist_img[0].cols();
  size_t output_size = 10;

  nn::Net net;
  net.alloc({input_size, 128, 64, output_size});
  net.activation = Act::RELU;

  nn::Mat target(count, input_size + output_size);
  for (size_t i = 0; i < count; i++) {
    for (size_t j = 0; j < input_size; j++) {
      target(i, j) = mnist_img[i](0, j);
    }
    for (size_t j = 0; j < output_size; j++) {
      target(i, input_size + j) = (j == mnist_lab[i]) ? 1.0f : 0.0f;
    }
  }

  float learn_rate = 0.001f;
  size_t epochs = 10;
  size_t batch_size = 32;

  for (size_t epoch = 0; epoch < epochs; epoch++) {
    target.shuffle_rows(rng);

    float total_loss = 0.0f;
    size_t num_batches = count / batch_size;

    for (size_t batch = 0; batch < num_batches; batch++) {
      nn::Mat batch_target(batch_size, input_size + output_size);
      for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < input_size + output_size; j++) {
          batch_target(i, j) = target(batch * batch_size + i, j);
        }
      }

      float loss = net.loss(batch_target);
      total_loss += loss;
      nn::Net gradients = net.backprop(batch_target);

      net.learn(gradients, learn_rate);
    }

    float avg_loss = total_loss / num_batches;
    std::cout << "  Epoch " << epoch + 1 << "/" << epochs << std::endl;
    std::cout << "  Loss " << avg_loss << std::endl;
  }
  std::cout << "Train Finish" << std::endl;
  return 0;
}
