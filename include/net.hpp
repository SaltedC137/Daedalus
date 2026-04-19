#pragma once
#include "mat.hpp"
#include <cstddef>
#include <vector>

namespace nn {

class Net {

public:
  std::vector<size_t> arch;
  std::vector<Mat> ws; // weight
  std::vector<Mat> bs; // bias
  std::vector<Mat> as; // activation
  Act activation;

  Net() : activation(Act::SIGMOID) {}

  void alloc(const std::vector<size_t> &architecture);

  Mat &input() { return as[0]; }

  Mat &output() { return as[arch.size() - 1]; }

  void forward(std::vector<Mat> &as_local) const;

  float loss(const Mat &t);

  Net backprop(const Mat &t);

  void learn(const Net &g, float rate);
};

} // namespace nn
