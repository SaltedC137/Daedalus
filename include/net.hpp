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


  // adam optimizer

  std::vector<Mat> m_ws, m_bs;
  std::vector<Mat> v_ws, v_bs;
  size_t adam_t = 0;


  static constexpr float ADAM_BETA1 = 0.9f;
  static constexpr float ADAM_BETA2 = 0.999f;
  static constexpr float ADAM_EPS = 1e-8f;

  
  Net() : activation(Act::SIGMOID) {}

  void alloc(const std::vector<size_t> &architecture);

  Mat &input() { return as[0]; }

  Mat &output() { return as[arch.size() - 1]; }

  void forward(std::vector<Mat> &as_local) const;

  float loss(const Mat &t);

  Net backprop(const Mat &t);

  void learn(const Net &g, float rate);

  void init_adam(); 
  void adam_learn(const Net &g, float rate);

};

} // namespace nn
 