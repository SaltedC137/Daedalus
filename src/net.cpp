

#include "net.hpp"
#include <cstddef>
#include <vector>

namespace nn {

void Net::alloc(const std::vector<size_t> &architecture) {
  arch = architecture;
  size_t n = arch.size();

  ws.resize(n - 1);
  bs.resize(n - 1);
  as.resize(n);

  as[0] = Mat(1, arch[0]);

  for (size_t i = 1; i < n; ++i) {
    ws[i - 1] = Mat(arch[i - 1], arch[i]);
    bs[i - 1] = Mat(1, arch[i]);
    as[i] = Mat(1, arch[i]);
  }
}

void Net::forward() {
  for (size_t i = 0; i < arch.size() - 1; ++i) {
    Mat::dot(as[i + 1], as[i], ws[i]);
    Mat::sum(as[i + 1], bs[i]);
    Mat::act(as[i + 1], activation);
  }
}

float Net::loss(const Mat &t) {
  assert(input().cols() + output().cols() == t.cols());

  float c = 0.0f;
  size_t input_cols = input().cols();

  for (size_t i = 0; i < t.rows(); ++i) {
    for (size_t j = 0; j < input_cols; ++j) {
      as[0](0, j) = t(i, j);
    }
    forward();
    for (size_t j = 0; j < output().cols(); ++j) {
      float d = output()(0, j) - t(i, input_cols + j);
      c += d * d;
    }
  }
  return c / t.rows();
}

// shit func
Net Net::backprop(const Mat &t) {
  Net g;
  g.alloc(arch);

  for (auto &w : g.ws)
    w.fill(0.0f);
  for (auto &b : g.bs)
    b.fill(0.0f);

  size_t n = t.rows();
  size_t input_cols = input().cols();

  for (size_t sample = 0; sample < n; ++sample) {
    for (size_t j = 0; j < input_cols; ++j) {
      as[0](0, j) = t(sample, j);
    }
    forward();

    std::vector<Mat> das(arch.size());
    for (size_t k = 0; k < arch.size(); ++k) {
      das[k] = Mat(1, arch[k]);
    }
    for (size_t j = 0; j < output().cols(); ++j) {
      float d = output()(0, j) - t(sample, input_cols + j);
      das[arch.size() - 1](0, j) = 2.0f * d;
    }

    for (int l = (int)arch.size() - 1; l > 0; --l) {

      for (size_t j = 0; j < arch[l]; ++j) {
        float a = as[l](0, j);
        float da = das[l](0, j);
        float qa = dactf(a, activation);

        g.bs[l - 1](0, j) += da * qa;

        for (size_t k = 0; k < arch[l - 1]; ++k) {
          float pa = as[l - 1](0, k);
          g.ws[l - 1](k, j) += da * qa * pa;
          das[l - 1](0, k) += da * qa * ws[l - 1](k, j);
        }
      }
    }
  }
  for (size_t i = 0; i < arch.size() - 1; ++i) {
    for (size_t j = 0; j < g.ws[i].rows(); ++j) {
      for (size_t k = 0; k < g.ws[i].cols(); ++k) {
        g.ws[i](j, k) /= n;
      }
    }
    for (size_t k = 0; k < g.bs[i].cols(); ++k) {
      g.bs[i](0, k) /= n;
    }
  }
  return g;
}

void Net::learn(const Net &g, float rate) {
  for (size_t i = 0; i < arch.size() - 1; ++i) {
    for (size_t j = 0; j < ws[i].rows(); ++j) {
      for (size_t k = 0; k < ws[i].cols(); ++k) {
        ws[i](j, k) -= rate * g.ws[i](j, k);
      }
    }
    for (size_t k = 0; k < bs[i].cols(); ++k) {
      bs[i](0, k) -= rate * g.bs[i](0, k);
    }
  }
}

} // namespace nn