
#include "net.hpp"
#include "act.hpp"
#include "layer.hpp"
#include "mat.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <iostream>
#include <omp.h>
#include <ostream>
#include <vector>

namespace nn {

#if 1
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
#endif

void Net::forward(std::vector<Mat> &as_local) const {
  for (size_t i = 0; i < arch.size() - 1; ++i) {
    Mat::dot(as_local[i + 1], as_local[i], ws[i]);
    Mat::sum(as_local[i + 1], bs[i]);

    Act layer_act = layer[i].activation;
    if (layer_act == Act::SOFTMAX) {
      Mat::softmax(as_local[i + 1], as_local[i + 1]);
    } else {
      Mat::act(as_local[i + 1], layer_act);
    }
  }
}

float Net::loss(const Mat &t) {
  assert(arch[0] + arch.back() == t.cols());
  float total_c = 0.0f;
  size_t input_cols = arch[0];
  size_t output_cols = arch.back();
  size_t num_samples = t.rows();

#pragma omp parallel reduction(+ : total_c)
  {
    std::vector<Mat> local_as(arch.size());
    for (size_t i = 0; i < arch.size(); ++i)
      local_as[i] = Mat(1, arch[i]);

#pragma omp for
    for (size_t i = 0; i < num_samples; ++i) {
      for (size_t j = 0; j < input_cols; ++j)
        local_as[0](0, j) = t(i, j);
      forward(local_as);
      for (size_t j = 0; j < output_cols; ++j) {
        float d = local_as.back()(0, j) - t(i, input_cols + j);
        total_c += d * d;
      }
    }
  }
  return total_c / num_samples;
}

Net Net::backprop(const Mat &t) {
  Net g_final;
  g_final.alloc(arch);
  for (auto &w : g_final.ws)
    w.fill(0.0f);
  for (auto &b : g_final.bs)
    b.fill(0.0f);

  size_t num_samples = t.rows();
  int num_threads = omp_get_max_threads();
  std::vector<Net> local_gs(num_threads);

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    Net &g = local_gs[tid];
    g.alloc(arch);
    for (auto &w : g.ws)
      w.fill(0.0f);
    for (auto &b : g.bs)
      b.fill(0.0f);

    // [Critical Optimization] Pre-allocate buffers outside the loop for high
    // performance
    std::vector<Mat> local_as(arch.size());
    std::vector<Mat> das(arch.size());
    for (size_t i = 0; i < arch.size(); ++i) {
      local_as[i] = Mat(1, arch[i]);
      das[i] = Mat(1, arch[i]);
    }

    size_t input_cols = arch[0];

#pragma omp for schedule(static)
    for (size_t sample = 0; sample < num_samples; ++sample) {
      for (size_t j = 0; j < input_cols; ++j)
        local_as[0](0, j) = t(sample, j);

      forward(local_as);

      // Reset error gradient buffers without reallocating memory
      for (size_t k = 0; k < arch.size(); ++k)
        das[k].fill(0.0f);

      for (size_t j = 0; j < arch.back(); ++j) {
        float d = local_as.back()(0, j) - t(sample, input_cols + j);
        das.back()(0, j) = 2.0f * d;
      }

      for (int l = (int)arch.size() - 1; l > 0; --l) {
        for (size_t j = 0; j < arch[l]; ++j) {
          float a = local_as[l](0, j);
          float da = das[l](0, j);

          Act layer_act = this->layer[l - 1].activation;
          float qa = dactf(a, layer_act);

          g.bs[l - 1](0, j) += da * qa;

          for (size_t k = 0; k < arch[l - 1]; ++k) {
            float pa = local_as[l - 1](0, k);
            g.ws[l - 1](k, j) += da * qa * pa;
            das[l - 1](0, k) += da * qa * ws[l - 1](k, j);
          }
        }
      }
    }
  }

  // Aggregate local gradients from threads
  for (int tid = 0; tid < num_threads; ++tid) {
    if (local_gs[tid].arch.empty())
      continue;
    for (size_t i = 0; i < arch.size() - 1; ++i) {
      g_final.ws[i].add_inplace(local_gs[tid].ws[i]);
      g_final.bs[i].add_inplace(local_gs[tid].bs[i]);
    }
  }

  // Normalize by number of samples
  for (size_t i = 0; i < arch.size() - 1; ++i) {
    float inv_n = 1.0f / num_samples;
    for (size_t j = 0; j < g_final.ws[i].rows() * g_final.ws[i].cols(); ++j)
      g_final.ws[i].data()[j] *= inv_n;
    for (size_t k = 0; k < g_final.bs[i].cols(); ++k)
      g_final.bs[i](0, k) *= inv_n;
  }
  return g_final;
}

void Net::learn(const Net &g, float rate) {
  for (size_t i = 0; i < arch.size() - 1; ++i) {
    size_t ws_size = ws[i].rows() * ws[i].cols();
    float *ws_ptr = ws[i].data();
    const float *g_ws_ptr = g.ws[i].data();
    for (size_t j = 0; j < ws_size; ++j)
      ws_ptr[j] -= rate * g_ws_ptr[j];

    size_t bs_size = bs[i].cols();
    float *bs_ptr = bs[i].data();
    const float *g_bs_ptr = g.bs[i].data();
    for (size_t k = 0; k < bs_size; ++k)
      bs_ptr[k] -= rate * g_bs_ptr[k];
  }
}

void Net::init_adam() {
  m_ws.clear();
  v_ws.clear();
  m_bs.clear();
  v_bs.clear();
  for (size_t i = 0; i < arch.size() - 1; ++i) {
    m_ws.push_back(Mat(ws[i].rows(), ws[i].cols()));
    m_ws.back().fill(0.0f);

    v_ws.push_back(Mat(ws[i].rows(), ws[i].cols()));
    v_ws.back().fill(0.0f);

    m_bs.push_back(Mat(1, bs[i].cols()));
    m_bs.back().fill(0.0f);

    v_bs.push_back(Mat(1, bs[i].cols()));
    v_bs.back().fill(0.0f);
  }
  adam_t = 0;
}

void Net::adam_learn(const Net &g, float lr) {
  adam_t++;

  float beta1_t = std::pow(ADAM_BETA1, adam_t);
  float beta2_t = std::pow(ADAM_BETA2, adam_t);

  for (size_t i = 0; i < arch.size() - 1; i++) {

    size_t ws_size = ws[i].rows() * ws[i].cols();
    // update weights
    for (size_t j = 0; j < ws_size; ++j) {
      float g_val = g.ws[i].data()[j];

      g_val += lambda * ws[i].data()[j];

      m_ws[i].data()[j] =
          ADAM_BETA1 * m_ws[i].data()[j] + (1 - ADAM_BETA1) * g_val;

      v_ws[i].data()[j] =
          ADAM_BETA2 * v_ws[i].data()[j] + (1 - ADAM_BETA2) * (g_val * g_val);

      float m_hat = m_ws[i].data()[j] / (1 - beta1_t);
      float v_hat = v_ws[i].data()[j] / (1 - beta2_t);

      ws[i].data()[j] -= lr * m_hat / (std::sqrt(v_hat) + ADAM_EPS);
    }

    // update biases

    size_t bs_size = bs[i].rows() * bs[i].cols();
    for (size_t t = 0; t < bs_size; t++) {
      float g_val = g.bs[i](0, t);

      m_bs[i](0, t) = ADAM_BETA1 * m_bs[i](0, t) + (1 - ADAM_BETA1) * g_val;

      v_bs[i](0, t) =
          ADAM_BETA2 * v_bs[i](0, t) + (1 - ADAM_BETA2) * (g_val * g_val);

      float m_hat = m_bs[i](0, t) / (1 - beta1_t);
      float v_hat = v_bs[i](0, t) / (1 - beta2_t);

      bs[i](0, t) -= lr * m_hat / (std::sqrt(v_hat) + ADAM_EPS);
    }
  }
}

void Net::add_dense(size_t output_size, Act activation_fn) {
  size_t input_size = arch.empty() ? 0 : arch.back();

  if (arch.empty()) {
    arch.push_back(output_size);
    as.push_back(Mat(1, output_size));
  } else {
    arch.push_back(output_size);

    ws.push_back(Mat(input_size, output_size));
    bs.push_back(Mat(1, output_size));
    as.push_back(Mat(1, output_size));

    layer.push_back(Layer(input_size, output_size, activation_fn));
  }
}

void Net::save(const char *filename) const {
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file for saving: " << filename
              << std::endl;
    return;
  }

  if (arch.empty() || ws.size() + 1 != arch.size() ||
      bs.size() + 1 != arch.size() || layer.size() + 1 != arch.size()) {
    std::cerr << "Error: Net Structure is inconsistent, cannot save. \n";
    return;
  }

  auto write_value = [&](const auto &v) {
    file.write(reinterpret_cast<const char *>(&v), sizeof(v));
  };

  auto write_mat = [&](const Mat &m) {
    const size_t r = m.rows();
    const size_t c = m.cols();

    write_value(r);
    write_value(c);
    file.write(reinterpret_cast<const char *>(m.data()),
               static_cast<std::streamsize>(r * c * sizeof(float)));
  };

  // model head
  const uint32_t magic = 0x314E4E44;
  const uint32_t version = 1;
  write_value(magic);
  write_value(version);

  // model arch
  const size_t arch_size = arch.size();
  write_value(arch_size);
  file.write(reinterpret_cast<const char *>(arch.data()),
             static_cast<std::streamsize>(arch.size() * sizeof(size_t)));

  // layers & params

  const size_t dense_count = arch_size - 1;
  write_value(dense_count);

  for (size_t i = 0; i < dense_count; i++) {
    const int act = static_cast<int>(layer[i].activation);
    write_value(act);
    write_value(ws[i]);
    write_value(bs[i]);
  }
  write_value(lambda);
  if (!file.good()) {
    std::cerr << "Error: Failed saving model file: " << filename << "\n";
  }
}

void Net::load(const char *filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    std::cerr << "Error: Could not open file for loading:" << filename << '\n';
    return;
  }

  auto read_value = [&](auto &v) {
    file.read(reinterpret_cast<char *>(&v), sizeof(v));
  };

  auto read_mat = [&](Mat &m) {
    size_t r = 0, c = 0;
    read_value(r);
    read_value(c);
    m = Mat(r, c);
    file.read(reinterpret_cast<char *>(m.data()),
              static_cast<std::streamsize>(r * c * sizeof(float)));
  };

  uint32_t magic = 0, version = 0;
  read_value(magic);
  read_value(version);
  if (magic != 0x314E4E44 || version != 1) {
    std::cerr << "Error: Invalid model format/version.\n";
    return;
  }

  size_t arch_size = 0;
  read_value(arch_size);
  arch.resize(arch_size);
  file.read(reinterpret_cast<char *>(arch.data()),
            static_cast<std::streamsize>(arch_size * sizeof(size_t)));

  // reset
  alloc(arch);
  layer.clear();

  size_t dense_count = 0;
  read_value(dense_count);

  if (dense_count + 1 != arch_size) {
    std::cerr << "Error: Layer count mismatch.\n";
    return;
  }

  for (size_t i = 0; i < dense_count; i++) {
    int act_i = 0;
    read_value(act_i);
    layer.emplace_back(arch[i], arch[i + 1], static_cast<Act>(act_i));

    read_mat(ws[i]);
    read_mat(bs[i]);
  }

  read_value(lambda);

  if (!file.good()) {
    std::cerr << "Error: Failed while reading model file. \n";
    return;
  }
  init_adam();
}

} // namespace nn
