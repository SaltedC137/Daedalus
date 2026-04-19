
#include "net.hpp"
#include <cstddef>
#include <vector>
#include <omp.h>
#include <cassert>

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

void Net::forward(std::vector<Mat> &as_local) const {
  for (size_t i = 0; i < arch.size() - 1; ++i) {
    Mat::dot(as_local[i + 1], as_local[i], ws[i]);
    Mat::sum(as_local[i + 1], bs[i]);
    Mat::act(as_local[i + 1], activation);
  }
}

float Net::loss(const Mat &t) {
  assert(arch[0] + arch.back() == t.cols());
  float total_c = 0.0f;
  size_t input_cols = arch[0];
  size_t output_cols = arch.back();
  size_t num_samples = t.rows();

  #pragma omp parallel reduction(+:total_c)
  {
    std::vector<Mat> local_as(arch.size());
    for (size_t i = 0; i < arch.size(); ++i) local_as[i] = Mat(1, arch[i]);

    #pragma omp for
    for (size_t i = 0; i < num_samples; ++i) {
      for (size_t j = 0; j < input_cols; ++j) local_as[0](0, j) = t(i, j);
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
  for (auto &w : g_final.ws) w.fill(0.0f);
  for (auto &b : g_final.bs) b.fill(0.0f);

  size_t num_samples = t.rows();
  int num_threads = omp_get_max_threads();
  std::vector<Net> local_gs(num_threads);

  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    Net &g = local_gs[tid];
    g.alloc(arch);
    for (auto &w : g.ws) w.fill(0.0f);
    for (auto &b : g.bs) b.fill(0.0f);
    
    // [Critical Optimization] Pre-allocate buffers outside the loop for high performance
    std::vector<Mat> local_as(arch.size());
    std::vector<Mat> das(arch.size());
    for (size_t i = 0; i < arch.size(); ++i) {
      local_as[i] = Mat(1, arch[i]);
      das[i] = Mat(1, arch[i]);
    }

    size_t input_cols = arch[0];

    #pragma omp for schedule(static)
    for (size_t sample = 0; sample < num_samples; ++sample) {
      for (size_t j = 0; j < input_cols; ++j) local_as[0](0, j) = t(sample, j);
      
      forward(local_as);

      // Reset error gradient buffers without reallocating memory
      for (size_t k = 0; k < arch.size(); ++k) das[k].fill(0.0f);

      for (size_t j = 0; j < arch.back(); ++j) {
        float d = local_as.back()(0, j) - t(sample, input_cols + j);
        das.back()(0, j) = 2.0f * d;
      }

      for (int l = (int)arch.size() - 1; l > 0; --l) {
        for (size_t j = 0; j < arch[l]; ++j) {
          float a = local_as[l](0, j);
          float da = das[l](0, j);
          float qa = dactf(a, activation);

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
    if (local_gs[tid].arch.empty()) continue;
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
    for (size_t j = 0; j < ws_size; ++j) ws_ptr[j] -= rate * g_ws_ptr[j];

    size_t bs_size = bs[i].cols();
    float *bs_ptr = bs[i].data();
    const float *g_bs_ptr = g.bs[i].data();
    for (size_t k = 0; k < bs_size; ++k) bs_ptr[k] -= rate * g_bs_ptr[k];
  }
}

} // namespace nn
