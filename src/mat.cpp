
#include "mat.hpp"
#include <cassert>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <random>
#include <utility>
#include <vector>

namespace nn {

RowView Mat::row(size_t i) {
  assert(i < rows_);
  return RowView{cols_, &data_[i * cols_]};
}

void Mat::fill(float v) { std::fill(data_.begin(), data_.end(), v); }

void Mat::rand_fill(float low, float high, std::mt19937 &rng) {
  std::uniform_real_distribution<float> d(low, high);
  for (auto &x : data_) {
    x = d(rng);
  }
}

void Mat::copy_from(const Mat &src) {
  assert(rows_ == src.rows_ && cols_ == src.cols_);
  data_ = src.data_;
}

void Mat::add_inplace(const Mat &other) {
  assert(rows_ == other.rows_ && cols_ == other.cols_);
  for (size_t i = 0, n = data_.size(); i < n; i++) {
    data_[i] += other.data_[i];
  }
}

// user-defined
void Mat::apply(const std::function<float(float)> &fn) {
  for (auto &v : data_) {
    fn(v);
  }
}

void Mat::matmul(Mat &dst, const Mat &a, const Mat &b) {
  assert(a.cols_ == b.rows_);
  assert(dst.rows_ == a.rows() && dst.cols_ == b.cols_);

  size_t m = a.rows_, n = a.cols_, p = b.cols_;

#pragma omp parallel for
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      dst(i, j) = 0.0f;
    }
    for (size_t k = 0; k < n; ++k) {
      float aik = a(i, k);
      for (size_t j = 0; j < p; j++) {
        dst(i, j) += aik * b(k, j);
      }
    }
  }
}

void Mat::shuffle_rows(std::mt19937 &rng) {
  for (size_t i = 0; i < rows_; i++) {
    std::uniform_int_distribution<size_t> d(i, rows_ - 1);
    size_t j = d(rng);
    if (i == j)
      continue;
    float *ri = &data_[i * cols_];
    float *rj = &data_[j * cols_];
    for (size_t c = 0; c < cols_; c++) {
      std::swap(ri[c], rj[c]);
    }
  }
}

void Mat::print(std::string_view name, int precision) const {
  if (!name.empty())
    std::cout << name << " = [\n";
  for (size_t i = 0; i < rows_; ++i) {
    std::cout << "  ";
    for (size_t j = 0; j < cols_; ++j)
      std::cout << std::fixed << std::setprecision(precision) << (*this)(i, j)
                << " ";
    std::cout << "\n";
  }
  if (!name.empty())
    std::cout << "    ]\n";
}

void Mat::dot(Mat &dst, const Mat &a, const Mat &b) {
  assert(a.cols_ == b.rows_);
  assert(dst.rows_ == a.rows_ && dst.cols_ == b.cols_);

  size_t n = a.cols_;
  for (size_t i = 0; i < dst.rows_; ++i) {
    for (size_t j = 0; j < dst.cols_; ++j) {
      dst(i, j) = 0.0f;
      for (size_t k = 0; k < n; k++) {
        dst(i, j) += a(i, k) * b(k, j);
      }
    }
  }
}

void Mat::sum(Mat &dst, const Mat &src) {
  assert(dst.rows_ == src.rows_ && dst.cols_ == src.cols_);
  for (size_t i = 0; i < dst.data_.size(); ++i) {
    dst.data_[i] += src.data_[i];
  }
}

void Mat::act(Mat &dst, Act activation) {
  for (auto &i : dst.data_) {
    i = actf(i, activation);
  }
}

// softmax
void Mat::softmax(Mat &dst, const Mat &src) {
  assert(dst.rows_ == src.rows_ && dst.cols_ == src.cols_);

  for (size_t i = 0; i < src.rows_; ++i) {
    float max_val = src(i, 0);
    for (size_t j = 1; j < src.cols_; ++j) {
      if (src(i, j) > max_val)
        max_val = src(i, j);
    }
    float sum_exp = 0.0f;
    for (size_t j = 0; j < src.cols_; ++j) {
      dst(i, j) = std::exp(src(i, j) - max_val);
      sum_exp += dst(i, j);
    }
    for (size_t j = 0; j < src.cols_; ++j) {
      dst(i, j) /= sum_exp;
    }
  }
}

} // namespace nn
