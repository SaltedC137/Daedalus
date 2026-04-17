
#include "mat.hpp"
#include <cassert>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <utility>

namespace nn {

RowView Matrix_::row(size_t i) {
  assert(i < rows_);
  return RowView{cols_, &data_[i * cols_]};
}

void Matrix_::fill(float v) { std::fill(data_.begin(), data_.end(), v); }

void Matrix_::rand_fill(float low, float high, std::mt19937 &rng) {
  std::uniform_real_distribution<float> d(low, high);
  for (auto &x : data_) {
    x = d(rng);
  }
}

void Matrix_::copy_from(const Matrix_ &src) {
  assert(rows_ == src.rows_ && cols_ == src.cols_);
  data_ = src.data_;
}

void Matrix_::add_inplace(const Matrix_ &other) {
  assert(rows_ == other.rows_ && cols_ == other.cols_);
  for (size_t i = 0, n = data_.size(); i < n; i++) {
    data_[i] += other.data_[i];
  }
}

// user-defined
void Matrix_::apply(const std::function<float(float)> &fn) {
  for (auto &v : data_) {
    fn(v);
  }
}

// TODO:muitiple thread calculate
void Matrix_::matmul(Matrix_ &dst, const Matrix_ &a, const Matrix_ &b) {
  assert(a.cols_ == b.rows_);
  assert(dst.rows_ == a.rows() && dst.cols_ == b.cols_);

  size_t m = a.rows_, n = a.cols_, p = b.cols_;
  // simple i-k-j may be slightly faster for CPU cache in many case

  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      dst(i, j) = 0.0f;
    }
    for (int k = 0; k < n; ++k) {
      float aik = a(i, k);
      for (size_t j = 0; j < p; j++) {
        dst(i, j) += aik * b(k, j);
      }
    }
  }
}

void Matrix_::shuffle_rows(std::mt19937 &rng) {
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

void Matrix_::print(std::string_view name, int precision) const {
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

} // namespace nn
