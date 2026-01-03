#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace titaev_m_yakobi {

using ValueType = double;

struct JacobiInput {
  std::vector<ValueType> A;   // N x N, построчно
  std::vector<ValueType> b;   // N
  std::vector<ValueType> x0;  // N
  int n = 0;                  // размер системы
  ValueType eps = 1e-6;
  int max_iter = 1000;
};

using InType = JacobiInput;
using OutType = std::vector<ValueType>;  // решение x
using TestType = std::tuple<int, std::string>;

using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace titaev_m_yakobi
