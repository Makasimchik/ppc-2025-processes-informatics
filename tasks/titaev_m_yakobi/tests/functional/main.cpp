#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "titaev_m_yakobi/common/include/common.hpp"
#include "titaev_m_yakobi/mpi/include/ops_mpi.hpp"
#include "titaev_m_yakobi/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace titaev_m_yakobi {

class TitaevMYakobiFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 protected:
  void SetUp() override {
    test_input_.n = 3;
    test_input_.A = {4, 1, 0, 1, 4, 1, 0, 1, 4};
    test_input_.b = {5, 6, 5};
    test_input_.x0 = {0, 0, 0};
    test_input_.eps = 1e-8;
    test_input_.max_iter = 10000;
  }

  bool CheckTestOutputData(OutType &output) override {
    const std::vector<ValueType> expected = {1, 1, 1};
    const double tol = 1e-6;
    for (size_t i = 0; i < expected.size(); ++i) {
      if (std::fabs(output[i] - expected[i]) > tol) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() override {
    return test_input_;
  }

 public:
  static std::string PrintTestParam(const TestType &param) {
    int n;
    std::string name;
    std::tie(n, name) = param;
    return name + "_n" + std::to_string(n);
  }

 private:
  InType test_input_;
};

TEST_P(TitaevMYakobiFuncTests, SolveSystem) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 1> kTestParam = {std::make_tuple(3, "small")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<TitaevMYakobiMPI, InType>(kTestParam, PPC_SETTINGS_titaev_m_yakobi),
                   ppc::util::AddFuncTask<TitaevMYakobiSEQ, InType>(kTestParam, PPC_SETTINGS_titaev_m_yakobi));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

INSTANTIATE_TEST_SUITE_P(SolveSystem, TitaevMYakobiFuncTests, kGtestValues,
                         TitaevMYakobiFuncTests::PrintFuncTestName<TitaevMYakobiFuncTests>);

}  // namespace titaev_m_yakobi
