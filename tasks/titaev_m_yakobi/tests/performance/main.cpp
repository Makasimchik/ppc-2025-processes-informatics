#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "titaev_m_yakobi/common/include/common.hpp"
#include "titaev_m_yakobi/mpi/include/ops_mpi.hpp"
#include "titaev_m_yakobi/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace titaev_m_yakobi {

class TitaevMYakobiPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    test_input_.n = 300;
    test_input_.A.assign(static_cast<std::size_t>(300) * 300, 0.0);
    test_input_.b.assign(300, 0.0);
    test_input_.x0.assign(300, 0.0);
    test_input_.eps = 1e-6;
    test_input_.max_iter = 10000;

    for (int i = 0; i < 300; i++) {
      for (int j = 0; j < 300; j++) {
        if (i == j) {
          test_input_.A[(i * 300) + j] = 4.0;
        } else if (std::abs(i - j) == 1) {
          test_input_.A[(i * 300) + j] = 1.0;
        }
      }
      test_input_.b[i] = 0.0;
      for (int j = 0; j < 300; j++) {
        test_input_.b[i] += test_input_.A[(i * 300) + j];
      }
    }
  }

  bool CheckTestOutputData(OutType &output) final {
    const double tol = 1e-4;
    bool all_ok = true;
    for (const auto x : output) {
      if (std::fabs(x - 1.0) > tol) {
        all_ok = false;
        break;
      }
    }
    return all_ok;
  }

  InType GetTestInputData() final {
    return test_input_;
  }

 private:
  InType test_input_;
};

TEST_P(TitaevMYakobiPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, TitaevMYakobiMPI, TitaevMYakobiSEQ>(PPC_SETTINGS_titaev_m_yakobi);

const auto kGtestValues = ppc::util::TupleToGTestValues(kPerfTasks);

INSTANTIATE_TEST_SUITE_P(RunPerfTests, TitaevMYakobiPerfTests, kGtestValues,
                         TitaevMYakobiPerfTests::CustomPerfTestName);

}  // namespace titaev_m_yakobi
