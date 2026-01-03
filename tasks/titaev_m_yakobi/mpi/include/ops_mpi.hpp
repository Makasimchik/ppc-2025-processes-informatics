#pragma once

#include <vector>

#include "task/include/task.hpp"
#include "titaev_m_yakobi/common/include/common.hpp"

namespace titaev_m_yakobi {

class TitaevMYakobiMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit TitaevMYakobiMPI(const InType &in);

 private:
  int CheckConvergence(int rank, int n, const std::vector<ValueType> &x_new_global, const std::vector<ValueType> &x_old,
                       ValueType eps);
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void ComputeLocal(const std::vector<ValueType> &x_old, std::vector<ValueType> &x_new_local, int start_row,
                    int my_rows);
};

}  // namespace titaev_m_yakobi
