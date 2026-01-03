#pragma once

#include <vector>

#include "task/include/task.hpp"
#include "titaev_m_yakobi/common/include/common.hpp"

namespace titaev_m_yakobi {

class TitaevMYakobiSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit TitaevMYakobiSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void Iterate(const std::vector<ValueType> &x_old, std::vector<ValueType> &x_new);
};

}  // namespace titaev_m_yakobi
