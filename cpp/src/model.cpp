#include "model.hpp"
#include <ATen/Parallel.h>
#include <torch/torch.h>

Model::Model() {
  if (torch::cuda::is_available()) {
    device_type_ = torch::kCUDA;
  } else {
    device_type_ = torch::kCPU;
  }
    torch::set_num_threads(1);
}

void Model::Init(const std::string& _model_path) {
  model_path_ = _model_path;
  try {
    model_ = torch::jit::load(model_path_);
    model_.to(device_type_);
    model_.eval(); // 确保是评价模式
    loaded_ = true;
  } catch (const std::exception& e) {
    loaded_ = false;
  }
}

std::vector<float> Model::Forward(const std::vector<float>& input_data) {
  if (!loaded_)
    throw std::runtime_error("Model not loaded.");
  torch::set_num_threads(1);
  torch::NoGradGuard no_grad;
  try {
    auto input_tensor =
        torch::tensor(input_data, torch::kFloat32).unsqueeze(0).to(device_type_);
    return Torch2Vec(model_.forward({input_tensor}).toTensor());
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("Model inference failed: ") +
                             e.what());
  }
}

std::vector<float> Model::Torch2Vec(const torch::Tensor& _tensor) {
  auto cpu_tensor = _tensor.to(torch::kCPU).contiguous();
  return {cpu_tensor.data_ptr<float>(),
          cpu_tensor.data_ptr<float>() + cpu_tensor.numel()};
}