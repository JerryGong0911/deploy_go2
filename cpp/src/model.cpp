#include "model.hpp"
#include <ATen/Parallel.h>

Model::Model(){
  torch::set_num_threads(1);
}

void Model::Init(const std::string& _model_path) {
  model_path_ = _model_path;
  try {
    model_ = torch::jit::load(model_path_);
    loaded_ = true;
  } catch (const std::exception& e) {
    loaded_ = false;
  }
}

std::vector<float> Model::Forward(
    const std::vector<std::vector<float>>& input_vec) {
  if (!loaded_) throw std::runtime_error("Model not loaded.");

  torch::NoGradGuard no_grad;
  try {
    auto input_tensor = torch::tensor(input_vec[0], torch::kFloat32).unsqueeze(0);
    return Torch2Vec(model_.forward({input_tensor}).toTensor());
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("Model inference failed: ") + e.what());
  }
}

std::vector<float> Model::Torch2Vec(const torch::Tensor& input) {
  auto t = input.to(torch::kCPU).contiguous();
  return {t.data_ptr<float>(), t.data_ptr<float>() + t.numel()};
}