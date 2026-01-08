#include "model.hpp"
#include <ATen/Parallel.h>

Model::Model(const std::string& _model_path) : model_path_(_model_path) {
  torch::set_num_threads(1);
  try {
    model_ = torch::jit::load(model_path_);
    loaded_ = true;
  } catch (const std::exception& e) {
    loaded_ = false;
  }
}

Model::~Model() {}

std::vector<float> Model::Forward(
    const std::vector<std::vector<float>>& input_vec) {
  if (!loaded_) {
    throw std::runtime_error("Model not loaded.");
  }
  try {
    // Prepare input tensor
    const auto& input = input_vec[0];
    auto input_tensor = torch::tensor(input, torch::kFloat32)
                            .reshape({1, static_cast<int64_t>(input.size())});

    // Disable gradient calculation for inference
    torch::autograd::GradMode::set_enabled(false);

    // Limit to single thread for inference
    torch::set_num_threads(1);

    // Perform inference
    auto output_tensor = model_.forward({input_tensor}).toTensor();

    // Convert output tensor to vector
    return Torch2Vec(output_tensor);
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("Model inference failed: ") +
                             e.what());
  }
}

std::vector<float> Model::Torch2Vec(const torch::Tensor& input) {
  auto cpu_tensor = input.is_contiguous() ? input : input.contiguous();
  if (cpu_tensor.device().type() != torch::kCPU) {
    cpu_tensor = cpu_tensor.to(torch::kCPU);
  }

  // Get data pointer and size
  float* data_ptr = cpu_tensor.data_ptr<float>();
  int64_t num_elements = cpu_tensor.numel();

  // Copy data to vector
  return std::vector<float>(data_ptr, data_ptr + num_elements);
}