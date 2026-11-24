#include "model.hpp"
#include <algorithm>
#include <stdexcept>

Ort::Env& Model::GetEnv() {
  static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "go2_deploy");
  return env;
}

void Model::Init(const std::string& _model_path) {
  model_path_ = _model_path;
  loaded_ = false;
  try {
    Ort::SessionOptions opts;
    // Small MLPs: single-threaded CPU inference is fastest and most stable.
    opts.SetIntraOpNumThreads(1);
    opts.SetInterOpNumThreads(1);
    opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    opts.AddConfigEntry("session.intra_op.allow_spinning", "0");

    session_ =
        std::make_unique<Ort::Session>(GetEnv(), model_path_.c_str(), opts);

    if (session_->GetInputCount() != 1 || session_->GetOutputCount() != 1)
      throw std::runtime_error("expected exactly 1 input and 1 output");

    Ort::AllocatorWithDefaultOptions alloc;
    input_name_ = session_->GetInputNameAllocated(0, alloc).get();
    output_name_ = session_->GetOutputNameAllocated(0, alloc).get();

    input_shape_ =
        session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    input_size_ = 1;
    for (auto& d : input_shape_) {
      if (d < 0) d = 1;  // dynamic batch -> 1
      input_size_ *= static_cast<size_t>(d);
    }
    input_buf_.assign(input_size_, 0.0f);
    loaded_ = true;

    // Warm up so the first control step does not pay for allocations.
    Forward(input_buf_);
  } catch (const std::exception& e) {
    loaded_ = false;
    session_.reset();
    throw std::runtime_error("Failed to load ONNX model '" + model_path_ +
                             "': " + e.what());
  }
}

std::vector<float> Model::Forward(const std::vector<float>& input_data) {
  if (!loaded_)
    throw std::runtime_error("Model not loaded.");
  if (input_data.size() != input_size_)
    throw std::runtime_error("Model input size mismatch for '" + model_path_ +
                             "': expected " + std::to_string(input_size_) +
                             ", got " + std::to_string(input_data.size()));

  std::copy(input_data.begin(), input_data.end(), input_buf_.begin());
  try {
    Ort::Value input = Ort::Value::CreateTensor<float>(
        mem_info_, input_buf_.data(), input_buf_.size(), input_shape_.data(),
        input_shape_.size());

    const char* in_names[] = {input_name_.c_str()};
    const char* out_names[] = {output_name_.c_str()};
    auto outputs = session_->Run(Ort::RunOptions{nullptr}, in_names, &input,
                                 1, out_names, 1);

    const float* out = outputs[0].GetTensorData<float>();
    size_t n = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    return {out, out + n};
  } catch (const Ort::Exception& e) {
    throw std::runtime_error(std::string("Model inference failed: ") +
                             e.what());
  }
}
