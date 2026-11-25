#include "model.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace {
std::string ShapeStr(const std::vector<int64_t>& shape) {
  std::string s = "[";
  for (size_t i = 0; i < shape.size(); ++i)
    s += (i ? ", " : "") + std::to_string(shape[i]);
  return s + "]";
}
}  // namespace

Ort::Env& Model::GetEnv() {
  static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "go2_deploy");
  return env;
}

// Dynamic dims (batch, sequence length) are resolved to 1: one step, one env.
std::vector<int64_t> Model::ResolveShape(std::vector<int64_t> shape,
                                         size_t* numel) {
  *numel = 1;
  for (auto& d : shape) {
    if (d < 0)
      d = 1;
    *numel *= static_cast<size_t>(d);
  }
  return shape;
}

void Model::Init(const std::string& _model_path) {
  model_path_ = _model_path;
  loaded_ = false;
  states_.clear();
  try {
    Ort::SessionOptions opts;
    // Small networks: single-threaded CPU inference is fastest and most stable.
    opts.SetIntraOpNumThreads(1);
    opts.SetInterOpNumThreads(1);
    opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    opts.AddConfigEntry("session.intra_op.allow_spinning", "0");

    session_ =
        std::make_unique<Ort::Session>(GetEnv(), model_path_.c_str(), opts);

    const size_t n_in = session_->GetInputCount();
    const size_t n_out = session_->GetOutputCount();
    if (n_in == 0 || n_in != n_out)
      throw std::runtime_error(
          "expected 1 input / 1 output (feed-forward) or N inputs / N "
          "outputs (recurrent), got " +
          std::to_string(n_in) + " inputs / " + std::to_string(n_out) +
          " outputs");

    Ort::AllocatorWithDefaultOptions alloc;
    input_name_ = session_->GetInputNameAllocated(0, alloc).get();
    output_name_ = session_->GetOutputNameAllocated(0, alloc).get();
    input_shape_ = ResolveShape(
        session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape(),
        &input_size_);
    input_buf_.assign(input_size_, 0.0f);

    // Recurrent states: input k pairs with output k (k >= 1).
    for (size_t k = 1; k < n_in; ++k) {
      State st;
      st.in_name = session_->GetInputNameAllocated(k, alloc).get();
      st.out_name = session_->GetOutputNameAllocated(k, alloc).get();

      // The TypeInfo objects own the data the shape-info views point into,
      // so they must stay alive while in_info / out_info are used.
      Ort::TypeInfo in_type = session_->GetInputTypeInfo(k);
      Ort::TypeInfo out_type = session_->GetOutputTypeInfo(k);
      auto in_info = in_type.GetTensorTypeAndShapeInfo();
      auto out_info = out_type.GetTensorTypeAndShapeInfo();
      if (in_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
          out_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        throw std::runtime_error("state '" + st.in_name + "' / '" +
                                 st.out_name + "' must be float32");

      size_t in_numel = 0, out_numel = 0;
      st.shape = ResolveShape(in_info.GetShape(), &in_numel);
      auto out_shape = ResolveShape(out_info.GetShape(), &out_numel);
      if (st.shape != out_shape)
        throw std::runtime_error("state shape mismatch: input '" + st.in_name +
                                 "' " + ShapeStr(st.shape) + " vs output '" +
                                 st.out_name + "' " + ShapeStr(out_shape));
      st.data.assign(in_numel, 0.0f);
      states_.push_back(std::move(st));
    }

    // states_ is final now, so these pointers stay valid.
    in_names_ = {input_name_.c_str()};
    out_names_ = {output_name_.c_str()};
    for (const auto& st : states_) {
      in_names_.push_back(st.in_name.c_str());
      out_names_.push_back(st.out_name.c_str());
    }
    loaded_ = true;

    std::cout << "Loaded " << (IsRecurrent() ? "recurrent" : "feed-forward")
              << " model '" << model_path_ << "': input '" << input_name_
              << "' " << ShapeStr(input_shape_);
    for (const auto& st : states_)
      std::cout << ", state '" << st.in_name << "' -> '" << st.out_name << "' "
                << ShapeStr(st.shape);
    std::cout << std::endl;

    // Warm up so the first control step does not pay for allocations, then
    // clear the hidden state the warm-up call produced.
    Forward(std::vector<float>(input_size_, 0.0f));
    ResetStates();
  } catch (const std::exception& e) {
    loaded_ = false;
    session_.reset();
    states_.clear();
    throw std::runtime_error("Failed to load ONNX model '" + model_path_ +
                             "': " + e.what());
  }
}

void Model::ResetStates() {
  for (auto& st : states_)
    std::fill(st.data.begin(), st.data.end(), 0.0f);
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
    std::vector<Ort::Value> inputs;
    inputs.reserve(1 + states_.size());
    inputs.push_back(Ort::Value::CreateTensor<float>(
        mem_info_, input_buf_.data(), input_buf_.size(), input_shape_.data(),
        input_shape_.size()));
    for (auto& st : states_)
      inputs.push_back(Ort::Value::CreateTensor<float>(
          mem_info_, st.data.data(), st.data.size(), st.shape.data(),
          st.shape.size()));

    auto outputs =
        session_->Run(Ort::RunOptions{nullptr}, in_names_.data(), inputs.data(),
                      inputs.size(), out_names_.data(), out_names_.size());

    // Carry the new hidden state into the next step.
    for (size_t k = 0; k < states_.size(); ++k) {
      auto& st = states_[k];
      const auto& out = outputs[k + 1];
      if (out.GetTensorTypeAndShapeInfo().GetElementCount() != st.data.size())
        throw std::runtime_error("state output '" + st.out_name +
                                 "' size changed at runtime");
      const float* p = out.GetTensorData<float>();
      std::copy(p, p + st.data.size(), st.data.begin());
    }

    const float* out = outputs[0].GetTensorData<float>();
    size_t n = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    return {out, out + n};
  } catch (const Ort::Exception& e) {
    throw std::runtime_error(std::string("Model inference failed: ") +
                             e.what());
  }
}
