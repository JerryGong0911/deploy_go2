#include "model.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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
  inputs_.clear();
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
    if (n_out == 0 || n_in < n_out)
      throw std::runtime_error(
          "expected M data inputs + S state inputs -> 1 + S outputs, got " +
          std::to_string(n_in) + " inputs / " + std::to_string(n_out) +
          " outputs");
    const size_t n_states = n_out - 1;
    const size_t n_data = n_in - n_states;

    Ort::AllocatorWithDefaultOptions alloc;
    output_name_ = session_->GetOutputNameAllocated(0, alloc).get();

    for (size_t i = 0; i < n_data; ++i) {
      Input in;
      in.name = session_->GetInputNameAllocated(i, alloc).get();
      // Keep the TypeInfo alive: the shape-info view points into it.
      Ort::TypeInfo type = session_->GetInputTypeInfo(i);
      auto info = type.GetTensorTypeAndShapeInfo();
      if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        throw std::runtime_error("input '" + in.name + "' must be float32");
      in.shape = ResolveShape(info.GetShape(), &in.size);
      in.buf.assign(in.size, 0.0f);
      inputs_.push_back(std::move(in));
    }

    // Recurrent states: input n_data + k pairs with output 1 + k.
    for (size_t k = 0; k < n_states; ++k) {
      State st;
      st.in_name = session_->GetInputNameAllocated(n_data + k, alloc).get();
      st.out_name = session_->GetOutputNameAllocated(1 + k, alloc).get();

      Ort::TypeInfo in_type = session_->GetInputTypeInfo(n_data + k);
      Ort::TypeInfo out_type = session_->GetOutputTypeInfo(1 + k);
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

    // inputs_ / states_ are final now, so these pointers stay valid.
    in_names_.clear();
    out_names_ = {output_name_.c_str()};
    for (const auto& in : inputs_)
      in_names_.push_back(in.name.c_str());
    for (const auto& st : states_) {
      in_names_.push_back(st.in_name.c_str());
      out_names_.push_back(st.out_name.c_str());
    }
    loaded_ = true;

    std::cout << "Loaded " << (IsRecurrent() ? "recurrent" : "feed-forward")
              << " model '" << model_path_ << "':";
    for (const auto& in : inputs_)
      std::cout << " input '" << in.name << "' " << ShapeStr(in.shape) << ";";
    for (const auto& st : states_)
      std::cout << " state '" << st.in_name << "' -> '" << st.out_name << "' "
                << ShapeStr(st.shape) << ";";
    std::cout << std::endl;

    // Warm up so the first control step does not pay for allocations, then
    // clear the hidden state the warm-up call produced.
    std::vector<std::vector<float>> zeros;
    for (const auto& in : inputs_)
      zeros.emplace_back(in.size, 0.0f);
    Forward(zeros);
    ResetStates();
  } catch (const std::exception& e) {
    loaded_ = false;
    session_.reset();
    inputs_.clear();
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
  if (inputs_.size() != 1)
    throw std::runtime_error("Model '" + model_path_ + "' has " +
                             std::to_string(inputs_.size()) +
                             " data inputs, single-input Forward used");
  return Forward(std::vector<std::vector<float>>{input_data});
}

std::vector<float> Model::Forward(
    const std::vector<std::vector<float>>& input_data) {
  if (!loaded_)
    throw std::runtime_error("Model not loaded.");
  if (input_data.size() != inputs_.size())
    throw std::runtime_error("Model '" + model_path_ + "' expects " +
                             std::to_string(inputs_.size()) +
                             " data inputs, got " +
                             std::to_string(input_data.size()));
  for (size_t i = 0; i < inputs_.size(); ++i) {
    if (input_data[i].size() != inputs_[i].size)
      throw std::runtime_error("Model input size mismatch for '" + model_path_ +
                               "' input '" + inputs_[i].name + "': expected " +
                               std::to_string(inputs_[i].size) + ", got " +
                               std::to_string(input_data[i].size()));
    std::copy(input_data[i].begin(), input_data[i].end(),
              inputs_[i].buf.begin());
  }

  try {
    std::vector<Ort::Value> tensors;
    tensors.reserve(inputs_.size() + states_.size());
    for (auto& in : inputs_)
      tensors.push_back(Ort::Value::CreateTensor<float>(
          mem_info_, in.buf.data(), in.buf.size(), in.shape.data(),
          in.shape.size()));
    for (auto& st : states_)
      tensors.push_back(Ort::Value::CreateTensor<float>(
          mem_info_, st.data.data(), st.data.size(), st.shape.data(),
          st.shape.size()));

    auto outputs = session_->Run(Ort::RunOptions{nullptr}, in_names_.data(),
                                 tensors.data(), tensors.size(),
                                 out_names_.data(), out_names_.size());

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
