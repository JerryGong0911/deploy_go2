#ifndef __MODEL_HPP__
#define __MODEL_HPP__

#include <onnxruntime_cxx_api.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Single-input / single-output ONNX model run on CPU with ONNX Runtime.
class Model {
 public:
  Model() = default;
  ~Model() = default;

 public:
  void Init(const std::string& _model_path);
  bool IsLoaded() { return loaded_; }
  std::vector<float> Forward(const std::vector<float>& input_data);

 private:
  static Ort::Env& GetEnv();  // shared by all sessions, must outlive them

 private:
  bool loaded_ = false;
  std::string model_path_;
  std::unique_ptr<Ort::Session> session_;
  Ort::MemoryInfo mem_info_ =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::string input_name_, output_name_;
  std::vector<int64_t> input_shape_;  // dynamic dims resolved to 1
  size_t input_size_ = 0;             // number of elements per Forward call
  std::vector<float> input_buf_;      // CreateTensor needs a mutable buffer
};

#endif
