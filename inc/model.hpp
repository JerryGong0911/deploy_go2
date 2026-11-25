#ifndef __MODEL_HPP__
#define __MODEL_HPP__

#include <onnxruntime_cxx_api.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// ONNX model run on CPU with ONNX Runtime.
//
// Two layouts are supported, detected from the model signature:
//   - Feed-forward: 1 input, 1 output (e.g. MLP policy / encoder).
//   - Recurrent:    N inputs, N outputs, N > 1 (e.g. GRU / LSTM).
//       input[0]  = observation,  output[0] = network output
//       input[k]  = state k in,   output[k] = state k out   (k = 1..N-1)
//     States are paired by position, must have matching shapes, are kept
//     inside the Model, start at zero, and are fed back on every Forward.
//     A GRU exported as (obs, h_in) -> (out, h_out) is the N = 2 case;
//     an LSTM exported as (obs, h_in, c_in) -> (out, h_out, c_out) is N = 3.
class Model {
 public:
  Model() = default;
  ~Model() = default;

 public:
  void Init(const std::string& _model_path);
  bool IsLoaded() const { return loaded_; }
  bool IsRecurrent() const { return !states_.empty(); }
  size_t InputSize() const { return input_size_; }
  std::vector<float> Forward(const std::vector<float>& input_data);
  void ResetStates();  // zero all recurrent states (no-op for feed-forward)

 private:
  struct State {
    std::string in_name, out_name;
    std::vector<int64_t> shape;  // dynamic dims resolved to 1
    std::vector<float> data;
  };

  static Ort::Env& GetEnv();  // shared by all sessions, must outlive them
  static std::vector<int64_t> ResolveShape(std::vector<int64_t> shape,
                                           size_t* numel);

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
  std::vector<State> states_;         // empty for feed-forward models

  // Cached name arrays for Session::Run (point into the strings above).
  std::vector<const char*> in_names_, out_names_;
};

#endif
