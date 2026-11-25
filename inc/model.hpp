#ifndef __MODEL_HPP__
#define __MODEL_HPP__

#include <onnxruntime_cxx_api.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// ONNX model run on CPU with ONNX Runtime.
//
// Signature: M data inputs followed by S recurrent states, 1 + S outputs.
//   input[0 .. M-1]   = data inputs (e.g. proprioception, depth image)
//   input[M + k]      = state k in     output[1 + k] = state k out
//   output[0]         = network output
// S = (#outputs - 1) and M = (#inputs - S), so the layout follows from the
// input/output counts:
//   - MLP:                    (obs)              -> (out)
//   - GRU:                    (obs, h_in)        -> (out, h_out)
//   - LSTM:                   (obs, h_in, c_in)  -> (out, h_out, c_out)
//   - depth + proprio, GRU:   (obs, depth, h_in) -> (out, h_out)
// States must be float32 with matching in/out shapes. They are kept inside
// the Model, start at zero, and are fed back on every Forward.
class Model {
 public:
  Model() = default;
  ~Model() = default;

 public:
  void Init(const std::string& _model_path);
  bool IsLoaded() const { return loaded_; }
  bool IsRecurrent() const { return !states_.empty(); }
  size_t NumInputs() const { return inputs_.size(); }  // data inputs only
  size_t InputSize(size_t i = 0) const { return inputs_.at(i).size; }

  // One step. inputs[i] feeds data input i (row-major, InputSize(i) floats).
  std::vector<float> Forward(const std::vector<std::vector<float>>& inputs);
  // Convenience for single-input models.
  std::vector<float> Forward(const std::vector<float>& input_data);

  void ResetStates();  // zero all recurrent states (no-op for feed-forward)

 private:
  struct Input {
    std::string name;
    std::vector<int64_t> shape;  // dynamic dims resolved to 1
    size_t size = 0;             // number of elements per Forward call
    std::vector<float> buf;      // CreateTensor needs a mutable buffer
  };
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

  std::vector<Input> inputs_;  // data inputs, in model order
  std::string output_name_;
  std::vector<State> states_;  // empty for feed-forward models

  // Cached name arrays for Session::Run (point into the strings above).
  std::vector<const char*> in_names_, out_names_;
};

#endif
