#ifndef __MODEL_HPP__
#define __MODEL_HPP__

#include <torch/script.h>
#include <memory>
#include <string>
#include <vector>

class Model {
 public:
  Model();
  ~Model() = default;

 public:
  void Init(const std::string& _model_path);
  bool IsLoaded() { return loaded_; }
  std::vector<float> Forward(const std::vector<std::vector<float>>& input_vec);

 private:
  std::vector<float> Torch2Vec(const torch::Tensor& input);

 private:
  bool loaded_;
  std::string model_path_;
  torch::jit::script::Module model_;
};

#endif