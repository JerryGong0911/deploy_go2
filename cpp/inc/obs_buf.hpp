#ifndef CPP_INC_OBS_BUF_HPP_
#define CPP_INC_OBS_BUF_HPP_

#include <vector>

class ObsBuf {
 public:
  ObsBuf() = default;
  ~ObsBuf() = default;

  void Init(int _history_len, int _num_obs);
  void Reset(const std::vector<float>& obs);
  void Insert(const std::vector<float>& obs);
  std::vector<float> GetFlattenedData() const;

 private:
  int history_len_, num_obs_;
  std::vector<int> obs_dims_;
  std::vector<std::vector<float>> data_;
};

#endif  // CPP_INC_OBS_BUF_HPP_
