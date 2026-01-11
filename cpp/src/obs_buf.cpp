#include "obs_buf.hpp"
#include <algorithm>

void ObsBuf::Init(int _history_len, int _num_obs) {
  history_len_ = _history_len;
  num_obs_ = _num_obs;
  data_.resize(history_len_, std::vector<float>(num_obs_, 0.0f));
}

void ObsBuf::Reset(const std::vector<float>& obs) {
  data_.assign(history_len_, obs);
}

void ObsBuf::Insert(const std::vector<float>& obs) {
  if (data_.empty())
    return;
  std::rotate(data_.begin(), data_.begin() + 1, data_.end());
  data_.back() = obs;
}

std::vector<float> ObsBuf::GetFlattenedData() const {
  std::vector<float> flat;
  flat.reserve(history_len_ * num_obs_);
  for (const auto& row : data_) {
    flat.insert(flat.end(), row.begin(), row.end());
  }
  return flat;
}
