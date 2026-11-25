#ifndef CPP_INC_DEPTH_PROCESSOR_HPP_
#define CPP_INC_DEPTH_PROCESSOR_HPP_

#include <string>
#include <vector>

namespace YAML {
class Node;
}

// Turns a raw depth image into the network's depth input.
//
// Pipeline (must match training):
//   1. depth_scale:  raw value * depth_scale -> meters
//   2. crop:         drop [top, bottom, left, right] pixels
//   3. clip:         to [near_clip, far_clip]; invalid pixels (<= 0, NaN,
//                    inf: no return from the sensor) become far_clip
//   4. resize:       bilinear to out_height x out_width
//                    (align_corners = false, no antialias; same as
//                    torch.nn.functional.interpolate(mode="bilinear"))
//   5. normalize:    (d - near_clip) / (far_clip - near_clip)
//                    * norm_scale + norm_offset
// Output is row-major, out_height * out_width floats.
class DepthProcessor {
 public:
  struct Params {
    float depth_scale = 1.0f;
    int crop_top = 0, crop_bottom = 0, crop_left = 0, crop_right = 0;
    float near_clip = 0.0f, far_clip = 2.0f;
    int out_width = 0, out_height = 0;
    float norm_scale = 1.0f, norm_offset = 0.0f;
  };

  void Init(const Params& params);
  void Init(const YAML::Node& node);  // the "depth" block of go2.yaml
  const Params& params() const { return p_; }
  size_t OutputSize() const { return (size_t)p_.out_width * p_.out_height; }

  // Returns false (and sets *error) if the image cannot be processed,
  // e.g. wrong data size or the crop removes the whole image.
  bool Process(const float* data, int width, int height,
               std::vector<float>* out, std::string* error) const;

 private:
  Params p_;
};

#endif  // CPP_INC_DEPTH_PROCESSOR_HPP_
