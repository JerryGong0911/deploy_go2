#include "depth_processor.hpp"
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

void DepthProcessor::Init(const Params& params) {
  if (params.out_width <= 0 || params.out_height <= 0)
    throw std::runtime_error("depth: out_width / out_height must be > 0");
  if (!(params.far_clip > params.near_clip))
    throw std::runtime_error("depth: far_clip must be > near_clip");
  if (params.crop_top < 0 || params.crop_bottom < 0 || params.crop_left < 0 ||
      params.crop_right < 0)
    throw std::runtime_error("depth: crop values must be >= 0");
  p_ = params;
}

void DepthProcessor::Init(const YAML::Node& node) {
  Params p;
  p.depth_scale = node["depth_scale"].as<float>(1.0f);
  auto crop = node["crop"].as<std::vector<int>>(std::vector<int>{0, 0, 0, 0});
  if (crop.size() != 4)
    throw std::runtime_error("depth: crop must be [top, bottom, left, right]");
  p.crop_top = crop[0];
  p.crop_bottom = crop[1];
  p.crop_left = crop[2];
  p.crop_right = crop[3];
  p.near_clip = node["near_clip"].as<float>();
  p.far_clip = node["far_clip"].as<float>();
  p.out_width = node["width"].as<int>();
  p.out_height = node["height"].as<int>();
  p.norm_scale = node["norm_scale"].as<float>(1.0f);
  p.norm_offset = node["norm_offset"].as<float>(0.0f);
  Init(p);
}

bool DepthProcessor::Process(const float* data, int width, int height,
                             std::vector<float>* out,
                             std::string* error) const {
  if (width <= 0 || height <= 0) {
    *error = "empty depth image";
    return false;
  }
  const int cw = width - p_.crop_left - p_.crop_right;
  const int ch = height - p_.crop_top - p_.crop_bottom;
  if (cw <= 0 || ch <= 0) {
    *error = "crop removes the whole " + std::to_string(width) + "x" +
             std::to_string(height) + " image";
    return false;
  }

  // 1-3: scale, crop, clip.
  std::vector<float> clipped((size_t)cw * ch);
  for (int y = 0; y < ch; ++y) {
    const float* row = data + (size_t)(y + p_.crop_top) * width + p_.crop_left;
    for (int x = 0; x < cw; ++x) {
      float d = row[x] * p_.depth_scale;
      if (!std::isfinite(d) || d <= 0.0f)
        d = p_.far_clip;
      clipped[(size_t)y * cw + x] = std::clamp(d, p_.near_clip, p_.far_clip);
    }
  }

  // 4-5: bilinear resize (align_corners = false), normalize.
  const int ow = p_.out_width, oh = p_.out_height;
  const float sx = (float)cw / ow, sy = (float)ch / oh;
  const float inv_range = 1.0f / (p_.far_clip - p_.near_clip);
  out->resize((size_t)ow * oh);
  for (int y = 0; y < oh; ++y) {
    float fy = std::max((y + 0.5f) * sy - 0.5f, 0.0f);
    int y0 = std::min((int)fy, ch - 1);
    int y1 = std::min(y0 + 1, ch - 1);
    float wy = fy - y0;
    for (int x = 0; x < ow; ++x) {
      float fx = std::max((x + 0.5f) * sx - 0.5f, 0.0f);
      int x0 = std::min((int)fx, cw - 1);
      int x1 = std::min(x0 + 1, cw - 1);
      float wx = fx - x0;
      const float* r0 = &clipped[(size_t)y0 * cw];
      const float* r1 = &clipped[(size_t)y1 * cw];
      float d = (1 - wy) * ((1 - wx) * r0[x0] + wx * r0[x1]) +
                wy * ((1 - wx) * r1[x0] + wx * r1[x1]);
      (*out)[(size_t)y * ow + x] =
          (d - p_.near_clip) * inv_range * p_.norm_scale + p_.norm_offset;
    }
  }
  return true;
}
