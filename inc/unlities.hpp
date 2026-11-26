#ifndef __UNLITIES_HPP__
#define __UNLITIES_HPP__

#include <cstdint>
#include <vector>

uint32_t crc32_core(uint32_t* ptr, uint32_t len);
std::vector<float> GetGravityOrientation(const std::vector<float>& q);

// Yaw [rad] in [-pi, pi] from a (w, x, y, z) quaternion.
float GetYaw(const std::vector<float>& q);

// Wraps an angle [rad] to [-pi, pi].
float Wrap2Pi(float angle);

#endif