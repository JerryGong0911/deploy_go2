#ifndef __UNLITIES_HPP__
#define __UNLITIES_HPP__

#include <cstdint>
#include <vector>

uint32_t crc32_core(uint32_t* ptr, uint32_t len);
std::vector<float> GetGravityOrientation(const std::vector<float>& q);

#endif