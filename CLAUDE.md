# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a reinforcement learning deployment system for the Unitree Go2 quadruped robot. This is the **C++** branch: the controller implementation lives at the repository root (`src/`, `inc/`, `CMakeLists.txt`). (The Python implementation lives on the `python` branch.)

It loads ONNX RL policies (encoder + policy models) and runs CPU inference with ONNX Runtime to control the robot's 12 joint motors.

## Common Commands

### C++ Build
```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

The C++ binary will be at `build/go2_deploy_cpp`. Run it from `build/`: config and model paths are relative to that directory (`../config/go2.yaml`, `../model/`).

If librealsense2 is installed, `build/go2_depth_pub` is built too: a separate process that publishes RealSense depth (meters) on `depth.topic`. Start it alongside `go2_deploy_cpp` (also from `build/`) when `depth.enable: true`; it waits for the camera and reconnects after unplugging.

### Configuration
Edit `config/go2.yaml` to modify:
- Deployment target: `real_robot` (true: real Go2, DDS domain 0 on `network_interface`; false: unitree_mujoco, DDS domain `sim_domain_id` on `lo`)
- Model paths (`policy_path`, `encoder_path`)
- Motor indices (`leg_joint2motor_idx`)
- PID gains (`kps`, `kds`)
- Default joint angles (`default_angles`)
- Observation/action scaling factors
- `depth`: depth input (enable, topic, which network takes it, preprocessing). Real robot waits for the first frame; simulation feeds zeros until one arrives
- `obs_history_len`: observation frames stacked as encoder input (5 for the MLP encoder, typically 1 for a GRU encoder); checked against the encoder input size at startup

## Architecture

### C++ (`src/`, `inc/`)
- `Controller` class: Main controller in `controller.cpp`
- Uses Unitree SDK2 with C++ bindings
- `Model` class: ONNX Runtime model loading/inference, single-threaded CPU (`model.cpp`). Signature is M data inputs + S state inputs -> 1 + S outputs (S = #outputs - 1), e.g. MLP `(obs)->(out)`, GRU `(obs, h_in)->(out, h_out)`, depth+GRU `(obs, depth, h_in)->(out, h_out)`; states are kept inside `Model`, zero-initialised, and reset with `ResetStates()`
- `DepthProcessor` (`depth_processor.cpp`): depth image -> network input (scale, crop, clip, bilinear resize, normalize); depth arrives over DDS as `unitree_go::msg::dds_::HeightMap_` (used as a 2D float grid) and is preprocessed in the DDS callback
- `obs_buf.cpp`: Observation history buffer for encoder
- Headers in `inc/` directory

### Components
- `tools/depth_publisher.cpp`: standalone RealSense -> DDS `HeightMap_` depth publisher (`go2_depth_pub`), kept out of the controller process so camera I/O cannot stall the control loop; it only converts to meters, all training preprocessing is done by the controller
- `config/go2.yaml`: Robot and policy configuration
- `model/`: ONNX policy (`policy_1.onnx`) and encoder (`encoder_1.onnx`) models, exported from the TorchScript `.pt` files

### Robot Control Flow
1. Receive LowState (motor positions, velocities, IMU data)
2. Compute observation (normalized joint angles, velocities, angular velocity, gravity vector)
3. Pass observation history (+ depth if `depth.target: encoder`) through encoder to get latent
4. Pass observation + command + latent (+ depth if `depth.target: policy`) through policy to get action
5. Convert action to target joint positions
6. Send LowCmd with target positions and PD gains

## Key Files
- `src/controller.cpp:126-161`: C++ Forward() implementation
- `config/go2.yaml`: All tunable parameters

## Dependencies
- C++: `unitree_sdk2`, ONNX Runtime (prebuilt package in `library/thirdparty/onnxruntime` or `-DORT_DIR=...`; aarch64 build on Jetson), `yaml-cpp`, `iceoryx`, `ddscxx`
