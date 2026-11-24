# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a reinforcement learning deployment system for the Unitree Go2 quadruped robot. This is the **C++** branch: `cpp/` contains the controller implementation. (The Python implementation lives on the `python` branch.)

It loads PyTorch RL policies (encoder + policy models) and runs inference to control the robot's 12 joint motors.

## Common Commands

### C++ Build
```bash
cd cpp/build
cmake ..
make -j$(nproc)
```

The C++ binary will be at `cpp/build/go2_deploy_cpp`.

### Configuration
Edit `config/go2.yaml` to modify:
- Model paths (`policy_path`, `encoder_path`)
- Motor indices (`leg_joint2motor_idx`)
- PID gains (`kps`, `kds`)
- Default joint angles (`default_angles`)
- Observation/action scaling factors

## Architecture

### C++ (`cpp/`)
- `Controller` class: Main controller in `controller.cpp`
- Uses Unitree SDK2 with C++ bindings
- `Model` class: PyTorch model loading/inference (`model.cpp`)
- `obs_buf.cpp`: Observation history buffer for encoder
- Headers in `inc/` directory

### Components
- `config/go2.yaml`: Robot and policy configuration
- `model/`: PyTorch policy (`policy_1.pt`) and encoder (`encoder_1.pt`) models

### Robot Control Flow
1. Receive LowState (motor positions, velocities, IMU data)
2. Compute observation (normalized joint angles, velocities, angular velocity, gravity vector)
3. Pass observation + command through encoder to get latent
4. Pass observation + command + latent through policy to get action
5. Convert action to target joint positions
6. Send LowCmd with target positions and PD gains

## Key Files
- `cpp/src/controller.cpp:126-161`: C++ Forward() implementation
- `config/go2.yaml`: All tunable parameters

## Dependencies
- C++: `unitree_sdk2`, `libtorch`, `yaml-cpp`, `iceoryx`, `ddscxx`
