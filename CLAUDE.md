# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a reinforcement learning deployment system for the Unitree Go2 quadruped robot. The project has two implementations:
- **Python**: `python/deploy_real.py` - Full-featured deployment with state machine
- **C++**: `cpp/` - Performance-critical controller implementation

Both implementations load PyTorch RL policies (encoder + policy models) and run inference to control the robot's 12 joint motors.

## Common Commands

### Python Deployment (Primary)
```bash
# Deploy with loopback interface (simulation/testing)
cd python && python deploy_real.py

# Deploy on real robot via network interface
cd python && python deploy_real.py <interface>  # e.g., python deploy_real.py eth0
```

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

### Python (`python/deploy_real.py`)
- `RLDeploy` class: Main deployment controller
- Uses Unitree SDK2 (DDS) for robot communication
- Three recurrent threads: command write (1ms), control loop (20ms), FSM (10ms)
- Remote controller input maps to velocity commands
- State machine: Stop → Move → Default Stand

### C++ (`cpp/`)
- `Controller` class: Main controller in `controller.cpp`
- Uses Unitree SDK2 with C++ bindings
- `Model` class: PyTorch model loading/inference (`model.cpp`)
- `obs_buf.cpp`: Observation history buffer for encoder
- Headers in `inc/` directory

### Shared Components
- `config/go2.yaml`: Robot and policy configuration
- `model/`: PyTorch policy (`policy_1.pt`) and encoder (`encoder_1.pt`) models
- `python/common/`: Python utilities
  - `remote_controller.py`: Gamepad/joystick input parsing
  - `rotation_helper.py`: IMU quaternion to gravity orientation conversion

### Robot Control Flow
1. Receive LowState (motor positions, velocities, IMU data)
2. Compute observation (normalized joint angles, velocities, angular velocity, gravity vector)
3. Pass observation + command through encoder to get latent
4. Pass observation + command + latent through policy to get action
5. Convert action to target joint positions
6. Send LowCmd with target positions and PD gains

## Key Files
- `python/deploy_real.py:160-206`: Main Forward() method - policy inference
- `cpp/src/controller.cpp:126-161`: C++ Forward() implementation
- `config/go2.yaml`: All tunable parameters
- `python/common/remote_controller.py:4-20`: Button/key mappings

## Dependencies
- Python: `unitree_sdk2py`, `torch`, `numpy`, `yaml`, `scipy`
- C++: `unitree_sdk2`, `libtorch`, `yaml-cpp`, `iceoryx`, `ddscxx`
