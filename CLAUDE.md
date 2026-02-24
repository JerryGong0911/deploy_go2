# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project deploys reinforcement learning models to control a Unitree Go2 quadruped robot. It implements a two-stage neural network architecture (encoder + policy) for locomotion control, with both Python and C++ implementations.

## Build and Run

### Python Deployment
```bash
# Local network deployment
python deploy_real.py

# Network interface specified (e.g., "eth0", "en0")
python deploy_real.py eth0
```

### C++ Deployment
```bash
# Build (from repo root)
cd cpp/build && cmake .. && make

# Or rebuild if build directory exists
cd cpp/build && make

# Run
./go2_deploy_cpp
```

### Dependencies
- Python: `torch`, `numpy`, `yaml`, `scipy`, `unitree_sdk2py`
- C++: `libtorch`, `unitree_sdk2`, `yaml-cpp`, `dds`/`ddscxx`

## Architecture

### Dual Implementation Structure

The project has two implementations that provide equivalent functionality:

1. **Python** (`deploy_real.py`, `common/`, `config.py`) - Reference implementation
2. **C++** (`cpp/src/`, `cpp/inc/`) - Production implementation

When modifying behavior, ensure both implementations stay aligned or document any intentional differences.

### Core Components

#### 1. Neural Network Inference
- **Encoder** (`model/encoder_1.pt`): Processes observation history into latent representation
- **Policy** (`model/policy_1.pt`): Maps current observation + latent + command to motor actions
- Both are PyTorch JIT models loaded at runtime

#### 2. Control Loop Architecture (C++)

Three recurrent threads running at different rates:

| Thread | Rate (Hz) | Purpose |
|--------|-----------|---------|
| `low_cmd_write` | 500 | Publish motor commands via DDS |
| `low_ctrl` | 50 | Execute FSM-selected motion mode |
| `fsm_handler` | 100 | Process joystick input and update FSM state |

#### 3. Observation Processing

Observation vector structure (42 dimensions):
- `0:2`: Angular velocity (gyro, scaled)
- `3:5`: Gravity orientation (from IMU quaternion)
- `6:17`: Joint positions (12 DOF, normalized relative to default)
- `18:29`: Joint velocities (12 DOF)
- `30:41`: Previous actions (12 DOF)

The `ObsBuf` class maintains a 5-step history buffer, which the encoder processes into a latent vector.

#### 4. FSM States

The controller operates in three modes controlled by gamepad buttons:

| Mode | Gamepad Trigger | Behavior |
|------|-----------------|----------|
| `stop` | R1 | Damping (low stiffness, high damping) |
| `default` | L2 | Move to default pose, then maintain |
| `move` | L1 + A (when ready) | RL policy inference and control |

C++ uses: `L2` to enter default mode, `A+L1` to enter move mode
Python uses: `R1` for stop, `L1+B` for move, `L1+A` for default

#### 5. DDS Communication Topics

- `rt/lowcmd`: Motor commands (published)
- `rt/lowstate`: IMU and motor state (subscribed)
- `rt/wirelesscontroller`: Gamepad input (subscribed)

### Key C++ Classes

- **`Controller`**: Main orchestration class, manages all threads and state
- **`Model`**: Wraps PyTorch JIT model loading and inference
- **`ObsBuf`**: Rolling buffer for observation history
- **`DataBuffer<T>`**: Thread-safe data exchange with atomic locks
- **`Gamepad`**: Joystick state with smoothing and deadzone handling
- **`Button`**: Tracks pressed/on_press/on_release states

### Configuration

`config/go2.yaml` contains:
- Model paths
- Joint-to-motor mapping (12 legs joints map to specific motor indices)
- PD gains (`kps`, `kds`)
- Default joint angles
- Observation/action/cmd scaling factors

## Motor Index Mapping

The robot has 20 motors total. The 12 leg joints are mapped via `leg_joint2motor_idx`:
- Left front: [3, 4, 5] (hip, thigh, calf)
- Right front: [0, 1, 2]
- Left rear: [9, 10, 11]
- Right rear: [6, 7, 8]

## Code Location Patterns

- C++ headers: `cpp/inc/*.hpp`
- C++ sources: `cpp/src/*.cpp`
- Python utilities: `common/*.py`
- Models: `model/*.pt`
- Config: `config/go2.yaml`

## Motion Switcher Integration (Optional)

Both implementations support motion switcher integration to release built-in Go2 motion control. This is currently commented out in the C++ code but present in the Python implementation when run with arguments.

When implementing changes that affect the motion pipeline:
- Python: Check `MotionSwitcherClient` usage in `__main__`
- C++: See commented code in `Controller` constructor related to `sc_` (SportClient) and `msc_` (MotionSwitcherClient)
