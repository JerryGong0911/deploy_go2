# Repository Guidelines

## Project Structure & Module Organization
This repository deploys RL locomotion policies for Unitree Go2 with a C++ implementation (the Python implementation lives on the `python` branch).

- `src/`, `inc/`, `CMakeLists.txt`: C++ runtime producing `go2_deploy_cpp`.
- `config/go2.yaml`: shared runtime configuration (model paths, gains, scaling, topics).
- `model/`: TorchScript artifacts (`policy_1.pt`, `encoder_1.pt`).
- `CLAUDE.md`: architecture and quick build/run notes for contributors.

When control logic changes here, consider whether the `python` branch needs the same change.

## Build, Test, and Development Commands
- `mkdir -p build && cd build && cmake .. && make -j$(nproc)`: configure and build C++ binary.
- `cd build && ./go2_deploy_cpp`: run the C++ deployment executable.

When changing configuration-sensitive logic, validate with `config/go2.yaml` values used in your run.

## Coding Style & Naming Conventions
- C++ style follows `.clang-format` (`BasedOnStyle: Chromium`); format C++ changes before review.
- Class names use `PascalCase` (for example, `Controller`, `Model`).
- Keep file/module names lowercase with underscores (`obs_buf.cpp`, `data_buffer.hpp`).

## Testing Guidelines
There is no automated test suite configured yet (`ctest` not present). Use targeted smoke tests:

1. Build C++ cleanly in `build/`.
2. Run `go2_deploy_cpp` and verify model load + DDS initialization.
3. Verify control-loop safety paths (stop/default posture) before motion commands.

For logic-heavy changes, add small deterministic checks where possible and document manual validation steps in your PR.

## Commit & Pull Request Guidelines
Recent history uses Conventional Commit-style prefixes: `feat:`, `fix:`, `docs:`, `refactor:`. Follow `type: short imperative summary`.

PRs should include:
- what changed and why,
- affected paths (for example, `src/controller.cpp`, `config/go2.yaml`),
- runtime impact (control behavior, gains, model/config compatibility),
- brief validation evidence (build/run commands executed, environment/interface used).
