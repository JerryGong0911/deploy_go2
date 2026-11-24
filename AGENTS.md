# Repository Guidelines

## Project Structure & Module Organization
This repository deploys RL locomotion policies for Unitree Go2 with Python and C++ implementations.

- `python/`: primary runtime (`deploy_real.py`) plus helpers in `python/common/`.
- `cpp/`: C++ runtime (`src/`, `inc/`, `CMakeLists.txt`) producing `go2_deploy_cpp`.
- `config/go2.yaml`: shared runtime configuration (model paths, gains, scaling, topics).
- `model/`: TorchScript artifacts (`policy_1.pt`, `encoder_1.pt`).
- `CLAUDE.md`: architecture and quick build/run notes for contributors.

Keep behavior-aligned changes in both `python/` and `cpp/` when updating control logic.

## Build, Test, and Development Commands
- `cd python && python deploy_real.py`: run deployment in local/loopback mode.
- `cd python && python deploy_real.py eth0`: run deployment against robot network interface.
- `mkdir -p cpp/build && cd cpp/build && cmake .. && make -j$(nproc)`: configure and build C++ binary.
- `cd cpp/build && ./go2_deploy_cpp`: run the C++ deployment executable.

When changing configuration-sensitive logic, validate with `config/go2.yaml` values used in your run.

## Coding Style & Naming Conventions
- C++ style follows `.clang-format` (`BasedOnStyle: Chromium`); format C++ changes before review.
- Python uses PEP 8 style with 4-space indentation and descriptive `snake_case` names.
- Class names use `PascalCase` (for example, `RLDeploy`, `Config`).
- Keep file/module names lowercase with underscores (`remote_controller.py`, `obs_buf.cpp`).

## Testing Guidelines
There is no automated test suite configured yet (`pytest`/`ctest` not present). Use targeted smoke tests:

1. Build C++ cleanly in `cpp/build`.
2. Start Python deployment and verify model load + DDS initialization.
3. Verify control-loop safety paths (stop/default posture) before motion commands.

For logic-heavy changes, add small deterministic checks where possible and document manual validation steps in your PR.

## Commit & Pull Request Guidelines
Recent history uses Conventional Commit-style prefixes: `feat:`, `fix:`, `docs:`, `refactor:`. Follow `type: short imperative summary`.

PRs should include:
- what changed and why,
- affected paths (for example, `python/deploy_real.py`, `config/go2.yaml`),
- runtime impact (control behavior, gains, model/config compatibility),
- brief validation evidence (build/run commands executed, environment/interface used).
