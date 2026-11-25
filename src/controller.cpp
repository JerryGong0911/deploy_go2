#include "controller.hpp"
#include <unistd.h>
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include "unlities.hpp"

#define TOPIC_LOWCMD "rt/lowcmd"
#define TOPIC_LOWSTATE "rt/lowstate"
#define TOPIC_JOYSTICK "rt/wirelesscontroller"

namespace {
// Rate limit for warnings raised inside the control / DDS loops.
bool ShouldWarn(std::chrono::steady_clock::time_point* last) {
  auto now = std::chrono::steady_clock::now();
  if (now - *last < std::chrono::seconds(1))
    return false;
  *last = now;
  return true;
}
}  // namespace

Controller::Controller(const std::string& config_path) {
  YAML::Node yaml_node = YAML::LoadFile(config_path);

  real_robot_ = yaml_node["real_robot"].as<bool>();
  policy_path_ = yaml_node["policy_path"].as<std::string>();
  encoder_path_ = yaml_node["encoder_path"].as<std::string>();

  motor_idx_ = yaml_node["leg_joint2motor_idx"].as<std::vector<int>>();
  kps_ = yaml_node["kps"].as<std::vector<float>>();
  kds_ = yaml_node["kds"].as<std::vector<float>>();
  init_pos_ = yaml_node["default_angles"].as<std::vector<float>>();
  max_cmd_ = yaml_node["max_cmd_slope"].as<std::vector<float>>();

  cmd_scale_ = yaml_node["cmd_scale"].as<std::vector<float>>();
  ang_vel_scale_ = yaml_node["ang_vel_scale"].as<float>();
  dof_pos_scale_ = yaml_node["dof_pos_scale"].as<float>();
  dof_vel_scale_ = yaml_node["dof_vel_scale"].as<float>();
  action_scale_ = yaml_node["action_scale"].as<float>();

  num_actions_ = yaml_node["num_actions"].as<int>();
  num_obs_ = yaml_node["num_obs"].as<int>();
  // Frames of observation history fed to the encoder. MLP encoders use a
  // stacked window (5); a recurrent (GRU) encoder usually takes 1 frame and
  // carries the past in its hidden state.
  obs_history_len_ = yaml_node["obs_history_len"].as<int>(5);
  cmd_.assign(3, 0.0f);
  obs_.gyro.assign(3, 0.0f);
  obs_.accel.assign(3, 0.0f);
  obs_.q.assign(num_actions_, 0.0f);
  obs_.dq.assign(num_actions_, 0.0f);
  obs_.actions.assign(num_actions_, 0.0f);
  interp_start_.assign(motor_idx_.size(), 0.0f);

  InitLowCmd();
  obs_history_buf_.Init(obs_history_len_, num_obs_);
  policy_.Init(policy_path_);
  encoder_.Init(encoder_path_);
  InitDepth(yaml_node["depth"]);
  CheckModelInputs();

  lowcmd_publisher_.reset(
      new unitree::robot::ChannelPublisher<unitree_go::msg::dds_::LowCmd_>(
          TOPIC_LOWCMD));
  lowstate_subscriber_.reset(
      new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::LowState_>(
          TOPIC_LOWSTATE));
  joystick_subscriber_.reset(
      new unitree::robot::ChannelSubscriber<
          unitree_go::msg::dds_::WirelessController_>(TOPIC_JOYSTICK));
  lowcmd_publisher_->InitChannel();
  lowstate_subscriber_->InitChannel(
      std::bind(&Controller::LowStateMsgHandler, this, std::placeholders::_1),
      1);
  joystick_subscriber_->InitChannel(
      std::bind(&Controller::JoystickMsgHandler, this, std::placeholders::_1),
      1);
  if (depth_enable_) {
    depth_subscriber_.reset(new unitree::robot::ChannelSubscriber<
                            unitree_go::msg::dds_::HeightMap_>(depth_topic_));
    depth_subscriber_->InitChannel(
        std::bind(&Controller::DepthMsgHandler, this, std::placeholders::_1),
        1);
  }

  // The sport / motion-switcher services only exist on the real robot.
  if (real_robot_) {
    this->sc_.SetTimeout(10.0f);
    this->sc_.Init();
    this->msc_.SetTimeout(10.0f);
    this->msc_.Init();
  }

  // Wait for LowState only; the joystick is optional (e.g. sim without pad).
  std::cout << "Waiting for LowState..." << std::endl;
  while (!low_state_.GetDataPtr()) {
    usleep(100000);
  }
  std::cout << "Get LowState Succeeded." << std::endl;

  // On the real robot never start without depth. In simulation nothing may
  // publish depth, so the network gets zeros until a frame arrives.
  if (depth_enable_ && real_robot_) {
    std::cout << "Waiting for depth image on " << depth_topic_ << "..."
              << std::endl;
    while (!depth_frame_.GetDataPtr()) {
      usleep(100000);
    }
    std::cout << "Get depth image Succeeded." << std::endl;
  } else if (depth_enable_) {
    std::cout << "Simulation: depth input is zero until a frame arrives on "
              << depth_topic_ << std::endl;
  }

  // Release the built-in locomotion controller so it does not fight lowcmd.
  while (real_robot_ && this->QueryMotionStatus() == 1) {
    std::cout << "Try to deactivate the motion control-related service."
              << std::endl;
    this->sc_.StandDown();
    int32_t ret = this->msc_.ReleaseMode();
    if (ret == 0) {
      std::cout << "ReleaseMode succeeded." << std::endl;
    } else {
      std::cout << "ReleaseMode failed. Error code: " << ret << std::endl;
    }
    sleep(1);
  }

  low_cmd_write_thread_ptr_ = unitree::common::CreateRecurrentThreadEx(
      "low_cmd_write", UT_CPU_ID_NONE, 2000, &Controller::LowCmdWriteHandler,
      this);
  fsm_thread_ptr_ = unitree::common::CreateRecurrentThreadEx(
      "fsm_handler", UT_CPU_ID_NONE, 10000, &Controller::FSMHandler, this);
  low_ctrl_thread_ptr_ = unitree::common::CreateRecurrentThreadEx(
      "low_ctrl", UT_CPU_ID_NONE, 20000, &Controller::LowCtrlHandler, this);

  std::cout << "Controller initialized." << std::endl;
}

Controller::~Controller() {
  // Damp();
  std::cout << "Controller Done" << std::endl;
}

void Controller::InitLowCmd() {
  // Implementation of low command initialization
  auto low_cmd = std::make_shared<unitree_go::msg::dds_::LowCmd_>();

  for (int i = 0; i < 20; ++i) {
    low_cmd->motor_cmd()[i].mode() = (0x01);
    low_cmd->motor_cmd()[i].q() = (PosStopF);
    low_cmd->motor_cmd()[i].kp() = 0.0f;
    low_cmd->motor_cmd()[i].dq() = (VelStopF);
    low_cmd->motor_cmd()[i].kd() = 0.0f;
    low_cmd->motor_cmd()[i].tau() = 0.0f;
  }
  low_cmd_.SetDataPtr(low_cmd);
}

LowCmdPtr Controller::Damp() {
  auto low_cmd = std::make_shared<unitree_go::msg::dds_::LowCmd_>();

  for (int i = 0; i < 20; ++i) {
    low_cmd->motor_cmd()[i].q() = 0.0f;
    low_cmd->motor_cmd()[i].kp() = 0.0f;
    low_cmd->motor_cmd()[i].dq() = 0.0f;
    low_cmd->motor_cmd()[i].kd() = 3.0f;
    low_cmd->motor_cmd()[i].tau() = 0.0f;
  }
  return low_cmd;
}

// The policy runs every step (shadow mode) to keep the observation history,
// latent and any recurrent (GRU) hidden state current. When its output is NOT
// executed, overwrite the "last action" fed back into the observation with the
// action equivalent of the command that was actually sent, so the history stays
// self-consistent. Joints without a position target (kp == 0, e.g. damping) map
// to 0.
void Controller::SetExecutedActions(const LowCmdPtr& cmd) {
  for (size_t i = 0; i < motor_idx_.size(); ++i) {
    const auto& m = cmd->motor_cmd()[motor_idx_[i]];
    obs_.actions[i] =
        m.kp() > 0.0f ? (m.q() - init_pos_[i]) / action_scale_ : 0.0f;
  }
}

LowCmdPtr Controller::Forward() {
  auto low_cmd = std::make_shared<unitree_go::msg::dds_::LowCmd_>();
  auto low_state = low_state_.GetDataPtr();

  for (int i = 0; i < motor_idx_.size(); ++i) {
    obs_.q[i] = low_state->motor_state()[motor_idx_[i]].q();
    obs_.dq[i] = low_state->motor_state()[motor_idx_[i]].dq();
  }
  for (int i = 0; i < 3; ++i)
    obs_.gyro[i] = low_state->imu_state().gyroscope()[i];

  obs_.quat.assign(low_state->imu_state().quaternion().begin(),
                   low_state->imu_state().quaternion().end());

  std::vector<float> obs_vector = ComputeObs();
  obs_history_buf_.Insert(obs_vector);
  obs_history_ = obs_history_buf_.GetFlattenedData();

  cmd_[0] = gamepad_.ly * max_cmd_[0] * cmd_scale_[0];
  cmd_[1] = -gamepad_.lx * max_cmd_[1] * cmd_scale_[1];
  cmd_[2] = -gamepad_.rx * max_cmd_[2] * cmd_scale_[2];

  std::vector<std::vector<float>> encoder_in{obs_history_};
  if (depth_enable_) {
    UpdateDepthInput();
    if (depth_to_encoder_)
      encoder_in.push_back(depth_input_);
  }
  auto latent = encoder_.Forward(encoder_in);
  obs_vector.insert(obs_vector.end(), cmd_.begin(), cmd_.end());
  obs_vector.insert(obs_vector.end(), latent.begin(), latent.end());
  std::vector<std::vector<float>> policy_in{obs_vector};
  if (depth_enable_ && !depth_to_encoder_)
    policy_in.push_back(depth_input_);
  obs_.actions = policy_.Forward(policy_in);

  for (int i = 0; i < motor_idx_.size(); ++i) {
    low_cmd->motor_cmd()[motor_idx_[i]].q() =
        obs_.actions[i] * action_scale_ + init_pos_[i];
    low_cmd->motor_cmd()[motor_idx_[i]].kp() = kps_[i];
    low_cmd->motor_cmd()[motor_idx_[i]].dq() = 0.0f;
    low_cmd->motor_cmd()[motor_idx_[i]].kd() = kds_[i];
    low_cmd->motor_cmd()[motor_idx_[i]].tau() = 0.0f;
  }
  return low_cmd;
}

// Linear interpolation from the joint positions at the start of the move to
// the default pose over 2 s. interp_step_ is reset by LowCtrlHandler whenever
// another mode runs, so every new request starts from the current pose.
LowCmdPtr Controller::MoveToDefault() {
  static const int kInterpSteps = (int)(2.0f / 0.02f);

  if (interp_step_ >= kInterpSteps) {
    interp_step_ = 0;
    ready_ = true;
    return DefaultStand();  // never return an unfilled (zero-gain) command
  }

  if (interp_step_ == 0) {
    auto low_state = low_state_.GetDataPtr();
    for (size_t i = 0; i < motor_idx_.size(); ++i)
      interp_start_[i] = low_state->motor_state()[motor_idx_[i]].q();
  }

  ++interp_step_;
  float alpha = (float)interp_step_ / (float)kInterpSteps;  // (0, 1]

  auto low_cmd = std::make_shared<unitree_go::msg::dds_::LowCmd_>();
  for (size_t i = 0; i < motor_idx_.size(); ++i) {
    low_cmd->motor_cmd()[motor_idx_[i]].q() =
        (1 - alpha) * interp_start_[i] + alpha * init_pos_[i];
    low_cmd->motor_cmd()[motor_idx_[i]].kp() = 40.f;
    low_cmd->motor_cmd()[motor_idx_[i]].dq() = 0.0f;
    low_cmd->motor_cmd()[motor_idx_[i]].kd() = 1.f;
    low_cmd->motor_cmd()[motor_idx_[i]].tau() = 0.0f;
  }
  return low_cmd;
}

LowCmdPtr Controller::DefaultStand() {
  auto low_cmd = std::make_shared<unitree_go::msg::dds_::LowCmd_>();
  for (int i = 0; i < motor_idx_.size(); ++i) {
    low_cmd->motor_cmd()[motor_idx_[i]].q() = init_pos_[i];
    low_cmd->motor_cmd()[motor_idx_[i]].kp() = 40.0f;
    low_cmd->motor_cmd()[motor_idx_[i]].dq() = 0.0f;
    low_cmd->motor_cmd()[motor_idx_[i]].kd() = 0.6f;
    low_cmd->motor_cmd()[motor_idx_[i]].tau() = 0.0f;
  }
  return low_cmd;
}

void Controller::FSMHandler() {
  JoystickUpdate();

  if (gamepad_.L2.pressed) {
    default_ = true;
  } else if (gamepad_.R2.pressed) {
    default_ = false;
    ready_ = false;
  } else if (gamepad_.A.pressed && gamepad_.L1.pressed) {
    if (ready_) {
      move_ = true;
      default_ = false;
    }
  }
}

void Controller::LowCtrlHandler() {
  // No state yet: keep publishing the current (initial) command.
  if (!low_state_.GetDataPtr())
    return;

  // Snapshot the flags once so this step sees a consistent mode.
  const bool is_default = default_, is_move = move_, is_ready = ready_;

  // Run the policy every step so history/latent are current at hand-over.
  LowCmdPtr policy_cmd = Forward();

  LowCmdPtr target_cmd;
  bool interpolating = false;
  if (is_default) {
    if (is_ready) {
      target_cmd = DefaultStand();
    } else {
      target_cmd = MoveToDefault();
      interpolating = true;
    }
  } else if (is_move && is_ready) {
    target_cmd = policy_cmd;
  } else {
    target_cmd = Damp();
  }

  if (target_cmd != policy_cmd)
    SetExecutedActions(target_cmd);  // policy output was not executed
  if (!interpolating)
    interp_step_ = 0;  // next MoveToDefault starts fresh from the current pose
  low_cmd_.SetDataPtr(target_cmd);
}

void Controller::LowStateMsgHandler(const void* message) {
  unitree_go::msg::dds_::LowState_* ptr =
      (unitree_go::msg::dds_::LowState_*)message;
  low_state_.SetData(*ptr);
}

void Controller::JoystickMsgHandler(const void* message) {
  unitree_go::msg::dds_::WirelessController_* ptr =
      (unitree_go::msg::dds_::WirelessController_*)message;
  joystick_.SetData(*ptr);
}

void Controller::InitDepth(const YAML::Node& node) {
  depth_enable_ = node && node["enable"].as<bool>(false);
  if (!depth_enable_) {
    std::cout << "Depth input: disabled" << std::endl;
    return;
  }
  depth_topic_ = node["topic"].as<std::string>();
  const auto target = node["target"].as<std::string>("policy");
  if (target != "policy" && target != "encoder")
    throw std::runtime_error(
        "depth.target must be 'policy' or 'encoder', got '" + target + "'");
  depth_to_encoder_ = target == "encoder";
  depth_update_interval_ = node["update_interval"].as<int>(1);
  if (depth_update_interval_ < 1)
    throw std::runtime_error("depth.update_interval must be >= 1");
  depth_timeout_ = node["timeout"].as<float>(0.5f);
  depth_proc_.Init(node);
  depth_input_.assign(depth_proc_.OutputSize(), 0.0f);

  const auto& p = depth_proc_.params();
  std::cout << "Depth input: topic " << depth_topic_ << " -> " << target << ", "
            << p.out_height << "x" << p.out_width << ", clip [" << p.near_clip
            << ", " << p.far_clip << "] m, every " << depth_update_interval_
            << " step(s)" << std::endl;
}

// Fail at startup, not in the control loop, if the models and the config
// disagree about which network takes which inputs.
void Controller::CheckModelInputs() {
  auto expect_inputs = [](const Model& m, const char* name, size_t n) {
    if (m.NumInputs() != n)
      throw std::runtime_error(
          std::string(name) + " model has " + std::to_string(m.NumInputs()) +
          " data inputs, config expects " + std::to_string(n) +
          " (check depth.enable / depth.target)");
  };
  expect_inputs(encoder_, "encoder",
                depth_enable_ && depth_to_encoder_ ? 2 : 1);
  expect_inputs(policy_, "policy", depth_enable_ && !depth_to_encoder_ ? 2 : 1);

  if (encoder_.InputSize(0) != (size_t)(obs_history_len_ * num_obs_))
    throw std::runtime_error(
        "encoder input size " + std::to_string(encoder_.InputSize(0)) +
        " != obs_history_len (" + std::to_string(obs_history_len_) +
        ") * num_obs (" + std::to_string(num_obs_) + ")");

  if (depth_enable_) {
    const Model& m = depth_to_encoder_ ? encoder_ : policy_;
    if (m.InputSize(1) != depth_proc_.OutputSize())
      throw std::runtime_error("depth input size " +
                               std::to_string(m.InputSize(1)) +
                               " != depth.height * depth.width (" +
                               std::to_string(depth_proc_.OutputSize()) + ")");
  }
}

// DDS thread: preprocess here so the control loop only copies the result.
void Controller::DepthMsgHandler(const void* message) {
  const auto* msg = (const unitree_go::msg::dds_::HeightMap_*)message;
  std::string err;
  auto frame = std::make_shared<DepthFrame>();
  if (msg->data().size() != (size_t)msg->width() * msg->height()) {
    err = "data size " + std::to_string(msg->data().size()) +
          " != width * height (" + std::to_string(msg->width()) + "x" +
          std::to_string(msg->height()) + ")";
  } else {
    depth_proc_.Process(msg->data().data(), msg->width(), msg->height(),
                        &frame->data, &err);
  }
  if (!err.empty()) {
    if (ShouldWarn(&depth_err_warn_))
      std::cerr << "Dropping depth image: " << err << std::endl;
    return;
  }
  frame->received = std::chrono::steady_clock::now();
  depth_frame_.SetDataPtr(frame);
}

// low_ctrl thread: refresh the held depth input every update_interval steps.
// Before the first frame arrives (simulation) the input stays zero.
void Controller::UpdateDepthInput() {
  auto frame = depth_frame_.GetDataPtr();
  if (frame && real_robot_) {
    float age = std::chrono::duration<float>(std::chrono::steady_clock::now() -
                                             frame->received)
                    .count();
    if (age > depth_timeout_ && ShouldWarn(&depth_stale_warn_))
      std::cerr << "WARNING: depth image is " << age << " s old" << std::endl;
  }
  const bool refresh = depth_step_ == 0;
  depth_step_ = (depth_step_ + 1) % depth_update_interval_;
  if (refresh && frame)
    depth_input_ = frame->data;
}

void Controller::JoystickUpdate() {
  auto joystick_ptr = joystick_.GetDataPtr();
  xRockerBtnDataStruct key_data;

  if (joystick_ptr == nullptr)
    return;

  key_data.lx = joystick_ptr->lx();
  key_data.rx = joystick_ptr->rx();
  key_data.ry = joystick_ptr->ry();
  key_data.ly = joystick_ptr->ly();
  key_data.btn.value = joystick_ptr->keys();
  gamepad_.update(key_data);
}

void Controller::LowCmdWriteHandler() {
  if (auto low_cmd_ptr = low_cmd_.GetDataPtr()) {
    low_cmd_ptr->head()[0] = 0xFE;
    low_cmd_ptr->head()[1] = 0xEF;
    low_cmd_ptr->level_flag() = 0xFF;
    low_cmd_ptr->gpio() = 0;

    for (auto& cmd : low_cmd_ptr->motor_cmd()) {
      cmd.mode() = 0x01;
    }

    low_cmd_ptr->crc() =
        crc32_core((uint32_t*)(low_cmd_ptr.get()),
                   (sizeof(unitree_go::msg::dds_::LowCmd_) >> 2) - 1);
    lowcmd_publisher_->Write(*low_cmd_ptr);
  }
}

int Controller::QueryMotionStatus() {
  std::string robotForm, motionName;
  int motionStatus;
  int32_t ret = msc_.CheckMode(robotForm, motionName);
  if (ret == 0) {
    std::cout << "CheckMode succeeded." << std::endl;
  } else {
    std::cout << "CheckMode failed. Error code: " << ret << std::endl;
  }
  if (motionName.empty()) {
    std::cout << "The motion control-related service is deactivated."
              << std::endl;
    motionStatus = 0;
  } else {
    std::cout << "Service is activate" << std::endl;
    motionStatus = 1;
  }
  return motionStatus;
}

std::vector<float> Controller::ComputeObs() {
  std::vector<float> obs(num_obs_);

  for (int i = 0; i < motor_idx_.size(); ++i) {
    obs_.q[i] = dof_pos_scale_ * (obs_.q[i] - init_pos_[i]);
    obs_.dq[i] = dof_vel_scale_ * obs_.dq[i];
  }
  for (int i = 0; i < 3; ++i)
    obs_.gyro[i] = ang_vel_scale_ * obs_.gyro[i];
  obs_.accel = GetGravityOrientation(obs_.quat);

  std::copy(obs_.gyro.begin(), obs_.gyro.end(), obs.begin());
  std::copy(obs_.accel.begin(), obs_.accel.end(), obs.begin() + 3);
  std::copy(obs_.q.begin(), obs_.q.end(), obs.begin() + 6);
  std::copy(obs_.dq.begin(), obs_.dq.end(), obs.begin() + 6 + num_actions_);
  std::copy(obs_.actions.begin(), obs_.actions.end(),
            obs.begin() + 6 + 2 * num_actions_);

  return obs;
}