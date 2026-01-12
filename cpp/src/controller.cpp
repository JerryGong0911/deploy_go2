#include "controller.hpp"
#include <yaml-cpp/yaml.h>
#include <string>
#include "unlities.hpp"

#define TOPIC_LOWCMD "rt/lowcmd"
#define TOPIC_LOWSTATE "rt/lowstate"
#define TOPIC_JOYSTICK "rt/wirelesscontroller"

Controller::Controller() : obs_history_(5, num_obs_) {
  YAML::Node yaml_node = YAML::LoadFile("../../config/go2.yaml");

  policy_path_ = yaml_node["policy_path"].as<std::string>();
  encoder_path_ = yaml_node["encoder_path"].as<std::string>();

  motor_idx_ = yaml_node["leg_joint2motor_idx"].as<std::vector<int>>();
  kps_ = yaml_node["kps"].as<std::vector<float>>();
  kds_ = yaml_node["kds"].as<std::vector<float>>();
  init_pos_ = yaml_node["default_angles"].as<std::vector<float>>();
  max_cmd_ = yaml_node["max_cmd"].as<std::vector<float>>();

  cmd_scale_ = yaml_node["cmd_scale"].as<std::vector<float>>();
  ang_vel_scale_ = yaml_node["ang_vel_scale"].as<float>();
  dof_pos_scale_ = yaml_node["dof_pos_scale"].as<float>();
  dof_vel_scale_ = yaml_node["dof_vel_scale"].as<float>();
  action_scale_ = yaml_node["action_scale"].as<float>();

  num_actions_ = yaml_node["num_actions"].as<int>();
  num_obs_ = yaml_node["num_obs"].as<int>();

  InitLowCmd();
  obs_history_buf_.Init(5, num_obs_);
  policy_.Init(policy_path_);
  encoder_.Init(encoder_path_);

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

  this->sc_.SetTimeout(10.0f);
  this->sc_.Init();
  this->msc_.SetTimeout(10.0f);
  this->msc_.Init();

  while (!low_state_.GetDataPtr() && !joystick_.GetDataPtr()) {
    usleep(100000);
  }
  std::cout << "Get LowState Succeeded." << std::endl;

  while (this->QueryMotionStatus() == 1) {
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

auto Controller::Damp() {
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

auto Controller::Forward() {
  auto low_cmd = std::make_shared<unitree_go::msg::dds_::LowCmd_>();

  for (int i = 0; i < motor_idx_.size(); ++i) {
    low_cmd->motor_cmd()[motor_idx_[i]].q() =
        actions_[i] * action_scale_ + init_pos_[i];
    low_cmd->motor_cmd()[motor_idx_[i]].kp() = kps_[i];
    low_cmd->motor_cmd()[motor_idx_[i]].dq() = 0.0f;
    low_cmd->motor_cmd()[motor_idx_[i]].kd() = kds_[i];
    low_cmd->motor_cmd()[motor_idx_[i]].tau() = 0.0f;
  }
  return low_cmd;
}

auto Controller::MoveToDefault() {
  static int total_steps = 0;
  static const int max_steps = (int)(2.0f / 0.02f);
  auto low_cmd = std::make_shared<unitree_go::msg::dds_::LowCmd_>();

  auto low_state = low_state_.GetDataPtr();
  std::array<float, 12> current_pos;
  for (int i = 0; i < motor_idx_.size(); ++i) {
    current_pos[i] = low_state->motor_state()[motor_idx_[i]].q();
  }

  if (total_steps < max_steps && !ready_) {
    float alpha = (float)(total_steps) / (float)(max_steps);
    for (int i = 0; i < motor_idx_.size(); ++i) {
      low_cmd->motor_cmd()[motor_idx_[i]].q() =
          (1 - alpha) * current_pos[i] + alpha * init_pos_[i];
      low_cmd->motor_cmd()[motor_idx_[i]].kp() = 40.f;
      low_cmd->motor_cmd()[motor_idx_[i]].dq() = 0.0f;
      low_cmd->motor_cmd()[motor_idx_[i]].kd() = 1.f;
      low_cmd->motor_cmd()[motor_idx_[i]].tau() = 0.0f;
    }
    total_steps++;
  } else {
    total_steps = 0;
    ready_ = true;
  }
  return low_cmd;
}

auto Controller::DefaultStand() {
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
  }
}

void Controller::LowCtrlHandler() {
  std::shared_ptr<unitree_go::msg::dds_::LowCmd_> target_cmd;
  auto forward_cmd = Forward();
  if (default_) {
    target_cmd = (ready_ ? DefaultStand() : MoveToDefault());
  } else if (move_ && ready_) {
    target_cmd = forward_cmd;
  } else {
    target_cmd = Damp();
  }
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
