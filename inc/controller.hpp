#ifndef __CONTROLLER_HPP__
#define __CONTROLLER_HPP__

#include <memory>
#include <string>
#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/idl/go2/WirelessController_.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/go2/sport/sport_client.hpp>
#include <vector>
#include "data_buffer.hpp"
#include "joystick.hpp"
#include "model.hpp"
#include "obs_buf.hpp"

constexpr double PosStopF = (2.146E+9f);
constexpr double VelStopF = (16000.0f);

template <typename T>
struct Obs {
  std::vector<T> gyro;
  std::vector<T> accel;
  std::vector<T> quat;
  std::vector<T> q;
  std::vector<T> dq;
  std::vector<T> actions;
};

using LowCmdPtr = std::shared_ptr<unitree_go::msg::dds_::LowCmd_>;

class Controller {
 public:
  explicit Controller(const std::string& config_path);
  ~Controller();

 private:
  void InitLowCmd();
  LowCmdPtr Damp();
  LowCmdPtr Forward();
  LowCmdPtr MoveToDefault();
  LowCmdPtr DefaultStand();
  void SetExecutedActions(const LowCmdPtr& cmd);
  void JoystickUpdate();
  void LowStateMsgHandler(const void* message);
  void JoystickMsgHandler(const void* message);
  void LowCmdWriteHandler();
  void FSMHandler();
  void LowCtrlHandler();
  int QueryMotionStatus();
  std::vector<float> ComputeObs();

 private:
  Model policy_, encoder_;
  DataBuffer<unitree_go::msg::dds_::LowCmd_> low_cmd_;
  DataBuffer<unitree_go::msg::dds_::LowState_> low_state_;
  DataBuffer<unitree_go::msg::dds_::WirelessController_> joystick_;
  unitree::robot::ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_>
      lowcmd_publisher_;
  unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_>
      lowstate_subscriber_;
  unitree::robot::ChannelSubscriberPtr<
      unitree_go::msg::dds_::WirelessController_>
      joystick_subscriber_;
  unitree::common::ThreadPtr low_cmd_write_thread_ptr_, fsm_thread_ptr_,
      low_ctrl_thread_ptr_;
  unitree::robot::b2::MotionSwitcherClient msc_;
  unitree::robot::go2::SportClient sc_;

  Obs<float> obs_;
  ObsBuf obs_history_buf_;
  Gamepad gamepad_;
  std::vector<int> motor_idx_;
  std::vector<float> kps_, kds_, init_pos_;
  std::vector<float> cmd_, obs_history_, max_cmd_, cmd_scale_;
  std::string policy_path_, encoder_path_;
  float ang_vel_scale_, dof_pos_scale_, dof_vel_scale_, action_scale_;
  int num_actions_, num_obs_, obs_history_len_;

  bool real_robot_ = false;
  bool move_ = false, ready_ = false, default_ = false, stop_ = false;

  // Owned by the low_ctrl thread only.
  int interp_step_ = 0;              // MoveToDefault progress, 0 = not started
  std::vector<float> interp_start_;  // joint positions when interpolation began
};

#endif