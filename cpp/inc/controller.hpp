#ifndef __CONTROLLER_HPP__
#define __CONTROLLER_HPP__

#include <string>
#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <unitree/robot/go2/sport/sport_client.hpp>
#include <vector>
#include "data_buffer.hpp"
#include "model.hpp"
#include "joystick.hpp"

constexpr double PosStopF = (2.146E+9f);
constexpr double VelStopF = (16000.0f);

template <typename T>
struct Obs {
  std::vector<T> gyro;
  std::vector<T> accel;
  std::vector<T> q;
  std::vector<T> dq;
  std::vector<T> actions;
};

class Controller {
 public:
  Controller();
  ~Controller();

 private:
  void InitLowCmd();
  void Damp();
  void Forward();
  void MoveToDefault();
  void DefaultStand();
  void LowStateMsgHandler(const void* message);
  void LowCmdWriteHandler();
  void FSMHandler();
  void LowCtrlHandler();
  int QueryMotionStatus();

 private:
  //   Model policy_, encoder_;
  DataBuffer<unitree_go::msg::dds_::LowCmd_> low_cmd_;
  DataBuffer<unitree_go::msg::dds_::LowState_> low_state_;
  unitree::robot::ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_>
      lowcmd_publisher_;
  unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_>
      lowstate_subscriber_;
  unitree::common::ThreadPtr low_cmd_write_thread_ptr_, fsm_thread_ptr_,
      low_ctrl_thread_ptr_;
  unitree::robot::b2::MotionSwitcherClient msc_;
  unitree::robot::go2::SportClient sc_;

  Obs<float> obs_;
  Gamepad gamepad_;
  std::vector<int> motor_idx_;
  std::vector<float> kps_, kds_, init_pos_;
  std::vector<float> actions_, obs_history_, max_cmd_, cmd_scale_;
  float ang_vel_scale_, dof_pos_scale_, dof_vel_scale_, action_scale_;
  int num_actions_, num_obs_;

  bool move_ = false, ready_ = false, default_ = false;
};

#endif