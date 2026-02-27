import torch
import time
import sys
import numpy as np

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient

from common.remote_controller import KeyMap, RemoteController
from common.rotation_helper import get_gravity_orientation
from config import Config


class RLDeploy:

    def __init__(self, config: Config) -> None:

        self.config = config
        self.num_actions = self.config.num_actions

        self.remote_controller = RemoteController()

        # Initialize the policy network
        self.policy = torch.jit.load(config.policy_path)
        self.encoder = torch.jit.load(config.encoder_path)
        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.init_dof_pos = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.obs_history = np.zeros(config.num_obs * 5, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.zeros_like(config.max_cmd, dtype=np.float32)
        self.cmd_max = np.zeros_like(config.max_cmd, dtype=np.float32)

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = unitree_go_msg_dds__LowState_()
        self.InitLowCmd()

        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

        self.move = 0
        self.stop = 1
        self.ready = 0
        self.task_current_step = 0
        self.climb_mode = 0

        if len(sys.argv) >= 2:
            self.sc = SportClient()  
            self.sc.SetTimeout(5.0)
            self.sc.Init()

            self.msc = MotionSwitcherClient()
            self.msc.SetTimeout(5.0)
            self.msc.Init()

            status, result = self.msc.CheckMode()
            while result['name']:
                self.sc.StandDown()
                self.msc.ReleaseMode()
                status, result = self.msc.CheckMode()
                time.sleep(1)
        else:
            pass

        self.InitRecurrentThread()

    # Handler for low state messages
    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    # Initialize recurrent threads
    def InitRecurrentThread(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(interval=0.001, target=self.SendCmd, name="writebasiccmd")
        self.lowCtrlThreadPtr = RecurrentThread(interval=0.02, target=self.Ctrl, name="lowCtrlThread")
        self.FSMThreadPtr = RecurrentThread(interval=0.01, target=self.FSM, name="FSMThread")
        self.lowCmdWriteThreadPtr.Start()
        self.lowCtrlThreadPtr.Start()
        self.FSMThreadPtr.Start()

    # Initialize low command message
    def InitLowCmd(self):
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        self.PosStopF = 2.146e9
        self.VelStopF = 16000.0

        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.low_cmd.motor_cmd[i].q = self.PosStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = self.VelStopF
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def CreateDampingCmd(self):
        size = len(self.low_cmd.motor_cmd)
        for i in range(size):
            self.low_cmd.motor_cmd[i].q = 0
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = 0
            self.low_cmd.motor_cmd[i].kd = 3.
            self.low_cmd.motor_cmd[i].tau = 0

    def CreateZeroCmd(self):
        size = len(self.low_cmd.motor_cmd)
        for i in range(size):
            self.low_cmd.motor_cmd[i].q = 0
            self.low_cmd.motor_cmd[i].dq = 0
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def MoveToDefault(self):
        if self.task_current_step == 0:
            total_time = 2.0
            self.task_total_step = int(total_time / 0.02)

        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.init_dof_pos[i] = self.low_state.motor_state[motor_idx].q

        if self.task_current_step < self.task_total_step:
            alpha = self.task_current_step / self.task_total_step
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = (1 - alpha) * self.init_dof_pos[i] + alpha * self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].kp = 40
                self.low_cmd.motor_cmd[motor_idx].dq = 0.0
                self.low_cmd.motor_cmd[motor_idx].kd = 1
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0
            self.task_current_step += 1
        else:
            self.ready = 1
            self.task_current_step = 0

    def DefaultPosState(self):
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
            self.low_cmd.motor_cmd[motor_idx].kp = 40
            self.low_cmd.motor_cmd[motor_idx].dq = 0.0
            self.low_cmd.motor_cmd[motor_idx].kd = 0.6
            self.low_cmd.motor_cmd[motor_idx].tau = 0.0

    def Forward(self):
        # Get the current joint position and velocity
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.qj[i] = self.low_state.motor_state[motor_idx].q
            self.dqj[i] = self.low_state.motor_state[motor_idx].dq

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        gravity_orientation = get_gravity_orientation(quat)
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        qj_obs = (qj_obs - self.config.default_angles) * self.config.dof_pos_scale
        dqj_obs = dqj_obs * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale

        self.cmd[0] = self.remote_controller.ly
        self.cmd[1] = self.remote_controller.lx * -1
        self.cmd[2] = self.remote_controller.rx * -1

        self.obs[:3] = ang_vel
        self.obs[3:6] = gravity_orientation
        self.obs[6:6 + self.num_actions] = qj_obs
        self.obs[6 + self.num_actions:6 + 2 * self.num_actions] = dqj_obs
        self.obs[6 + 2 * self.num_actions:6 + 3 * self.num_actions] = self.action
        self.obs_history = np.roll(self.obs_history, -self.config.num_obs)
        self.obs_history[-self.config.num_obs:] = self.obs

        # Convert to PyTorch tensors
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        obs_history_tensor = torch.from_numpy(self.obs_history).unsqueeze(0)
        cmd_tensor = torch.from_numpy(self.cmd * self.config.cmd_scale * self.cmd_max).unsqueeze(0)

        self.latents = self.encoder(obs_history_tensor).detach()
        self.action = self.policy(torch.cat([obs_tensor, cmd_tensor, self.latents], dim=1)).detach().numpy().squeeze()
        self.target_dof_pos = self.action * self.config.action_scale + self.config.default_angles

        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = self.target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].dq = 0.0
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0.0

    def Ctrl(self):
        self.CMDSwitch()
        self.MotionSwitch()

    def CMDSwitch(self):
        if self.climb_mode == 1:
            self.cmd_max = self.config.max_cmd_slope
        else:
            self.cmd_max = self.config.max_cmd

    def MotionSwitch(self):
        if (self.stop == 1):
            self.CreateDampingCmd()
            print("Stop")
        elif (self.move == 1 and self.ready == 1):
            self.Forward()
            print("Forward")
        elif (self.default == 1):
            if (self.ready == 1):
                self.DefaultPosState()
                print("Ready")
            else:
                self.MoveToDefault()
                print("Move to Default")

    def FSM(self):
        self.last_stop = self.stop
        if (self.remote_controller.button[KeyMap.R1] == 1):
            self.stop = 1
            self.move = 0
            self.default = 0
            self.ready = 0
        elif (self.remote_controller.button[KeyMap.L1] == 1 and self.remote_controller.button[KeyMap.B] == 1):
            self.stop = 0
            self.move = 1
            self.default = 0
        elif (self.remote_controller.button[KeyMap.L1] == 1 and self.remote_controller.button[KeyMap.A] == 1):
            self.stop = 0
            self.move = 0
            self.default = 1

        if (self.remote_controller.button[KeyMap.X] == 1):
            self.climb_mode = 1
        elif (self.remote_controller.button[KeyMap.Y] == 1):
            self.climb_mode = 0

    # Send command
    def SendCmd(self):
        self.low_cmd.crc = CRC.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        ChannelFactoryInitialize(1, "lo")
    else:
        ChannelFactoryInitialize(0, sys.argv[1])

    config_path = "config/go2.yaml"
    config = Config(config_path)
    rl_deploy = RLDeploy(config)

    while True:
        time.sleep(1)
