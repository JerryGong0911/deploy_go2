// Standalone depth publisher: RealSense depth -> DDS (HeightMap_).
//
// Runs as its own process so camera I/O (USB stalls, driver hiccups, camera
// restarts) never blocks the controller's 50 Hz loop. Publishes the full
// depth image in meters; all training-specific preprocessing (crop, clip,
// resize, normalize) stays in go2_deploy_cpp so it is configured in one place.
//
// Reads ../config/go2.yaml (run from build/, like go2_deploy_cpp) for the DDS
// domain / interface and depth.topic, and depth.camera for the stream.

#include <unistd.h>
#include <yaml-cpp/yaml.h>
#include <chrono>
#include <csignal>
#include <exception>
#include <iostream>
#include <librealsense2/rs.hpp>
#include <string>
#include <unitree/idl/go2/HeightMap_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>

namespace {

constexpr const char* kConfigPath = "../config/go2.yaml";

volatile sig_atomic_t stop_signal = 0;
void SignalHandler(int) {
  stop_signal = 1;
}

double WallTimeSec() {
  return std::chrono::duration<double>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

// Non-blocking presence check. pipeline::start() blocks inside librealsense
// while no device is connected and cannot be interrupted by Ctrl+C.
bool CameraPresent(const std::string& serial) {
  rs2::context ctx;
  for (auto&& dev : ctx.query_devices()) {
    if (serial.empty() || serial == dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER))
      return true;
  }
  return false;
}

// Opens the camera; returns the depth scale (raw Z16 unit -> meters).
float StartCamera(rs2::pipeline* pipe, const std::string& serial, int width,
                  int height, int fps) {
  rs2::config cfg;
  if (!serial.empty())
    cfg.enable_device(serial);
  cfg.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16, fps);
  rs2::pipeline_profile profile = pipe->start(cfg);
  auto dev = profile.get_device();
  std::cout << "Camera: " << dev.get_info(RS2_CAMERA_INFO_NAME) << " (S/N "
            << dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) << "), depth "
            << width << "x" << height << " @ " << fps << " Hz" << std::endl;
  return dev.first<rs2::depth_sensor>().get_depth_scale();
}

}  // namespace

int main() {
  try {
    YAML::Node config = YAML::LoadFile(kConfigPath);
    YAML::Node depth = config["depth"];
    if (!depth)
      throw std::runtime_error("no 'depth' block in " +
                               std::string(kConfigPath));
    const std::string topic = depth["topic"].as<std::string>();
    YAML::Node cam = depth["camera"];
    const std::string serial = cam["serial"].as<std::string>("");
    const int width = cam["width"].as<int>(424);
    const int height = cam["height"].as<int>(240);
    const int fps = cam["fps"].as<int>(30);

    // Same DDS domain / interface selection as go2_deploy_cpp.
    if (config["real_robot"].as<bool>()) {
      const auto nic = config["network_interface"].as<std::string>();
      std::cout << "Mode: REAL ROBOT (DDS domain 0, interface " << nic << ")"
                << std::endl;
      unitree::robot::ChannelFactory::Instance()->Init(0, nic);
    } else {
      const int domain = config["sim_domain_id"].as<int>();
      std::cout << "Mode: SIMULATION (DDS domain " << domain
                << ", interface lo)" << std::endl;
      unitree::robot::ChannelFactory::Instance()->Init(domain, "lo");
    }

    unitree::robot::ChannelPublisher<unitree_go::msg::dds_::HeightMap_>
        publisher(topic);
    publisher.InitChannel();

    std::signal(SIGINT, SignalHandler);
    std::signal(SIGTERM, SignalHandler);

    unitree_go::msg::dds_::HeightMap_ msg;
    msg.frame_id("depth");
    msg.resolution(0.0f);  // unused: data is a camera image, not a map
    msg.origin({0.0f, 0.0f});

    // Outer loop: (re)open the camera, so unplug / replug recovers.
    while (!stop_signal) {
      try {
        if (!CameraPresent(serial)) {
          std::cerr << "Waiting for RealSense camera"
                    << (serial.empty() ? "" : " S/N " + serial) << "..."
                    << std::endl;
          sleep(1);
          continue;
        }
        rs2::pipeline pipe;
        const float scale = StartCamera(&pipe, serial, width, height, fps);
        std::cout << "Publishing depth (m) on " << topic << std::endl;

        size_t frames = 0;
        auto report = std::chrono::steady_clock::now();
        while (!stop_signal) {
          rs2::frameset fs;
          if (!pipe.try_wait_for_frames(&fs, 1000)) {
            std::cerr << "No depth frame for 1 s" << std::endl;
            continue;
          }
          rs2::depth_frame f = fs.get_depth_frame();
          const int w = f.get_width(), h = f.get_height();
          const int stride = f.get_stride_in_bytes() / 2;
          const auto* z = static_cast<const uint16_t*>(f.get_data());

          // 0 = no return; it stays 0 and the controller maps it to far_clip.
          auto& data = msg.data();
          data.resize((size_t)w * h);
          for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
              data[(size_t)y * w + x] = z[(size_t)y * stride + x] * scale;
          msg.width(w);
          msg.height(h);
          msg.stamp(WallTimeSec());
          publisher.Write(msg);

          ++frames;
          auto now = std::chrono::steady_clock::now();
          if (now - report >= std::chrono::seconds(5)) {
            float dt = std::chrono::duration<float>(now - report).count();
            std::cout << "depth: " << frames / dt << " Hz" << std::endl;
            frames = 0;
            report = now;
          }
        }
        pipe.stop();
      } catch (const rs2::error& e) {
        std::cerr << "RealSense error: " << e.what() << " -- retrying in 1 s"
                  << std::endl;
        sleep(1);
      }
    }
    std::cout << "Depth publisher stopped." << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Fatal: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
