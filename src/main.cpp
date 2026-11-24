#include <unistd.h>
#include <yaml-cpp/yaml.h>
#include <csignal>
#include <exception>
#include <iostream>
#include <string>
#include "controller.hpp"

namespace {

// Relative to build/, where the binary is run from.
constexpr const char* kConfigPath = "../config/go2.yaml";

volatile sig_atomic_t stop_signal = 0;
void SignalHandler(int) {
  stop_signal = 1;  // only async-signal-safe work here
}

}  // namespace

int main() {
  try {
    YAML::Node config = YAML::LoadFile(kConfigPath);
    const bool real_robot = config["real_robot"].as<bool>();

    if (real_robot) {
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

    // Ctrl+C during construction (waiting for LowState / releasing services)
    // keeps the default action and exits immediately: no command is sent yet.
    Controller controller(kConfigPath);
    signal(SIGINT, SignalHandler);
    while (!stop_signal) {
      sleep(1);
    }
    std::cout << "Shutting down..." << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Fatal: " << e.what() << std::endl;
    return 1;
  }

  std::cout << "Done" << std::endl;
  return 0;
}
