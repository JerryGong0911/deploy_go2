#include "controller.hpp"

volatile sig_atomic_t stop_signal = 0;
void SignalHandler(int signum) {
  std::cout << "Shutting down..." <<std::endl;
  stop_signal = 1;
}

int main(int argc, char **argv) {

  signal(SIGINT, SignalHandler);
  unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);
  Controller controller;
  while (!stop_signal) {
    sleep(1);
  }
  std::cout << "Done" << std::endl;
  return 0;
}