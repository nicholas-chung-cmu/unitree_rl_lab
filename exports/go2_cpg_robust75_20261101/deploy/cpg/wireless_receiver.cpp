// Receive joystick input ONLY. No LowCmd, SportClient, or DDS publisher exists.
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/go2/WirelessController_.hpp>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thread>
#include <unistd.h>

static volatile std::sig_atomic_t running = 1;
static void stop(int) { running = 0; }

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: wireless_receiver INTERFACE DOMAIN OUTPUT.json\n"
                     "Read-only rt/wireless_controller subscription; no robot commands.\n";
        return 2;
    }
    const std::string output = argv[3];
    const std::string temporary = output + "." + std::to_string(getpid()) + ".tmp";
    unitree::robot::ChannelFactory::Instance()->Init(std::stoi(argv[2]), argv[1]);
    unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::WirelessController_> subscriber("rt/wireless_controller");
    subscriber.InitChannel([&](const void* message) {
        const auto& input = *static_cast<const unitree_go::msg::dds_::WirelessController_*>(message);
        const double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        std::ofstream file(temporary, std::ios::trunc);
        file << std::setprecision(17) << "{\"monotonic_seconds\":" << now
             << ",\"lx\":" << input.lx() << ",\"ly\":" << input.ly()
             << ",\"rx\":" << input.rx() << ",\"ry\":" << input.ry()
             << ",\"keys\":" << input.keys() << "}\n";
        file.close();
        if (!file || std::rename(temporary.c_str(), output.c_str()) != 0)
            std::cerr << "Cannot write wireless snapshot: " << output << '\n';
    }, 1);
    std::signal(SIGINT, stop);
    std::signal(SIGTERM, stop);
    std::cout << "Receiving rt/wireless_controller on " << argv[1] << ", domain " << argv[2] << '\n';
    while (running) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    subscriber.CloseChannel();
    std::remove(temporary.c_str());
}
