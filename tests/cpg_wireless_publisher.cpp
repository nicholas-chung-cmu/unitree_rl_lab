// Local test fixture only: fixed loopback interface/domain, wireless input only.
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/idl/go2/WirelessController_.hpp>
#include <chrono>
#include <thread>
int main() {
    unitree::robot::ChannelFactory::Instance()->Init(77, "lo");
    unitree::robot::ChannelPublisher<unitree_go::msg::dds_::WirelessController_> pub("rt/wireless_controller");
    pub.InitChannel();
    unitree_go::msg::dds_::WirelessController_ message;
    message.lx(-.5f);message.ly(.75f);message.rx(.25f);message.ry(0);message.keys(0);
    for (int i=0;i<150;++i) {
        pub.Write(message);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}
