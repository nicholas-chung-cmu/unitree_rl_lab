// Read-only raw sensing. No publisher, MotionSwitcher, SportClient or motor output.
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <chrono>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thread>
#include <filesystem>
static volatile std::sig_atomic_t running=1;
static void stop(int){running=0;}
int main(int argc,char** argv){
 if(argc!=5){std::cerr<<"Usage: lowstate_receiver INTERFACE DOMAIN OUTPUT.jsonl SECONDS\nRead-only LowState capture; existing controller is untouched.\n";return 2;}
 if(std::filesystem::exists(argv[3])) {std::cerr<<"Output exists; choose a new recording path\n";return 2;}
 double seconds=std::stod(argv[4]);
 if(!(seconds>0&&seconds<=600))return 2;
 std::ofstream file(argv[3]);
 if(!file){std::cerr<<"Cannot open output\n";return 2;}
 unitree::robot::ChannelFactory::Instance()->Init(std::stoi(argv[2]),argv[1]);
 unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::LowState_> sub("rt/lowstate");
 sub.InitChannel([&](const void* msg){
  const auto& s=*static_cast<const unitree_go::msg::dds_::LowState_*>(msg);
  double t=std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  file<<std::setprecision(17)<<"{\"monotonic_seconds\":"<<t<<",\"tick\":"<<s.tick()<<",\"q\":[";
  for(int i=0;i<12;++i)file<<(i?",":"")<<s.motor_state()[i].q();
  file<<"],\"qd\":[";for(int i=0;i<12;++i)file<<(i?",":"")<<s.motor_state()[i].dq();
  file<<"],\"tau_est\":[";for(int i=0;i<12;++i)file<<(i?",":"")<<s.motor_state()[i].tau_est();
  file<<"],\"quaternion\":[";for(int i=0;i<4;++i)file<<(i?",":"")<<s.imu_state().quaternion()[i];
  file<<"],\"gyro\":[";for(int i=0;i<3;++i)file<<(i?",":"")<<s.imu_state().gyroscope()[i];
  file<<"],\"foot_force_sdk\":[";for(int i=0;i<4;++i)file<<(i?",":"")<<s.foot_force()[i];
  file<<"]}\n";
 },1);
 std::signal(SIGINT,stop);std::signal(SIGTERM,stop);
 auto start=std::chrono::steady_clock::now();
 while(running && std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<seconds)
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
 sub.CloseChannel();file.close();
}
