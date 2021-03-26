- [金川工厂天车视觉定位](#金川工厂天车视觉定位)
- [依赖](#依赖)
- [运行](#运行)
- [开发和测试](#开发和测试)
  - [相机参数标定](#相机参数标定)
  - [功能性测试](#功能性测试)
- [docker](#docker)
  - [image创建](#image创建)
  - [image使用](#image使用)

## 金川工厂天车视觉定位

## 依赖

opencv （在4.1.0上测试通过，3.4.1测试通过）
Eigen >= 3.3 (for g2o)

## 编译

```bash
# 代码下载
git clone --recursive git@gitlab.com:3d-mapping/3d-reconstruction/case-base/crane_localization.git

#编译g2o
cd third_party/g2o
mkdir build
cd build
cmake -DOpenCV_DIR="/usr/local/opencv341/share/OpenCV" ..
make -j
cd ../../..

#编译DBoW2
mkdir build
cd build
F
make -j
cd ../../..

#编译Release
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR="/usr/local/opencv341/share/OpenCV" ..
make -j
cd ..

#编译Debug
mkdir debug
cd debug
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j
cd ..
```

## 开发和测试

### 相机参数标定

> 棋盘格进行相机标定
详见calibration_imu_cam中，[kalibr标定BL-EX346HP-15M内参](https://gitlab.com/3d-mapping/3d-reconstruction/case-base/calibration_imu_cam#kalibr%E6%A0%87%E5%AE%9Abl-ex346hp-15m%E5%86%85%E5%8F%82)
> 畸变矫正有部分效果

```bash
./build/test/test_distortion_fisheye
```

### 功能性测试

注：需修改g2o_path，以及用于测试的mp4视频的路径

```bash
# 定位系统
g2o_path=./third_party/g2o/lib
export LD_LIBRARY_PATH=${g2o_path}

./build/test/test_system <crane_id> <video_file> <config_yaml> <skip_frames>
./build/test/test_system /home/ipsg/dataset_temp/78_cut.mp4 ./conf/pipeline.yaml 0
./build/test/test_system /home/ipsg/dataset_temp/78.mp4 ./conf/pipeline.yaml 4300

./build/test/test_system 1 /home/ipsg/dataset_temp/78.mp4 ./conf/pipeline_online.yaml 4300
./build/test/test_system 1 /home/ipsg/dataset_temp/output.mp4 ./conf/pipeline_online.yaml 0
./build/test/test_system 1 /home/ipsg/dataset_temp/78_cut_3.mp4 ./conf/pipeline_online.yaml 0
./build/test/test_system 1 /home/ipsg/dataset_temp/78_cut_4.mp4 ./conf/pipeline_online.yaml 0

./build/test_websocket
./run.sh

./build/test/test_system /data/DATASETS/ros_bag/BL-EX346HP-15M/crane/78.mp4 ./conf/pipeline.yaml 4300


# DBoW2相似性评估
./build/test/test_DBoW2

# 词典训练
./build/test/test_trainingVoc

```

## 部署上线
### image创建
```bash
# 宿主机中创建容器
docker pull ubuntu:16.04
#docker run --name crane_dev -it ubuntu:16.04 bash
docker run --name crane_dev -v /home/xt/Documents/data/:/data/ -it ubuntu:16.04 bash
```

```bash
# 在容器中安装依赖
apt-get update
apt-get install -y vim wget 
apt-get install -y cmake build-essential gdb

# yaml
apt-get install -y libyaml-cpp-dev

# eigen for g2o
wget https://gitlab.com/libeigen/eigen/-/archive/3.3.4/eigen-3.3.4.tar.gz
tar zxvf eigen-3.3.4.tar.gz && rm eigen-3.3.4.tar.gz
cd eigen-3.3.4 && mkdir build && cd build
cmake .. && make install 
cd ../.. && rm -rf eigen-3.3.4

# ffmpeg
apt-get install -y software-properties-common

add-apt-repository ppa:djcj/hybrid
apt-get update
apt-get -y install ffmpeg 

# opencv 依赖
apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libgtk2.0-dev pkg-config
# opencv 无contrib
wget https://github.com/opencv/opencv/archive/3.4.1.tar.gz
tar zxvf 3.4.1.tar.gz && rm 3.4.1.tar.gz 
cd opencv-3.4.1 && mkdir build && cd build
cmake .. && make -j && make install
cd ../.. && rm -rf opencv-3.4.1

# 测试天车定位程序正常运行（具体过程略）

exit
```
```bash
# 宿主机中保存修改，创建镜像，并将镜像推送至阿里云容器镜像服务中（需使用公司账号密码登录）
docker commit crane_dev crane_localization:v0.1
docker tag crane_localization:v0.1 registry.cn-hangzhou.aliyuncs.com/wattman/crane_localization:v0.1
docker push registry.cn-hangzhou.aliyuncs.com/wattman/crane_localization:v0.1
```

### image使用
```bash
docker pull registry.cn-hangzhou.aliyuncs.com/wattman/crane_localization:v0.1

# 不图形展示
sudo docker run --name crane_loc -v /home/wattman/crane_localization/:/root/crane_localization -it registry.cn-hangzhou.aliyuncs.com/wattman/crane_localization:v0.1 bash

# 支持图形展示
sudo docker run --name crane_loc -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/wattman/crane_localization/:/root/crane_localization -it registry.cn-hangzhou.aliyuncs.com/wattman/crane_localization:v0.1 bash

```

### 运行
```bash
sudo docker exec -it crane_loc bash

cd ~/crane_localization/crane_localization

cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
cd ..

crane_id=1
video_rtsp="rtsp://admin:wattman2020@192.168.1.78:554/Streaming/Channels/101?transportmode=unicast&profile=Profile_1"
config_yaml=./conf/pipeline_online.yaml
skip_frames=0

./build/test/test_system $crane_id $video_rtsp $config_yaml $skip_frames > log/run.log &

shell run
sh run.sh
```

