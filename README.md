- [金川工厂天车视觉定位](#金川工厂天车视觉定位)
- [依赖](#依赖)
- [运行](#运行)
- [开发和测试](#开发和测试)
  - [相机参数标定](#相机参数标定)
  - [功能性测试](#功能性测试)

## 金川工厂天车视觉定位

## 依赖

opencv （在4.1.0上测试通过，3.4.1测试通过）
Eigen >= 3.3 (for g2o)

## 运行

```bash
# 代码下载
git clone --recursive git@gitlab.com:3d-mapping/3d-reconstruction/case-base/crane_localization.git

#编译g2o
cd third_party/g2o
mkdir build
cd build
cmake ..
make -j
cd ../../..

#编译DBoW2(todo)
mkdir build
cd build
cmake -D OpenCV_DIR="/usr/local/opencv341/share/OpenCV" ..
make -j
cd ../../..

#编译
mkdir build
cd build
cmake -DOpenCV_DIR="/usr/local/opencv341/share/OpenCV" ..
make -j
cd ..

#运行(todo)

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

./build/test/test_system <video_file> <config_yaml> <skip_frames>
./build/test/test_system /home/ipsg/dataset_temp/78_cut.mp4 ./conf/pipeline.yaml 0
./build/test/test_system /home/ipsg/dataset_temp/78.mp4 ./conf/pipeline.yaml 4300

./build/test/test_system /data/DATASETS/ros_bag/BL-EX346HP-15M/crane/78.mp4 ./conf/pipeline.yaml 4300


# DBoW2相似性评估
./build/test/test_DBoW2

# 词典训练
./build/test/test_trainingVoc

```
