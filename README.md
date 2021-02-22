- [金川工厂天车视觉定位](#金川工厂天车视觉定位)
- [依赖](#依赖)
- [运行](#运行)
- [开发和测试](#开发和测试)
  - [相机参数标定](#相机参数标定)
  - [功能性测试](#功能性测试)

## 金川工厂天车视觉定位

## 依赖

opencv （在4.1.0上测试通过，3.2.0应该也可以，暂未测试）

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


#编译
mkdir build
cd build
cmake ..
make -j

#运行(todo)

```

## 开发和测试

### 相机参数标定

> 棋盘格进行相机标定
详见calibration_imu_cam中，[kalibr标定BL-EX346HP-15M内参](https://gitlab.com/3d-mapping/3d-reconstruction/case-base/calibration_imu_cam#kalibr%E6%A0%87%E5%AE%9Abl-ex346hp-15m%E5%86%85%E5%8F%82)
> 畸变矫正以及检验，发现畸变系数有误，暂时不进行去畸变

```bash
./test/test_distortion
```

### 功能性测试

注：需修改g2o_path，以及用于测试的mp4视频的路径

```bash
# 单目初始化
./test/test_initializer

# 单目初始化+初始化地图ba
g2o_path=/home/xt/Documents/data/3D-Mapping/3D-Reconstruction/case-base/crane_localization/third_party/g2o/lib
export LD_LIBRARY_PATH=${g2o_path}

./test/test_initialize_map

# 定位系统（未完成）
g2o_path=/home/xt/Documents/data/3D-Mapping/3D-Reconstruction/case-base/crane_localization/third_party/g2o/lib
export LD_LIBRARY_PATH=${g2o_path}

./test/test_system


```
