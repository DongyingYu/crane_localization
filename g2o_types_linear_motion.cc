/**
 * @file g2o_types_linear_motion.cc
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2021-02-27
 *
 * @copyright Copyright (c) 2021
 *
 */
#include "g2o_types_linear_motion.h"

namespace g2o {

// vertix for camera rotation
VertexSO3Expmap::VertexSO3Expmap() {}

bool VertexSO3Expmap::read(std::istream &is) {}

bool VertexSO3Expmap::write(std::ostream &os) const {}

void VertexSO3Expmap::setToOriginImpl() { _estimate = Quaternion::Identity(); }

// oplusImpl() 是顶点更新函数，用于优化过程中增量 Δx的计算；左扰动模型更新
void VertexSO3Expmap::oplusImpl(const number_t *update_) {
  Vector6 update;
  update.setZero();
  for (int i = 0; i < 3; ++i) {
    update[i] = update_[i];
  }
  // 四元数加法
  setEstimate(SE3Quat::exp(update).rotation() * estimate());
}

// vertix for camera translation
VertexLineTranslation::VertexLineTranslation() {}

bool VertexLineTranslation::read(std::istream &is) {}

bool VertexLineTranslation::write(std::ostream &os) const {}

void VertexLineTranslation::setToOriginImpl() { _estimate = 0.0; }
// 直接做加法
void VertexLineTranslation::oplusImpl(const number_t *update_) {
  setEstimate(*update_ + estimate());
}

// edge for pose, do not use
EdgeLinearMotionOnlyPose::EdgeLinearMotionOnlyPose() {}

EdgeLinearMotionOnlyPose::EdgeLinearMotionOnlyPose(const Vector3 &Xw_value,
                                                   const Matrix3 &K)
    : Xw(Xw_value) {
  fx = K(0, 0);
  fy = K(1, 1);
  cx = K(0, 2);
  cy = K(1, 2);
}

bool EdgeLinearMotionOnlyPose::read(std::istream &is) {}
bool EdgeLinearMotionOnlyPose::write(std::ostream &os) const {}

void EdgeLinearMotionOnlyPose::computeError() {
  // 观测值，即在图像上的像素坐标
  Vector2 obs(_measurement);

  // 相机位姿
  const VertexSO3Expmap *v1 = static_cast<VertexSO3Expmap *>(_vertices[0]);
  const VertexLineTranslation *v2 =
      static_cast<VertexLineTranslation *>(_vertices[1]);
  SE3Quat se3quat(v1->estimate(), Vector3(v2->estimate(), 0, 0));
  // std::cout << "[DEBUG]: SE3Quat: " << se3quat << std::endl;
  // 计算像素误差
  Vector3 Xc = se3quat.map(Xw);
  Vector2 proj = cam_project(Xc);
  _error = obs - proj;
  // std::cout << "[DEBUG]: Xw: " << Xw.transpose() << std::endl;
  // std::cout << "[DEBUG]: Xc: " << Xc.transpose() << std::endl;
  // std::cout << "[DEBUG]: obs: " << obs.transpose() << std::endl;
  // std::cout << "[DEBUG]: proj: " << proj.transpose() << std::endl;
  // std::cout << "[DEBUG]: _error: " << _error.transpose() << std::endl;
}

void EdgeLinearMotionOnlyPose::linearizeOplus() {
  // 相机位姿
  const VertexSO3Expmap *v1 = static_cast<VertexSO3Expmap *>(_vertices[0]);
  const VertexLineTranslation *v2 =
      static_cast<VertexLineTranslation *>(_vertices[1]);
  SE3Quat se3quat(v1->estimate(), Vector3(v2->estimate(), 0, 0));

  // 位姿变换后的三维点
  Vector3 xyz_trans = se3quat.map(Xw);

  number_t x = xyz_trans[0];
  number_t y = xyz_trans[1];
  number_t invz = 1.0 / xyz_trans[2];
  number_t invz_2 = invz * invz;

  // 顶点1的雅克比 VertexSO3Expmap
  _jacobianOplusXi(0, 0) = x * y * invz_2 * fx;
  _jacobianOplusXi(0, 1) = -(1 + (x * x * invz_2)) * fx;
  _jacobianOplusXi(0, 2) = y * invz * fx;
  _jacobianOplusXi(1, 0) = (1 + y * y * invz_2) * fy;
  _jacobianOplusXi(1, 1) = -x * y * invz_2 * fy;
  _jacobianOplusXi(1, 2) = -x * invz * fy;

  // 顶点2的雅克比 VertexLineTranslation
  _jacobianOplusXj(0, 0) = -invz * fx;
  _jacobianOplusXj(1, 0) = 0;
}

Vector2 EdgeLinearMotionOnlyPose::cam_project(const Vector3 &trans_xyz) const {
  Vector2 res;
  res[0] = fx * trans_xyz[0] / trans_xyz[2] + cx;
  res[1] = fy * trans_xyz[1] / trans_xyz[2] + cy;
  return res;
}

// 
EdgeLinearMotion::EdgeLinearMotion(const Matrix3 &K) {
  resize(3);
  fx = K(0, 0);
  fy = K(1, 1);
  cx = K(0, 2);
  cy = K(1, 2);
}

bool EdgeLinearMotion::read(std::istream &is) {}
bool EdgeLinearMotion::write(std::ostream &os) const {}

void EdgeLinearMotion::computeError() {
  // 观测值，即在图像上的像素坐标
  Vector2 obs(_measurement);

  // 相机位姿
  const VertexSO3Expmap *v1 = static_cast<VertexSO3Expmap *>(_vertices[0]);
  const VertexLineTranslation *v2 =
      static_cast<VertexLineTranslation *>(_vertices[1]);
  const VertexSBAPointXYZ *v3 = static_cast<VertexSBAPointXYZ *>(_vertices[2]);

  SE3Quat se3quat(v1->estimate(), Vector3(v2->estimate(), 0, 0));
  // std::cout << "[DEBUG]: SE3Quat: " << se3quat << std::endl;
  Vector3 Xw = v3->estimate();
  Vector3 Xc = se3quat.inverse().map(Xw);
  Vector2 proj = cam_project(Xc);
  // 计算像素误差
  _error = obs - proj;
  // std::cout << "[DEBUG]: Xw: " << Xw.transpose() << std::endl;
  // std::cout << "[DEBUG]: Xc: " << Xc.transpose() << std::endl;
  // std::cout << "[DEBUG]: obs: " << obs.transpose() << std::endl;
  // std::cout << "[DEBUG]: proj: " << proj.transpose() << std::endl;
  // std::cout << "[DEBUG]: _error: " << _error.transpose() << std::endl << std::endl;
}
// 在当前顶点的值下，该误差对优化变量的偏导数，Jacobian。
void EdgeLinearMotion::linearizeOplus() {
  // 相机位姿
  const VertexSO3Expmap *v1 = static_cast<VertexSO3Expmap *>(_vertices[0]);
  const VertexLineTranslation *v2 =
      static_cast<VertexLineTranslation *>(_vertices[1]);
  const VertexSBAPointXYZ *v3 = static_cast<VertexSBAPointXYZ *>(_vertices[2]);

  SE3Quat se3quat(v1->estimate(), Vector3(v2->estimate(), 0, 0));
  se3quat = se3quat.inverse();

  Vector3 Xw = v3->estimate();

  // 位姿变换后的三维点
  Vector3 xyz_trans = se3quat.map(Xw);

  number_t x = xyz_trans[0];
  number_t y = xyz_trans[1];
  number_t invz = 1.0 / xyz_trans[2];
  number_t invz_2 = invz * invz;

  // 按照推导，这个符号应该是负
  double sign = -1;

  // 顶点1的雅克比 VertexSO3Expmap
  _jacobianOplus[0](0, 0) = sign * (x * y * invz_2 * fx);
  _jacobianOplus[0](0, 1) = sign * (-(1 + (x * x * invz_2)) * fx);
  _jacobianOplus[0](0, 2) = sign * (y * invz * fx);
  _jacobianOplus[0](1, 0) = sign * ((1 + y * y * invz_2) * fy);
  _jacobianOplus[0](1, 1) = sign * (-x * y * invz_2 * fy);
  _jacobianOplus[0](1, 2) = sign * (-x * invz * fy);

  // 顶点2的雅克比 VertexLineTranslation
  _jacobianOplus[1](0, 0) = sign * (-invz * fx);
  _jacobianOplus[1](1, 0) = sign * (0);

  // 顶点3的雅克比 VertexSBAPointXYZ
  Eigen::Matrix<number_t, 2, 3> tmp;
  tmp(0, 0) = fx;
  tmp(0, 1) = 0;
  tmp(0, 2) = -x * invz * fx;
  tmp(1, 0) = 0;
  tmp(1, 1) = fy;
  tmp(1, 2) = -y * invz * fy;
  _jacobianOplus[2] = -invz * tmp * se3quat.rotation().toRotationMatrix();
}

Vector2 EdgeLinearMotion::cam_project(const Vector3 &trans_xyz) const {
  Vector2 res;
  res[0] = fx * trans_xyz[0] / trans_xyz[2] + cx;
  res[1] = fy * trans_xyz[1] / trans_xyz[2] + cy;
  return res;
}

} // namespace g2o