/**
 * @file optimizer.h
 * @author xiaotaw (you@domain.com)
 * @brief g2o optimizer
 * @version 0.1
 * @date 2021-02-19
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once
#include "frame.h"
#include "map.h"
#include "third_party/g2o/g2o/core/block_solver.h"
#include "third_party/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "third_party/g2o/g2o/core/robust_kernel_impl.h"
#include "third_party/g2o/g2o/core/solver.h"
#include "third_party/g2o/g2o/core/sparse_optimizer.h"
#include "third_party/g2o/g2o/solvers/dense/linear_solver_dense.h"
#include "third_party/g2o/g2o/solvers/eigen/linear_solver_eigen.h"
#include "third_party/g2o/g2o/solvers/structure_only/structure_only_solver.h"
#include "third_party/g2o/g2o/stuff/sampler.h"
#include "third_party/g2o/g2o/types/sba/types_six_dof_expmap.h"

class G2oOptimizer {
public:
  /**
   * @brief 优化frame的位姿
   *
   * @param[in out] frame 待优化的帧
   * @param[in] map 当前地图
   * @param[in] n_iteration 迭代次数
   *
   * @todo 重新设计，避免使用裸指针Map*
   */
  static void optimizeFramePose(Frame::Ptr frame, Map *map,
                                const int &n_iteration = 10);

  /**
   * @brief 优化map中的mappoints和frame pose
   *
   * @param[in out] map 待优化的地图
   * @param[in] n_iteration 迭代次数
   */
  static void mapBundleAdjustment(Map::Ptr map, const int &n_iteration = 25);
};

namespace g2o {

/**
 * @brief SO3，部分运算借用SE3Quat
 */
class VertexSO3Expmap : public BaseVertex<3, Quaternion> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  VertexSO3Expmap() {}

  bool read(std::istream &is) override {}

  bool write(std::ostream &os) const override {}

  void setToOriginImpl() override { _estimate = Quaternion::Identity(); }

  void oplusImpl(const number_t *update_) override {
    Vector6 update;
    update.setZero();
    for (int i = 0; i < 3; ++i) {
      update[i] = update_[i];
    }
    setEstimate(SE3Quat::exp(update).rotation() * estimate());
  }
};

/**
 * @brief 沿着直线的运动的位移量
 */
class VertexLineTranslation : public BaseVertex<1, number_t> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  VertexLineTranslation() {}

  bool read(std::istream &is) override {}

  bool write(std::ostream &os) const override {}

  void setToOriginImpl() override { _estimate = 0.0; }

  void oplusImpl(const number_t *update_) {
    setEstimate(*update_ + estimate());
  }
};

class EdgeSO3LinearMotionProjectXYZOnlyPose
    : public BaseBinaryEdge<2, Vector2, VertexSO3Expmap,
                            VertexLineTranslation> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EdgeSO3LinearMotionProjectXYZOnlyPose() {}
  EdgeSO3LinearMotionProjectXYZOnlyPose(const Vector3 &Xw_value,
                                        const Matrix3 &K)
      : Xw(Xw_value) {
    fx = K(0, 0);
    fy = K(1, 1);
    cx = K(0, 2);
    cy = K(1, 2);
  }

  bool read(std::istream &is) override {}
  bool write(std::ostream &os) const override {}

  void computeError() override {
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

  void linearizeOplus() {
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

  Vector2 cam_project(const Vector3 &trans_xyz) const {
    Vector2 res;
    res[0] = fx * trans_xyz[0] / trans_xyz[2] + cx;
    res[1] = fx * trans_xyz[1] / trans_xyz[2] + cy;
    return res;
  }

  Vector3 Xw;
  number_t fx, fy, cx, cy;
};

} // namespace g2o

/**
 * @brief 限定直线运动的图优化
 *
 */
class G2oOptimizerForLinearMotion {
public:
  /**
   * @brief calculate the minimal rotation, which can convert src to dst.
   * @return Eigen::Quaterniond
   */
  static Eigen::Quaterniond calMinRotation(const Eigen::Vector3d &src,
                                           const Eigen::Vector3d &dst);

  static void mapBundleAdjustment(Map::Ptr map, const int &n_iteration = 10);
};