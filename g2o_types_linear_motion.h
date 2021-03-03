/**
 * @file g2o_types_linear_motion.h
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2021-02-27
 *
 * @copyright Copyright (c) 2021
 *
 */
#include "third_party/g2o/g2o/core/base_binary_edge.h"
#include "third_party/g2o/g2o/core/base_multi_edge.h"
#include "third_party/g2o/g2o/core/base_vertex.h"
#include "third_party/g2o/g2o/types/sba/types_sba.h"
#include "third_party/g2o/g2o/types/slam3d/se3quat.h"

namespace g2o {

/**
 * @brief 相机位姿旋转量
 * @note 使用SO3，部分运算借用SE3Quat
 */
class VertexSO3Expmap : public BaseVertex<3, Quaternion> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  VertexSO3Expmap();

  bool read(std::istream &is) override;

  bool write(std::ostream &os) const override;

  void setToOriginImpl() override;

  void oplusImpl(const number_t *update_) override;
};

/**
 * @brief 沿着X轴直线的运动的位移量
 * @note 注：世界坐标系中，相机运动为沿X方向运动的直线
 */
class VertexLineTranslation : public BaseVertex<1, number_t> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  VertexLineTranslation();

  bool read(std::istream &is) override;

  bool write(std::ostream &os) const override;

  void setToOriginImpl() override;

  void oplusImpl(const number_t *update_);
};

class EdgeLinearMotionOnlyPose
    : public BaseBinaryEdge<2, Vector2, VertexSO3Expmap,
                            VertexLineTranslation> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EdgeLinearMotionOnlyPose();
  EdgeLinearMotionOnlyPose(const Vector3 &Xw_value, const Matrix3 &K);

  bool read(std::istream &is) override;
  bool write(std::ostream &os) const override;

  void computeError() override;

  void linearizeOplus() override;

  Vector2 cam_project(const Vector3 &trans_xyz) const;

  Vector3 Xw;
  number_t fx, fy, cx, cy;
};

class EdgeLinearMotion : public BaseMultiEdge<2, Vector2> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EdgeLinearMotion(const Matrix3 &K);

  bool read(std::istream &is) override;
  bool write(std::ostream &os) const override;

  void computeError() override;

  void linearizeOplus() override;

  Vector2 cam_project(const Vector3 &trans_xyz) const;

  number_t fx, fy, cx, cy;
};

} // namespace g2o