// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

// Modification Note: 
// This file may have been modified by the authors of SchurVINS.
// (All authors of SchurVINS are with PICO department of ByteDance Corporation)

#include <svo/common/point.h>

#include <vikit/math_utils.h>
#include <svo/common/logging.h>
#include <svo/common/frame.h>
#include <svo/common/local_feature.h>

namespace svo {

std::atomic<int> PointIdProvider::last_id_ { 0 };

KeypointIdentifier::KeypointIdentifier(const FramePtr& _frame, const size_t _feature_index)
  : frame(_frame)
  , frame_id(_frame->id_)
  , keypoint_index_(_feature_index)
{ ; }

Point::Point(const Eigen::Vector3d& pos)
  : Point(PointIdProvider::getNewPointId(), pos)
{ ; }

Point::Point(const int id, const Eigen::Vector3d& pos)
  : id_(id)
  , pos_(pos)
{
  last_projected_kf_id_.fill(-1);
}

Point::~Point()
{}

std::atomic_uint64_t Point::global_map_value_version_ {0u};

bool Point::CheckStatus() {
    constexpr double MAX_DIST = 1e3;
    constexpr int MIN_OBS = 3;
    if (pos().norm() > MAX_DIST || local_obs_.size() < MIN_OBS)
        return false;

    return true;
}

bool Point::CheckLocalStatus() {
    if (local_status_ == false)
        return false;

    return true;
}

bool Point::CheckLocalStatus(const int prev_frame_id0, const int prev_frame_id1, const int curr_framebundle_id) {
    auto iter = local_obs_.find(curr_framebundle_id);
    bool has_opt_prev = ((last_structure_optim_ == prev_frame_id0) || (last_structure_optim_ == prev_frame_id1));
    bool has_obs_curr = (iter != local_obs_.end());

    if (has_opt_prev || has_obs_curr)
        return true;

    return false;
}

void Point::RemoveLocalObs(const int state_id, const int camera_id) {
    svo::LocalFeatureMap::iterator iter = local_obs_.find(state_id);
    if (iter == local_obs_.end())
        return;
    for (; iter != local_obs_.end(); ++iter) {
        if (iter->first != state_id)
            break;
        if (iter->first == state_id && iter->second->camera_id == camera_id) {
            // LOG(ERROR) << "find local obs erase it";
            local_obs_.erase(iter);
            break;
        }
    }
}

void Point::addObservation(const FramePtr& frame, const size_t feature_index)
{
  CHECK_NOTNULL(frame.get());

  // check that we don't have yet a reference to this frame
  // TODO(cfo): maybe we should use a std::unordered_map to store the observations.
  const auto id = frame->id();
  auto it = std::find_if(obs_.begin(), obs_.end(),
                         [&](const KeypointIdentifier& i){ return i.frame_id == id; });
  if(it == obs_.end())
  {
    obs_.emplace_back(KeypointIdentifier(frame, feature_index));
  }
  else
  {
    CHECK_EQ(it->keypoint_index_, feature_index);
  }
}

void Point::removeObservation(int frame_id)
{
  obs_.erase(
        std::remove_if(obs_.begin(), obs_.end(),
                       [&](const KeypointIdentifier& o){ return o.frame_id == frame_id; }),
        obs_.end());
}

void Point::initNormal()
{
  CHECK(!obs_.empty()) << "initializing normal without any observation";

  if(const FramePtr& frame = obs_.front().frame.lock())
  {
    BearingVector f = frame->f_vec_.col(obs_.front().keypoint_index_);
    normal_ = frame->T_f_w_.getRotation().inverseRotate(-f);
    normal_information_ = Eigen::Matrix2d::Identity(); //DiagonalMatrix<double,3,3>(pow(20/(pos_-ftr->frame->pos()).norm(),2), 1.0, 1.0);
    normal_set_ = true;
  }
  else
    SVO_ERROR_STREAM("could not unlock weak_ptr<frame> in normal initialization");
}

bool Point::getCloseViewObs(
    const Eigen::Vector3d& framepos,
    FramePtr& ref_frame,
    size_t& ref_feature_index) const
{
  double min_cos_angle = 0.0;
  Eigen::Vector3d obs_dir(framepos - pos_);
  obs_dir.normalize();

  //
  //  TODO: For edgelets, find another view that reduces an epipolar line that
  //        is orthogonal to the gradient!
  //

  // TODO: get frame with same point of view AND same pyramid level!
  for(const KeypointIdentifier& obs : obs_)
  {
    if(FramePtr frame = obs.frame.lock())
    {
      Eigen::Vector3d dir(frame->pos() - pos_);
      dir.normalize();
      const double cos_angle = obs_dir.dot(dir);
      if(cos_angle > min_cos_angle)
      {
        min_cos_angle = cos_angle;
        ref_frame = frame;
        ref_feature_index = obs.keypoint_index_;
      }
    }
    else
    {
      SVO_DEBUG_STREAM("could not unlock weak_ptr<Frame> in Point::getCloseViewObs"
                       <<", Point-ID = " << id_
                       <<", Point-nObs = " << obs_.size()
                       <<", Frame-ID = " << obs.frame_id
                       <<", Feature-ID = " << obs.keypoint_index_
                       <<", Point-Type = " << type_);
      return false;
    }
  }
  if(min_cos_angle < 0.4) // assume that observations larger than 60° are useless
  {
    SVO_DEBUG_STREAM("getCloseViewObs(): obs is from too far away: " << min_cos_angle);
    return false;
  }
  return true;
}

double Point::getTriangulationParallax() const
{
  CHECK(!obs_.empty()) << "getTriangualtionParallax(): obs_ is empty!";

  const FramePtr ref_frame = obs_.front().frame.lock();
  if(!ref_frame)
  {
    SVO_ERROR_STREAM("getTriangualtionParallax(): Could not lock ref_frame");
    return 0.0;
  }

  const Eigen::Vector3d r = (ref_frame->pos()-pos_).normalized();
  double max_parallax = 0.0;
  for(const KeypointIdentifier& obs : obs_)
  {
    if(const FramePtr frame = obs.frame.lock())
    {
      const Eigen::Vector3d v = (frame->pos()-pos_).normalized();
      const double parallax = std::acos(r.dot(v));
      max_parallax = std::max(parallax, max_parallax);
    }
  }
  return max_parallax;
}

FramePtr Point::getSeedFrame() const
{
  if(obs_.empty())
    return nullptr;

  // the seed should be in the observation with the smallest id
  if(auto ref_frame = obs_.front().frame.lock())
    return ref_frame;
  else
    SVO_ERROR_STREAM("could not lock weak_ptr<Frame> in point");
  return nullptr;
}

bool Point::triangulateLinear()
{
  const size_t n_obs = obs_.size();
  if(n_obs < 2)
  {
    return false; // not enough measurements to triangulate.
  }
  Eigen::Matrix3Xd f_world; // bearing vectors in world coordinates
  Eigen::Matrix3Xd p_world; // position vectors of cameras
  f_world.resize(Eigen::NoChange, n_obs);
  p_world.resize(Eigen::NoChange, n_obs);
  size_t index = 0;
  for(const KeypointIdentifier& obs : obs_)
  {
    if(const FramePtr frame = obs.frame.lock())
    {
      const Transformation T_world_cam = frame->T_world_cam();
      f_world.col(index) = T_world_cam*frame->f_vec_.col(obs.keypoint_index_);
      p_world.col(index) = T_world_cam.getPosition();
      ++index;
    }
  }
  if(index != n_obs)
  {
    SVO_ERROR_STREAM("triangulateLinear failed, could not lock all frames.");
    return false;
  }

  // from aslam: triangulation.cc
  const Eigen::MatrixXd BiD = f_world *
      f_world.colwise().squaredNorm().asDiagonal().inverse();
  const Eigen::Matrix3d AxtAx = n_obs * Eigen::Matrix3d::Identity() -
      BiD * f_world.transpose();
  const Eigen::Vector3d Axtbx = p_world.rowwise().sum() - BiD *
      f_world.cwiseProduct(p_world).colwise().sum().transpose();

  Eigen::ColPivHouseholderQR<Eigen::Matrix3d> qr = AxtAx.colPivHouseholderQr();
  static constexpr double kRankLossTolerance = 1e-5;
  qr.setThreshold(kRankLossTolerance);
  const size_t rank = qr.rank();
  if (rank < 3) {
    return false; // unobservable
  }
  pos_ = qr.solve(Axtbx);
  return true;
}

void Point::updateHessianGradientUnitPlane(
    const Eigen::Ref<BearingVector>& f,
    const Eigen::Vector3d& p_in_f,
    const Eigen::Matrix3d& R_f_w,
    Eigen::Matrix3d& A,
    Eigen::Vector3d& b,
    double& new_chi2)
{
  svo::Matrix23d J;
  Point::jacobian_xyz2uv(p_in_f, R_f_w, J);
  const Eigen::Vector2d e(vk::project2(f) - vk::project2(p_in_f));
  A.noalias() += J.transpose() * J;
  b.noalias() -= J.transpose() * e;
  new_chi2 += e.squaredNorm();
}

void Point::updateHessianGradientUnitSphere(
    const Eigen::Ref<BearingVector>& f,
    const Eigen::Vector3d& p_in_f,
    const Eigen::Matrix3d& R_f_w,
    Eigen::Matrix3d& A,
    Eigen::Vector3d& b,
    double& new_chi2)
{
  Eigen::Matrix3d J;
  Point::jacobian_xyz2f(p_in_f, R_f_w, J);
  const Eigen::Vector3d e = f - p_in_f.normalized();
  A.noalias() += J.transpose() * J;
  b.noalias() -= J.transpose() * e;
  new_chi2 += e.squaredNorm();
}

void Point::EkfInit() {
    ekf_init = true;
    cov = Eigen::Matrix3d::Identity() * 0.1 * 0.1;
}

void Point::EkfUpdate(const Eigen::VectorXd& dx, double obs_std) {
    if (!ekf_init) {
        EkfInit();
    }

    CHECK(!state_factors.empty());

    CHECK(state_factors.rbegin()->first <= (int)dx.rows());

    Eigen::Vector3d tmp_res = gv;  // calced residual.
    for (svo::StateFactorMap::iterator it = state_factors.begin(); it != state_factors.end(); it++) {
        CHECK(!it->second.state.expired());
        int bias = (*it).first * 6;
        tmp_res -= (*it).second.W.transpose() * dx.segment(bias, 6);
    }

    Eigen::Matrix3d Rcov = V * obs_std * obs_std;
    Eigen::Matrix3d S = V * cov * V.transpose() + Rcov;

    // Eigen::Matrix3d K_T = S.ldlt().solve(V * cov);
    Eigen::MatrixXd K_T = S.inverse() * V * cov;

    Eigen::Matrix3d K = K_T.transpose();

    Eigen::Vector3d delta_x = K * tmp_res;

    pos_ += delta_x;
    CHECK_LT(pos_.norm(), 1e4);

    Eigen::Matrix3d I_KH = Eigen::Matrix3d::Identity() - K * V;
    cov = I_KH * cov;
    Eigen::Matrix3d stable_cov = (cov + cov.transpose()) / 2.0;
    cov = stable_cov;
}

void Point::EkfUpdate(const Eigen::VectorXd& dx, const double obs_std, const double focal_length, const int min_idx,
                      const int max_idx, const double huberA, const double huberB) {
    if (!ekf_init) {
        EkfInit();
    }

    CHECK(!state_factors.empty());
    CHECK(state_factors.rbegin()->first <= (int)dx.rows());

    const double obs_invdev = 1.0 / obs_std;

    // TODO: compute keyframe observation
    for (svo::Point::KeypointIdentifierList::iterator fit = obs_.begin(); fit != obs_.end(); ++fit) {
        if (fit->frame.expired())
            continue;

        const svo::FramePtr& frame = fit->frame.lock();
        const int& curr_frame_idx = frame->bundleId();
        if (curr_frame_idx >= min_idx && curr_frame_idx <= max_idx)  // pass observations on schurvins local sliding window
            continue;

        const Eigen::Vector3d& obs = frame->f_vec_.col(fit->keypoint_index_);
        const Eigen::Matrix3d R_c_w = frame->T_f_w_.getRotationMatrix();
        const Eigen::Vector3d t_c_w = frame->T_f_w_.getPosition();
        const Eigen::Vector3d Pc = R_c_w * pos() + t_c_w;

        // calc residual
        Eigen::Vector2d r
            = (obs.head<2>() / obs.z() - Pc.head<2>() / Pc.z()) * (focal_length);  // unsupport unit sphere
        // LOG(ERROR) << "residual: " << r.transpose();

        // huber
        const double r_l2 = r.squaredNorm();
        double huber_scale = 1.0;
        if (r_l2 > huberB) {
            const double radius = sqrt(r_l2);
            double rho1 = std::max(std::numeric_limits<double>::min(), huberA / radius);
            huber_scale = sqrt(rho1);
            r *= huber_scale;
        }
        r *= obs_invdev;

        // calc jacobian
        Matrix2o3d dr_dpc = Matrix2o3d::Zero();
        const double pc22 = Pc[2] * Pc[2];
        dr_dpc(0, 0) = 1 / Pc[2];
        dr_dpc(1, 1) = 1 / Pc[2];
        dr_dpc(0, 2) = -Pc[0] / pc22;
        dr_dpc(1, 2) = -Pc[1] / pc22;
        dr_dpc *= (focal_length * obs_invdev * huber_scale);

        Matrix2o3d jf = Matrix2o3d::Zero();
        jf.noalias() = dr_dpc * R_c_w;
        V.noalias() += jf.transpose() * jf;  // pt 2 pt
        gv.noalias() += jf.transpose() * r;  // pt grad
    }

    Eigen::Vector3d tmp_res = gv;  // calced residual.
    for (svo::StateFactorMap::iterator it = state_factors.begin(); it != state_factors.end(); it++) {
        CHECK(!it->second.state.expired());
        int bias = (*it).first * 6;
        tmp_res -= (*it).second.W.transpose() * dx.segment(bias, 6);
    }

    Eigen::Matrix3d Rcov = V;
    Eigen::Matrix3d S = V * cov * V.transpose() + Rcov;

    // Eigen::Matrix3d K_T = S.ldlt().solve(V * cov);
    Eigen::MatrixXd K_T = S.inverse() * V * cov;

    Eigen::Matrix3d K = K_T.transpose();

    Eigen::Vector3d delta_x = K * tmp_res;

    pos_ += delta_x;
    CHECK_LT(pos_.norm(), 1e4);

    Eigen::Matrix3d I_KH = Eigen::Matrix3d::Identity() - K * V;
    cov = I_KH * cov;
    Eigen::Matrix3d stable_cov = (cov + cov.transpose()) / 2.0;
    cov = stable_cov;
}

void Point::optimize(const size_t n_iter, bool using_bearing_vector)
{
  Eigen::Vector3d old_point = pos_;
  double chi2 = 0.0;
  Eigen::Matrix3d A;
  Eigen::Vector3d b;

  if(obs_.size() < 2)
  {
    SVO_ERROR_STREAM("optimizing point with less than two observations");
    return;
  }

  const double eps = 0.0000000001;
  for(size_t i=0; i<n_iter; i++)
  {
    A.setZero();
    b.setZero();
    double new_chi2 = 0.0;

    // compute residuals
    for(const KeypointIdentifier& obs : obs_)
    {
      if(const FramePtr& frame = obs.frame.lock())
      {
        if(using_bearing_vector)
        {
          updateHessianGradientUnitSphere(
                frame->f_vec_.col(obs.keypoint_index_), frame->T_f_w_*pos_,
                frame->T_f_w_.getRotation().getRotationMatrix(),
                A, b, new_chi2);
        }
        else
        {
          updateHessianGradientUnitPlane(
                frame->f_vec_.col(obs.keypoint_index_), frame->T_f_w_*pos_,
                frame->T_f_w_.getRotation().getRotationMatrix(),
                A, b, new_chi2);
        }
      }
      else
        SVO_ERROR_STREAM("could not unlock weak_ptr<Frame> in Point::optimize");
    }

    // solve linear system
    const Eigen::Vector3d dp(A.ldlt().solve(b));

    // check if error increased
    if((i > 0 && new_chi2 > chi2) || (bool) std::isnan((double)dp[0]))
    {
#ifdef POINT_OPTIMIZER_DEBUG
      cout << "it " << i
           << "\t FAILURE \t new_chi2 = " << new_chi2 << endl;
#endif
      pos_ = old_point; // roll-back
      break;
    }

    // update the model
    Eigen::Vector3d new_point = pos_ + dp;
    old_point = pos_;
    pos_ = new_point;
    chi2 = new_chi2;
#ifdef POINT_OPTIMIZER_DEBUG
    cout << "it " << i
         << "\t Success \t new_chi2 = " << new_chi2
         << "\t norm(b) = " << vk::norm_max(b)
         << endl;
#endif

    // stop when converged
    if(vk::norm_max(dp) <= eps)
      break;
  }
#ifdef POINT_OPTIMIZER_DEBUG
  cout << endl;
#endif
}

void Point::print(const std::string& s) const
{
  std::cout << s << std::endl;
  std::cout << "  id = " << id_ << std::endl;
  std::cout << "  pos = [" << pos_.transpose() << "]" << std::endl;
  std::cout << "  num obs = " << obs_.size() << std::endl;
}

} // namespace svo
