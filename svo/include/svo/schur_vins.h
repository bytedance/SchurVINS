// Copyright 2024 ByteDance and/or its affiliates.
/*
This program is free software: you can redistribute it and/or modify it 
under the terms of the GNU General Public License as published by the 
Free Software Foundation, either version 3 of the License, or (at your 
option) any later version.

This program is distributed in the hope that it will be useful, but 
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY 
or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for 
more details.

You should have received a copy of the GNU General Public License along 
with this program. If not, see <https://www.gnu.org/licenses/>.
*/
#pragma once

#include <svo/common/frame.h>
#include <svo/global.h>

#include <condition_variable>
#include <fstream>
#include <iostream>
#include <set>

namespace schur_vins {

class SchurVINS {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    SchurVINS();

    void InitImuModel(double acc_n, double acc_w, double gyr_n, double gyr_w);
    void InitExtrinsic(const svo::CameraBundle::Ptr camera_bundle);
    void InitCov();
    void InitMaxState(int val);
    void InitFocalLength(double val);
    void InitObsStddev(double _obs_dev);
    void InitChi2(double chi2_rate);

    void SetKeyframe(const bool _is_keyframe);
    void Forward(const svo::FrameBundle::Ptr frame_bundle);
    int Backward(const svo::FrameBundle::Ptr frame_bundle);
    bool StructureOptimize(const svo::PointPtr& optimize_points);

   private:
    class ImuState {
       public:
        using Ptr = std::shared_ptr<ImuState>;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        ImuState() {
        }
        Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();
        Eigen::Vector3d pos = Eigen::Vector3d::Zero();
        Eigen::Vector3d vel = Eigen::Vector3d::Zero();
        Eigen::Vector3d ba = Eigen::Vector3d::Zero();
        Eigen::Vector3d bg = Eigen::Vector3d::Zero();
        Eigen::Quaterniond quat_fej = Eigen::Quaterniond::Identity();
        Eigen::Vector3d pos_fej = Eigen::Vector3d::Zero();
        Eigen::Vector3d vel_fej = Eigen::Vector3d::Zero();
        Eigen::Vector3d acc = Eigen::Vector3d::Zero();
        Eigen::Vector3d gyr = Eigen::Vector3d::Zero();
        double ts = -1;
        int64_t id = 0;
    };

    void InitState(double _ts, const Eigen::Vector3d& _acc, const Eigen::Vector3d& _gyr);
    void AugmentState(const svo::FrameBundle::Ptr frame_bundle);
    void PredictionState(const Eigen::Vector3d& acc, const Eigen::Vector3d& gyro, double dt);
    void Prediction(double _ts, const Eigen::Vector3d& _acc, const Eigen::Vector3d& _gyr);
    void Management(int index);
    void ManageLocalMap();
    bool KeyframeSelect();
    void StateUpdate(const Eigen::MatrixXd& hessian, const Eigen::VectorXd& gradient);
    void StateUpdate2(const Eigen::MatrixXd& hessian, const Eigen::VectorXd& gradient);
    // bool Chi2Check(const Eigen::MatrixXd& j, const Eigen::VectorXd& r, int dof);
    void StateCorrection(const Eigen::MatrixXd& K, const Eigen::MatrixXd& J, const Eigen::VectorXd& dX,
                         const Eigen::MatrixXd& R);
    void RegisterPoints(const svo::FrameBundle::Ptr& frame_bundle);

    void Solve3();
    int RemoveOutliers(const svo::FrameBundle::Ptr frame_bundle);
    int RemovePointOutliers();

   private:
    std::mutex msg_queue_mtx;
    std::condition_variable con;

    svo::StateMap states_map;
    ImuState::Ptr curr_state = nullptr;
    svo::CameraPtr cam0_param = nullptr, cam1_param = nullptr;
    svo::Transformation T_imu_cam0, T_imu_cam1;
    bool stereo = false;
    svo::LocalPointMap local_pts;

    // imu prediction
    Eigen::Vector3d prev_acc = Eigen::Vector3d::Zero();
    Eigen::Vector3d prev_gyr = Eigen::Vector3d::Zero();
    double prev_imu_ts = -1;

    Matrix12d imu_noise = Matrix12d::Zero();
    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(15, 15);
    int64_t id_creator = 0;
    Eigen::Vector3d gravity = Eigen::Vector3d(0, 0, -9.81007);
    Eigen::VectorXd dsw_;
    std::set<svo::PointPtr> schur_pts_;

    int state_max = 4;
    int curr_fts_cnt = 0;
    double focal_length = 1000;
    bool zupt_valid = false;
    double obs_dev = 1;
    double obs_invdev = 1;
    double huberA = 1.5;
    double huberB = huberA * huberA;
    std::vector<double> chi_square_lut;
};

namespace Utility {
inline Eigen::Matrix3d SkewMatrix(const Eigen::Vector3d& w) {
    Eigen::Matrix3d mat;
    mat << 0, -w.coeff(2), w.coeff(1), w.coeff(2), 0, -w.coeff(0), -w.coeff(1), w.coeff(0), 0;
    return mat;
}
}  // namespace Utility
}  // namespace schur_vins