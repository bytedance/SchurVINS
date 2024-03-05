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

#include <svo/common/types.h>

#include <Eigen/Geometry>
#include <list>
#include <memory>
#include <unordered_map>

namespace svo {

class LocalFeature {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LocalFeature(const AugStatePtr& _state, const PointPtr& _point, const FramePtr& _frame, const Eigen::Vector3d& _xyz,
                 const int _camera_id);
    LocalFeature() = delete;

    static bool unit_sphere;
    uint8_t status = 1;
    int camera_id;
    int level = 0;

    AugStateWeakPtr state;
    PointWeakPtr point;
    FramePtr frame = nullptr;

    Matrix2o3d tan_space;
    Vector3d xyz;  // unified sphere space
    Matrix6o3d W;
};

class StateFactor {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Matrix6o3d W;
    AugStateWeakPtr state;
    StateFactor(const Matrix6o3d& _W, const AugStatePtr& _as) : W(_W), state(_as) {
    }
};

class AugState {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    virtual ~AugState() {
    }
    AugState(double _ts, int64_t _id, const Eigen::Quaterniond& _quat, const Eigen::Vector3d& _pos,
             const Eigen::Vector3d& _acc, const Eigen::Vector3d& _gyr)
        : ts(_ts), id(_id), quat(_quat), rot(_quat.toRotationMatrix()), pos(_pos), acc(_acc), gyr(_gyr) {
    }
    AugState() {
    }

    double ts = -1;
    int64_t id = 0;
    int index = 0;

    FrameBundlePtr frame_bundle;

    Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();
    Eigen::Matrix3d rot = Eigen::Matrix3d::Zero();
    Eigen::Vector3d pos = Eigen::Vector3d::Zero();
    Eigen::Vector3d acc = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyr = Eigen::Vector3d::Zero();
    Eigen::Quaterniond quat_fej = Eigen::Quaterniond::Identity();
    Eigen::Vector3d pos_fej = Eigen::Vector3d::Zero();

    std::list<LocalFeaturePtr> features;
};
}  // namespace svo
