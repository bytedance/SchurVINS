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

#include <svo/common/local_feature.h>

namespace svo {
bool LocalFeature::unit_sphere = false;

LocalFeature::LocalFeature(const AugStatePtr& _state, const PointPtr& _point, const svo::FramePtr& _frame,
                           const Eigen::Vector3d& _xyz, const int _camera_id)
    : camera_id(_camera_id), state(_state), point(_point), frame(_frame), xyz(_xyz) {
    if (LocalFeature::unit_sphere) {
        Eigen::Vector3d b1, b2;
        Eigen::Vector3d tmp(0, 0, 1);
        if (std::abs(xyz[2]) > 0.8)
            tmp << 1, 0, 0;
        b1 = (tmp.cross(xyz)).normalized();
        b2 = xyz.cross(b1);
        tan_space.topRows(1) = b1.transpose();
        tan_space.bottomRows(1) = b2.transpose();
    } else {
        xyz /= xyz.z();
    }
    W.setZero();
}

}  // namespace svo