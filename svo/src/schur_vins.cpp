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
#include <svo/schur_vins.h>

#include <boost/math/distributions/chi_squared.hpp>
#include <opencv2/optflow.hpp>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace schur_vins {

SchurVINS::SchurVINS() : curr_state(new ImuState()) {
    // LOG(INFO) << "init begins";
}

void SchurVINS::InitCov() {
    double quat_cov = 0.0001;
    double pos_cov = 0.001;
    double vel_cov = 0.001;
    double ba_cov = 2e-3;
    double bg_cov = 1e-6;
    cov = Eigen::MatrixXd::Zero(15, 15);
    cov.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * quat_cov;
    cov.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * pos_cov;
    cov.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * vel_cov;
    cov.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * ba_cov;
    cov.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity() * bg_cov;
}

void SchurVINS::InitImuModel(double acc_n, double acc_w, double gyr_n, double gyr_w) {
    imu_noise = Matrix12d::Zero();
    // LOG(INFO) << "acc_n: " << acc_n << ", acc_w: " << acc_w << ", gyr_n: " << gyr_n << ", gyr_w: " << gyr_w;
    imu_noise.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * gyr_n * gyr_n;  // acc_n * acc_n;
    imu_noise.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * acc_n * acc_n;  // acc_w * acc_w;
    imu_noise.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * acc_w * acc_w;  // gyr_n * gyr_n;
    imu_noise.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * gyr_w * gyr_w;  // gyr_w * gyr_w;
}

void SchurVINS::InitExtrinsic(const svo::CameraBundle::Ptr camera_bundle) {
    CHECK(camera_bundle.get() != nullptr);
    int camera_num = camera_bundle->getNumCameras();
    cam0_param = camera_bundle->getCameraShared(0);
    T_imu_cam0 = camera_bundle->get_T_C_B(0).inverse();
    // LOG(INFO) << "cam0 extrinsic, q_i_c:\n" << T_imu_cam0;
    if (camera_num > 1) {
        cam0_param = camera_bundle->getCameraShared(1);
        T_imu_cam1 = camera_bundle->get_T_C_B(1).inverse();
        stereo = true;
        // LOG(INFO) << "cam1 extrinsic, q_i_c:\n" << T_imu_cam1;
    }
}

void SchurVINS::InitMaxState(int val) {
    state_max = val;
    // LOG(INFO) << "state_max: " << state_max;
}

void SchurVINS::InitFocalLength(double val) {
    focal_length = val;
    // LOG(INFO) << "focal_length: " << focal_length;
}

void SchurVINS::InitObsStddev(double _obs_dev) {
    obs_dev = _obs_dev;
    obs_invdev = 1.0 / obs_dev;
    // LOG(INFO) << "obs_dev: " << obs_dev;
}

void SchurVINS::InitChi2(double chi2_rate) {
    chi_square_lut.clear();
    const int max_dof = 50;
    std::vector<double> lut(max_dof + 1, 0.0);
    for (int i = 1; i <= max_dof; ++i) {
        boost::math::chi_squared chi_square_dist(i);
        lut.at(i) = boost::math::quantile(chi_square_dist, chi2_rate);
    }
    lut.swap(chi_square_lut);
    // LOG(INFO) << "chi2_rate: " << chi2_rate << ", max_dof: " << max_dof;
}

void SchurVINS::AugmentState(const svo::FrameBundle::Ptr frame_bundle) {
    CHECK_EQ(frame_bundle->getMinTimestampSeconds(), curr_state->ts);

    curr_state->id = id_creator++;
    svo::AugStatePtr aug_ptr(new svo::AugState(curr_state->ts, curr_state->id, curr_state->quat, curr_state->pos,
                                               curr_state->acc, curr_state->gyr));
    aug_ptr->quat_fej = curr_state->quat_fej;
    aug_ptr->pos_fej = curr_state->pos_fej;
    aug_ptr->frame_bundle = frame_bundle;
    states_map.emplace(aug_ptr->id, aug_ptr);

    CHECK((int)states_map.size() <= state_max);
    size_t old_rows = cov.rows();
    size_t old_cols = cov.cols();
    CHECK(old_rows == old_cols);
    CHECK(old_rows == (15 + states_map.size() * 6 - 6));
    cov.conservativeResize(old_rows + 6, old_cols + 6);

    cov.block(old_rows, 0, 6, old_cols) = cov.block(0, 0, 6, old_cols);
    cov.block(0, old_cols, old_rows, 6) = cov.block(0, 0, old_rows, 6);
    cov.block(old_rows, old_cols, 6, 6) = cov.block<6, 6>(0, 0);
}

void SchurVINS::PredictionState(const Eigen::Vector3d& acc, const Eigen::Vector3d& gyro, double dt) {
    const Eigen::Vector3d wd_full = gyro * dt;
    const Eigen::Vector3d wd_half = wd_full * 0.5;
    Eigen::Quaterniond dq_half = Eigen::Quaterniond(1, wd_half(0) / 2, wd_half(1) / 2, wd_half(2) / 2);
    Eigen::Quaterniond dq_full = Eigen::Quaterniond(1, wd_full(0) / 2, wd_full(1) / 2, wd_full(2) / 2);
    dq_half.normalize();
    dq_full.normalize();

    const Eigen::Quaterniond q_half = curr_state->quat * dq_half;
    const Eigen::Quaterniond q_full = curr_state->quat * dq_full;

    // k1
    const Eigen::Vector3d k1_dv = curr_state->quat * acc + gravity;
    const Eigen::Vector3d k1_dp = curr_state->vel;
    // LOG(INFO) << "k1_dv: " << k1_dv[0] << ", " << k1_dv[1] << ", " << k1_dv[2];

    // k2
    const Eigen::Vector3d k1_v = curr_state->vel + k1_dv * dt / 2;
    const Eigen::Vector3d k2_dv = q_half * acc + gravity;
    const Eigen::Vector3d k2_dp = k1_v;

    // k3
    const Eigen::Vector3d k2_v = curr_state->vel + k2_dv * dt / 2;
    const Eigen::Vector3d k3_dv = q_half * acc + gravity;
    const Eigen::Vector3d k3_dp = k2_v;

    // k4
    const Eigen::Vector3d k3_v = curr_state->vel + k3_dv * dt;
    const Eigen::Vector3d k4_dv = q_full * acc + gravity;
    const Eigen::Vector3d k4_dp = k3_v;

    // output
    curr_state->quat = q_full;
    curr_state->pos += dt / 6 * (k1_dp + 2 * k2_dp + 2 * k3_dp + k4_dp);
    curr_state->vel += dt / 6 * (k1_dv + 2 * k2_dv + 2 * k3_dv + k4_dv);
}

void SchurVINS::Prediction(double _dt, const Eigen::Vector3d& _acc, const Eigen::Vector3d& _gyr) {
    CHECK(_dt >= 0 && _dt < 0.015) << "dt: " << _dt;
    Eigen::Vector3d acc = _acc - curr_state->ba;
    Eigen::Vector3d gyr = _gyr - curr_state->bg;

    const Eigen::Matrix3d rot = curr_state->quat.toRotationMatrix();
    Matrix15d F = Matrix15d::Zero();
    Eigen::Matrix<double, 15, 12> G = Eigen::Matrix<double, 15, 12>::Zero();
    // Q, P, V, Ba, Bg
    F.block<3, 3>(0, 12) = -rot;
    F.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity();
    F.block<3, 3>(6, 0) = -Utility::SkewMatrix(rot * acc);
    F.block<3, 3>(6, 9) = -rot;

    G.block<3, 3>(0, 0) = -rot;
    G.block<3, 3>(6, 3) = -rot;
    G.block<3, 3>(9, 6) = Eigen::Matrix3d::Identity();
    G.block<3, 3>(12, 9) = Eigen::Matrix3d::Identity();

    Matrix15d Fdt = F * _dt;
    Matrix15d Fdt_square = Fdt * Fdt;
    Matrix15d Fdt_cube = Fdt_square * Fdt;
    Matrix15d Phi = Matrix15d::Identity() + Fdt + 0.5 * Fdt_square + (1.0 / 6.0) * Fdt_cube;

    // state prediction, 4th order Runge-Kutta
    PredictionState(acc, gyr, _dt);

    // fej impl
    // {
    //     Eigen::Matrix3d rr = (curr_state->quat * curr_state->quat_fej.inverse()).toRotationMatrix();
    //     Eigen::Matrix3d yy = - Utility::SkewMatrix(curr_state->pos - curr_state->pos_fej - curr_state->vel_fej * _dt
    //     - 0.5 * gravity * _dt * _dt); Eigen::Matrix3d ss = - Utility::SkewMatrix(curr_state->vel -
    //     curr_state->vel_fej - gravity * _dt); Phi.block<3, 3>(0, 0) = rr; Phi.block<3, 3>(3, 0) = yy; Phi.block<3,
    //     3>(6, 0) = ss;
    // }

    cov.block(0, 0, 15, 15)
        = Phi * cov.block(0, 0, 15, 15) * Phi.transpose() + Phi * G * imu_noise * G.transpose() * Phi.transpose() * _dt;

    if (!states_map.empty()) {
        int cov_len = cov.rows();
        cov.block(0, 15, 15, cov_len - 15) = Phi * cov.block(0, 15, 15, cov_len - 15);
        cov.block(15, 0, cov_len - 15, 15) = cov.block(15, 0, cov_len - 15, 15) * Phi.transpose();
    }
    Eigen::MatrixXd stable_cov = (cov + cov.transpose()) / 2.0;
    cov = stable_cov;

    curr_state->quat_fej = curr_state->quat;
    curr_state->pos_fej = curr_state->pos;
    curr_state->vel_fej = curr_state->vel;

    curr_state->ts += _dt;
    curr_state->acc = acc;
    curr_state->gyr = gyr;
}

void SchurVINS::Solve3() {
    const int state_size = (int)states_map.size();
    if (state_size < 2) {
        LOG(INFO) << "only one window, bypass solve()";
        return;
    }

    const int min_frame_idx = states_map.begin()->second->frame_bundle->getBundleId();
    const int prev_frame_id0 = (++states_map.rbegin())->second->frame_bundle->frames_[0]->id(),
              prev_frame_id1 = (++states_map.rbegin())->second->frame_bundle->frames_[1]->id();
    const int max_frame_idx = states_map.rbegin()->second->frame_bundle->getBundleId();

    const int state_len = state_size * 6;
    const int64_t curr_state_id = curr_state->id;
    Eigen::MatrixXd Amtx = Eigen::MatrixXd::Zero(state_len, state_len);
    Eigen::VectorXd Bvct = Eigen::VectorXd::Zero(state_len);
    Matrix2o3d dr_dpc = Matrix2o3d::Zero();
    Matrix3o6d dpc_dpos = Matrix3o6d::Zero();
    Matrix2o6d jx = Matrix2o6d::Zero();
    Matrix2o3d jf = Matrix2o3d::Zero();
    Eigen::Vector2d r = Eigen::Vector2d::Zero();

    // compute local points jacobian
    int num_obs = 0;
    double total_error = 0.0;
    schur_pts_.clear();
    for (svo::LocalPointMap::iterator pit = local_pts.begin(); pit != local_pts.end(); ++pit) {
        const svo::PointPtr& curr_pt = pit->second;
        if (curr_pt->register_id_ != curr_state_id) {  // init point jacobian
            curr_pt->gv.setZero();
            curr_pt->V.setZero();
            curr_pt->W.setZero();
            curr_pt->register_id_ = curr_state_id;  // avoid init multi times
        }

        if (curr_pt->CheckStatus() == false || curr_pt->CheckLocalStatus() == false
            || curr_pt->CheckLocalStatus(prev_frame_id0, prev_frame_id1, max_frame_idx) == false) {
            // LOG(INFO) << "Solve2 pass invalid point: " << curr_pt->id()
            //           << " feature num : " << curr_pt->local_obs_.size() << " pos: " <<
            //           curr_pt->pos().transpose();
            continue;
        }

        schur_pts_.insert(curr_pt);
        const Eigen::Vector3d& Pw = curr_pt->pos();

        // compute frame local observation
        for (svo::LocalFeatureMap::iterator fit = curr_pt->local_obs_.begin(); fit != curr_pt->local_obs_.end();
             ++fit) {
            const svo::LocalFeaturePtr& feature = fit->second;
            const int curr_frame_idx = feature->frame->bundleId();

            if (curr_frame_idx < min_frame_idx
                || curr_frame_idx > max_frame_idx)  // pass observations on schurvins local sliding window
                continue;

            if (feature->state.expired())
                continue;

            const svo::AugStatePtr& ft_state = feature->state.lock();
            const int& state_idx = ft_state->index;
            const svo::Transformation& T_imu_cam = feature->camera_id == 0 ? T_imu_cam0 : T_imu_cam1;
            const Eigen::Matrix3d& R_i_c = T_imu_cam.getRotationMatrix();
            const Eigen::Quaterniond& q_i_c = T_imu_cam.getEigenQuaternion();
            const Eigen::Vector3d& t_i_c = T_imu_cam.getPosition();
            const Eigen::Matrix3d R_c_w = (ft_state->quat * q_i_c).toRotationMatrix().transpose();
            const Eigen::Vector3d Pi = ft_state->quat.inverse() * (Pw - ft_state->pos);
            const Eigen::Vector3d Pc = q_i_c.inverse() * (Pi - t_i_c);
            // const double level_scale = (1 << feature->level);
            const double level_scale = 1;

            // calc residual
            if (svo::LocalFeature::unit_sphere) {
                r = feature->tan_space * (feature->xyz - Pc.normalized()) * (focal_length / level_scale);
            } else {
                r = (feature->xyz.head<2>() - Pc.head<2>() / Pc.z()) * (focal_length / level_scale);
            }
            total_error += r.norm();
            // LOG(INFO) << "kf residual: " << r.transpose();

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
            dr_dpc.setZero();
            if (svo::LocalFeature::unit_sphere) {
                double Pc_norm = Pc.norm();
                double inv_norm = 1.0 / Pc_norm;
                double inv_norm3 = 1.0 / (Pc_norm * Pc_norm * Pc_norm);
                Eigen::Matrix3d norm_jacob = Eigen::Matrix3d::Identity() * inv_norm - Pc * Pc.transpose() * inv_norm3;
                dr_dpc = feature->tan_space * norm_jacob * (focal_length * obs_invdev * huber_scale / level_scale);
            } else {
                const double pc22 = Pc[2] * Pc[2];
                dr_dpc(0, 0) = 1 / Pc[2];
                dr_dpc(1, 1) = 1 / Pc[2];
                dr_dpc(0, 2) = -Pc[0] / pc22;
                dr_dpc(1, 2) = -Pc[1] / pc22;
                dr_dpc *= (focal_length * obs_invdev * huber_scale / level_scale);
            }

            ++num_obs;
            dpc_dpos.leftCols(3).noalias()
                = R_i_c.transpose() * Utility::SkewMatrix(Pi) * ft_state->quat.toRotationMatrix().transpose();
            dpc_dpos.rightCols(3) = -R_c_w;

            // dpc_dext.leftCols(3).noalias() = Utility::SkewMatrix(Pc) * R_i_c.transpose();
            // dpc_dext.rightCols(3) = -R_i_c.transpose();

            // jext.noalias() = dr_dpc * dpc_dext;
            jx.noalias() = dr_dpc * dpc_dpos;
            jf.noalias() = dr_dpc * R_c_w;
            // LOG(INFO) << "jext:\n" << jext;
            // LOG(INFO) << "jx:\n" << jx;
            // LOG(INFO) << "jf:\n" << jf;

            ////////////////////////////////////////////////////////////
            // Amtx.block(0, 0, 6, 6).noalias() += jext.transpose() * jext;  // ext 2 ext
            // Matrix6d blk_ext2pos = jext.transpose() * jx;
            // const int pos_bias = 6 + ft->state->index * 6;
            const int pos_bias = state_idx * 6;
            // Amtx.block(0, pos_bias, 6, 6) += blk_ext2pos;                           // ext 2 pos
            // Amtx.block(pos_bias, 0, 6, 6) += blk_ext2pos.transpose();               // pos 2 ext
            Amtx.block(pos_bias, pos_bias, 6, 6).noalias() += jx.transpose() * jx;  // pos 2 pos

            // Bvct.segment(0, 6).noalias() += jext.transpose() * r;       // ext grad
            Bvct.segment(pos_bias, 6).noalias() += jx.transpose() * r;  // pos grad

            curr_pt->V.noalias() += jf.transpose() * jf;  // pt 2 pt
            curr_pt->gv.noalias() += jf.transpose() * r;  // pt grad
            // curr_pt->W.noalias() += jext.transpose() * jf;  // ext 2 pt
            feature->W.noalias() = jx.transpose() * jf;  // pos 2 pt
                                                         // LOG(INFO) << "V:\n" << curr_pt->V;
                                                         // LOG(INFO) << "Vinv:\n" << curr_pt->V.inverse();
                                                         // LOG(INFO) << "gv:\n" << curr_pt->gv.transpose();
                                                         // LOG(INFO) << "W:\n" << ft->W;
        }
    }

    // combine point observation on same body
    for (const svo::PointPtr& pt : schur_pts_) {
        pt->state_factors.clear();
        for (svo::LocalFeatureMap::iterator iti = pt->local_obs_.begin(); iti != pt->local_obs_.end(); iti++) {
            if (iti->second->state.expired())
                continue;
            const svo::AugStatePtr& ft_state = iti->second->state.lock();
            svo::StateFactorMap::iterator iter = pt->state_factors.find(ft_state->index);
            if (iter == pt->state_factors.end()) {
                pt->state_factors.emplace(ft_state->index, svo::StateFactor(iti->second->W, ft_state));
            } else {
                iter->second.W += iti->second->W;
            }
        }
    }

    // schur completment
    for (const svo::PointPtr& pt : schur_pts_) {
        pt->Vinv = pt->V.inverse();
        // LOG(INFO) << "pt->V:\n" << pt->V;
        // LOG(INFO) << "pt->Vinv:\n" << pt->Vinv;
        // const Matrix6o3d eWVinv = pt->W * pt->Vinv;
        // LOG(INFO) << "eWVinv:\n" << eWVinv;
        // Amtx.block(0, 0, 6, 6) -= eWVinv * pt->W.transpose();  // ext 2 ext schur
        // Bvct.segment(0, 6) -= eWVinv * pt->gv;                 // ext grad schur
        for (svo::StateFactorMap::iterator iti = pt->state_factors.begin(); iti != pt->state_factors.end(); iti++) {
            if (iti->second.state.expired())
                continue;
            const svo::AugStatePtr& statei = iti->second.state.lock();  // int pi_bias = 6 + statei->index * 6;
            const int pi_bias = statei->index * 6;

            // const Matrix6d e2p_schur = eWVinv * fti->W.transpose();  // ext 2 pos schur
            // Amtx.block(0, pi_bias, 6, 6) -= e2p_schur;               //
            // Amtx.block(pi_bias, 0, 6, 6) -= e2p_schur.transpose();   //

            // LOG(INFO) << "e2p_schur:\n" << e2p_schur;
            // LOG(INFO) << "Amtx:\n" << Amtx;

            const Matrix6o3d WVinv = iti->second.W * pt->Vinv;
            for (svo::StateFactorMap::iterator itj = iti; itj != pt->state_factors.end(); itj++) {
                if (itj->second.state.expired())
                    continue;
                const svo::AugStatePtr& statej = itj->second.state.lock();
                // int pj_bias = 6 + statej->index * 6;
                const int pj_bias = statej->index * 6;

                Matrix6d p2p_schur = WVinv * itj->second.W.transpose();  // posi 2 posj schur
                Amtx.block(pi_bias, pj_bias, 6, 6) -= p2p_schur;
                if (pi_bias != pj_bias) {
                    Amtx.block(pj_bias, pi_bias, 6, 6) -= p2p_schur.transpose();
                }
                // LOG(INFO) << "Amtx:\n" << Amtx;
            }
            Bvct.segment(pi_bias, 6) -= WVinv * pt->gv;  // pos grad schur
            // LOG(INFO) << "Bvct:\n" << Bvct.transpose();
        }
    }

    // LOG(INFO) << "Amtx:\n" << Amtx;
    // LOG(INFO) << "Bvct:\n" << Bvct;
    StateUpdate(Amtx, Bvct);

    // LOG(ERROR) << "total local pts: " << local_pts.size() << " schur pts: " << schur_pts_.size();
}

void SchurVINS::StateUpdate(const Eigen::MatrixXd& hessian, const Eigen::VectorXd& gradient) {
    // LOG(INFO) << "StateUpdate:\n";
    // LOG(INFO) << "hessian: " << hessian;
    // LOG(INFO) << "gradient: " << gradient.transpose();

    int rows = (int)hessian.rows();
    CHECK(rows == (int)states_map.size() * 6);
    const Eigen::MatrixXd& R = hessian;

    Eigen::MatrixXd Jacob_compress = Eigen::MatrixXd::Zero(rows, rows + 15);

    Jacob_compress.block(0, 15, rows, rows) = hessian;

    Eigen::MatrixXd S = hessian * cov.bottomRightCorner(rows, rows) * hessian.transpose() + R;
    Eigen::MatrixXd K_T = S.ldlt().solve(hessian * cov.bottomRows(rows));
    Eigen::MatrixXd K = K_T.transpose();
    // std::cout << "K: " << S.rows() << " " << S.cols() << " " << hessian.rows() << " " << hessian.cols() << std::endl;
    // std::cout << "K: " << cov.rows() << " " << cov.cols() << " " <<cov.bottomRows(rows).rows() << " " <<
    // cov.bottomRows(rows).cols() << std::endl;

    // Eigen::MatrixXd K = S.inverse() * hessian * cov.bottomRows(rows);

    Eigen::VectorXd delta_x = K * gradient;

    StateCorrection(K, Jacob_compress, delta_x, R);
    dsw_ = delta_x.tail(delta_x.rows() - 15);
}

void SchurVINS::StateCorrection(const Eigen::MatrixXd& K, const Eigen::MatrixXd& J, const Eigen::VectorXd& dX,
                            const Eigen::MatrixXd& R) {
    // Update the IMU state.
    // quat, pos, vel, ba, bg

    const Vector15d& dx_imu = dX.head(15);
    const Eigen::Quaterniond dq_imu = Eigen::Quaterniond(1.0, 0.5 * dx_imu[0], 0.5 * dx_imu[1], 0.5 * dx_imu[2]);
    Eigen::Quaterniond new_quat = dq_imu * curr_state->quat;
    new_quat.normalize();
    curr_state->quat = new_quat;
    curr_state->pos += dx_imu.segment<3>(3);
    curr_state->vel += dx_imu.segment<3>(6);
    curr_state->ba += dx_imu.segment<3>(9);
    curr_state->bg += dx_imu.segment<3>(12);

    int idx = 0;
    for (svo::StateMap::value_type& item : states_map) {
        const Vector6d& dAX = dX.segment<6>(15 + idx);
        Eigen::Quaterniond dq = Eigen::Quaterniond(1.0, 0.5 * dAX[0], 0.5 * dAX[1], 0.5 * dAX[2]);
        dq.normalize();

        item.second->quat = dq * item.second->quat;
        item.second->rot = item.second->quat.toRotationMatrix();
        item.second->pos += dAX.tail<3>();
        idx += 6;
    }

    Eigen::MatrixXd I_KH = Eigen::MatrixXd::Identity(K.rows(), K.rows()) - K * J;
    cov = I_KH * cov;
    Eigen::MatrixXd stable_cov = (cov + cov.transpose()) / 2.0;
    cov = stable_cov;
}

void SchurVINS::Management(int index) {
    CHECK((int)states_map.size() == state_max);

    int src_idx = 0;
    int dst_idx = 0;
    int cov_len = cov.rows();
    while (src_idx < state_max && dst_idx < state_max) {
        if (src_idx == index) {
            src_idx++;
            continue;
        }
        if (src_idx != dst_idx) {
            cov.block(dst_idx * 6 + 15, 0, 6, cov_len) = cov.block(src_idx * 6 + 15, 0, 6, cov_len);
            cov.block(0, dst_idx * 6 + 15, cov_len, 6) = cov.block(0, src_idx * 6 + 15, cov_len, 6);
        }
        src_idx++;
        dst_idx++;
    }
    cov.conservativeResize(dst_idx * 6 + 15, dst_idx * 6 + 15);

    int idx = 0;
    for (auto it = states_map.begin(); it != states_map.end();) {
        if (idx++ == index) {
            for (auto& ft : it->second->features) {
                if (const svo::PointPtr& pt = ft->point.lock()) {
                    int64_t state_id = it->second->id;
                    pt->local_obs_.erase(state_id);
                }
            }
            it->second->features.clear();
            it = states_map.erase(it);
            break;
        } else {
            it++;
        }
    }
    CHECK(((int)states_map.size() * 6 + 15) == cov.rows()) << states_map.size() << ", " << cov.rows();
    ManageLocalMap();
}

void SchurVINS::ManageLocalMap() {
    for (svo::LocalPointMap::iterator it = local_pts.begin(); it != local_pts.end();) {
        svo::PointPtr& pt = it->second;
        if (pt->local_obs_.empty()) {
            it = local_pts.erase(it);
        }
        // else if(pt->features.rbegin()->first != curr_state->id) {
        //     pt->FeaturesClear();
        //     it = local_pts.erase(it);
        // }
        else {
            it++;
        }
    }
}

bool SchurVINS::KeyframeSelect() {
    bool delete_r2th = false;
    // false - delete the oldest keyframe;
    // true - delete the second new keyframe.
    CHECK((int)states_map.size() == state_max)
        << "states_map.size: " << states_map.size() << ", state_max: " << state_max;

    std::unordered_set<int64_t> tail_set;

    size_t rindex = 0;
    int common_cnt = 0;
    for (svo::StateMap::reverse_iterator rit = states_map.rbegin(); rit != states_map.rend();) {
        if (1 == rindex) {
            for (const svo::LocalFeaturePtr& ft : rit->second->features) {
                if (const svo::PointPtr& point = ft->point.lock()) {
                    CHECK(point);
                    tail_set.insert(point->id());
                }
            }
        }

        if (2 == rindex) {
            for (svo::LocalFeaturePtr& ft : rit->second->features) {
                if (const svo::PointPtr& point = ft->point.lock()) {
                    CHECK(point);
                    if (tail_set.find(point->id()) != tail_set.end()) {
                        common_cnt++;
                    }
                }
            }
            break;
        }
        rit++;
        rindex++;
    }
    curr_fts_cnt = (int)tail_set.size();
    double track_rate = common_cnt / (curr_fts_cnt + 1e-6);
    if (curr_fts_cnt < 10 || track_rate < 0.70) {
        delete_r2th = false;
    } else {
        delete_r2th = true;
    }
    // LOG(ERROR) << "delete_r2th: " << delete_r2th << ", track_rate: " << track_rate
    //            << ", curr_fts_cnt: " << curr_fts_cnt;

    return delete_r2th;
}

void SchurVINS::RegisterPoints(const svo::FrameBundle::Ptr& frame_bundle) {
    CHECK(!states_map.empty());
    auto& state_ptr = states_map.rbegin()->second;
    const int state_id = state_ptr->id;
    for (const svo::FramePtr& frame : frame_bundle->frames_) {
        const int camera_id = frame->getNFrameIndex();
        int num_feature = 0;
        for (size_t i = 0; i < frame->numFeatures(); ++i) {
            svo::PointPtr point = frame->landmark_vec_[i];
            if (point != nullptr && svo::isCorner(frame->type_vec_[i])) {
                const svo::LocalFeaturePtr ft
                    = std::make_shared<svo::LocalFeature>(state_ptr, point, frame, frame->f_vec_.col(i), camera_id);
                ft->level = frame->level_vec_[i];
                svo::LocalPointMap::iterator iter = local_pts.find(point->id());
                if (iter == local_pts.end()) {
                    local_pts.insert({point->id(), point});
                    state_ptr->features.emplace_back(ft);
                    point->local_obs_.insert({state_id, ft});
                } else {
                    state_ptr->features.emplace_back(ft);
                    point->local_obs_.insert({state_id, ft});
                }
                ++num_feature;
            }
        }
        // LOG(INFO) << "RegisterPoints camera id: " << camera_id << " num feature: " << num_feature;
    }
}

void SchurVINS::InitState(double _ts, const Eigen::Vector3d& _acc, const Eigen::Vector3d& _gyr) {
    if (curr_state->ts > 0)
        return;

    Eigen::Vector3d acc_norm = _acc / _acc.norm();
    Eigen::Vector3d grav_norm = Eigen::Vector3d(0, 0, 1);
    // LOG(INFO) << "acc: " << _acc[0] << ", " << _acc[1] << ", " << _acc[2] << ", gyr: " << _gyr[0] << ", " << _gyr[1]
    //           << ", " << _gyr[2];

    Eigen::Quaterniond quat = Eigen::Quaterniond::FromTwoVectors(acc_norm, grav_norm);
    curr_state->ts = _ts;
    curr_state->quat = quat;
    // LOG(INFO) << "quat.rot: " << quat.coeffs().transpose();
    curr_state->pos.setZero();
    curr_state->vel.setZero();
    curr_state->ba.setZero();
    curr_state->bg.setZero();
    // LOG(INFO) << "InitState: " << curr_state->ts << ", quat: " << curr_state->quat.w() << ", " <<
    // curr_state->quat.x()
    //           << ", " << curr_state->quat.y() << ", " << curr_state->quat.z() << ", "
    //           << "pos: " << curr_state->pos[0] << ", " << curr_state->pos[1] << ", " << curr_state->pos[2] << ", "
    //           << "vel: " << curr_state->vel[0] << ", " << curr_state->vel[1] << ", " << curr_state->vel[2] << ", "
    //           << "ba: " << curr_state->ba[0] << ", " << curr_state->ba[1] << ", " << curr_state->ba[2] << ", "
    //           << "bg: " << curr_state->bg[0] << ", " << curr_state->bg[1] << ", " << curr_state->bg[2];

    curr_state->quat_fej = curr_state->quat;
    curr_state->pos_fej = curr_state->pos;
    curr_state->vel_fej = curr_state->vel;
    // Eigen::Quaterniond quat2 = Eigen::Quaterniond::FromTwoVectors(grav_norm, acc_norm);
    // std::cout << "quat2.rot: " << std::endl << quat2.toRotationMatrix() << std::endl;
}

void SchurVINS::SetKeyframe(const bool _is_keyframe) {
    const int L1 = 0, R2 = state_max - 2;
    if ((int)states_map.size() == state_max) {
        if (_is_keyframe == true) {
            // LOG(ERROR) << "keyframe select outside";
            const int idx = R2;  // delete r2 frame when curr frame is keyframe
            Management(idx);
        } else {
            // LOG(ERROR) << "keyframe select insside";
            const int idx = KeyframeSelect() ? R2 : L1;
            Management(idx);
        }
    }
}

void SchurVINS::Forward(const svo::FrameBundle::Ptr frame_bundle) {
    constexpr double EPS = 1e-9;

    // LOG(WARNING) << "schurvins Forward";

    double img_ts = frame_bundle->getMinTimestampSeconds();
    for (size_t i = 0; i < frame_bundle->imu_datas_.size(); i++) {
        const double tmp_ts = frame_bundle->imu_datas_[i].timestamp_;
        const Eigen::Vector3d tmp_acc = frame_bundle->imu_datas_[i].linear_acceleration_;
        const Eigen::Vector3d tmp_gyr = frame_bundle->imu_datas_[i].angular_velocity_;

        if (prev_imu_ts < 0) {
            prev_acc = tmp_acc;
            prev_gyr = tmp_gyr;
            prev_imu_ts = tmp_ts;
        }

        if (curr_state->ts < 0) {
            InitState(prev_imu_ts, prev_acc, prev_gyr);
        }

        if (tmp_ts < curr_state->ts + EPS) {
            // LOG(WARNING) << std::fixed << "pass imu data: " << tmp_ts;
            continue;
        }

        if (tmp_ts < img_ts + EPS) {
            double deltaT = tmp_ts - curr_state->ts;
            CHECK(deltaT >= 0) << "deltaT: " << deltaT;

            Prediction(deltaT, prev_acc, prev_gyr);

            prev_acc = tmp_acc;
            prev_gyr = tmp_gyr;
            prev_imu_ts = tmp_ts;
        } else {  // Interpolation img time imu data
            double dt_1 = img_ts - curr_state->ts;
            double dt_2 = tmp_ts - img_ts;
            CHECK(dt_1 >= 0) << "dt_1: " << dt_1;
            CHECK(dt_2 >= 0) << "dt_2: " << dt_2;
            CHECK(dt_1 + dt_2 > 0) << "dt_1 + dt_2: " << dt_1 + dt_2;
            double w1 = dt_2 / (dt_1 + dt_2);
            double w2 = dt_1 / (dt_1 + dt_2);

            Prediction(dt_1, prev_acc, prev_gyr);

            prev_acc = w1 * prev_acc + w2 * tmp_acc;
            prev_gyr = w1 * prev_gyr + w2 * tmp_gyr;
            prev_imu_ts = img_ts;
        }
    }

    // update result
    svo::Transformation T_world_imu = svo::Transformation(curr_state->quat, curr_state->pos);
    for (const svo::FramePtr& frame : frame_bundle->frames_) {
        frame->T_f_w_ = frame->T_cam_imu() * T_world_imu.inverse();
    }

    {
        Eigen::Quaterniond quat = curr_state->quat;
        Eigen::Vector3d pos = curr_state->pos;
        Eigen::Vector3d vel = curr_state->vel;
        Eigen::Vector3d ba = curr_state->ba;
        Eigen::Vector3d bg = curr_state->bg;
        // LOG(INFO) << "schurvins forward: " << std::fixed << std::setprecision(6) << curr_state->ts << ", quat: " <<
        // quat.w()
        //           << ", " << quat.x() << ", " << quat.y() << ", " << quat.z() << ", "
        //           << "pos: " << pos[0] << ", " << pos[1] << ", " << pos[2] << ", "
        //           << "vel: " << vel[0] << ", " << vel[1] << ", " << vel[2] << ", "
        //           << "ba: " << ba[0] << ", " << ba[1] << ", " << ba[2] << ", "
        //           << "bg: " << bg[0] << ", " << bg[1] << ", " << bg[2];
        // LOG(INFO) << "gravity: " << gravity[0] << ", " << gravity[1] << ", " << gravity[2];
    }
}

int SchurVINS::Backward(const svo::FrameBundle::Ptr frame_bundle) {
    AugmentState(frame_bundle);
    RegisterPoints(frame_bundle);

    int idx = 0;
    std::for_each(states_map.begin(), states_map.end(),
                  [&](svo::StateMap::value_type& it) { it.second->index = idx++; });

    Solve3();

    // update result
    for (const svo::StateMap::value_type& item : states_map) {
        const svo::Transformation T_world_imu = svo::Transformation(item.second->quat, item.second->pos);
        const svo::FrameBundle::Ptr& tmp_frame_bundle = item.second->frame_bundle;
        for (const svo::FramePtr& frame : tmp_frame_bundle->frames_) {
            frame->T_f_w_ = frame->T_cam_imu() * T_world_imu.inverse();
        }
    }

    int num_valid_feature = 0;
    if (states_map.size() == 1) {
        num_valid_feature = states_map.rbegin()->second->features.size();
    } else {
        num_valid_feature = RemoveOutliers(frame_bundle);  // remove curr frame outlier first
        RemovePointOutliers();                             // remove local point outlier second
    }

    {
        Eigen::Quaterniond quat = curr_state->quat;
        Eigen::Vector3d pos = curr_state->pos;
        Eigen::Vector3d vel = curr_state->vel;
        Eigen::Vector3d ba = curr_state->ba;
        Eigen::Vector3d bg = curr_state->bg;
        // LOG(INFO) << "schurvins backward: " << std::fixed << std::setprecision(6) << curr_state->ts
        //           << ", quat: " << quat.w() << ", " << quat.x() << ", " << quat.y() << ", " << quat.z() << ", "
        //           << "pos: " << pos[0] << ", " << pos[1] << ", " << pos[2] << ", "
        //           << "vel: " << vel[0] << ", " << vel[1] << ", " << vel[2] << ", "
        //           << "ba: " << ba[0] << ", " << ba[1] << ", " << ba[2] << ", "
        //           << "bg: " << bg[0] << ", " << bg[1] << ", " << bg[2];
        // LOG(INFO) << "gravity: " << gravity[0] << ", " << gravity[1] << ", " << gravity[2];
    }

    return num_valid_feature;
}

bool SchurVINS::StructureOptimize(const svo::PointPtr& optimize_point) {
    if (schur_pts_.count(optimize_point)) {
        const int min_idx = states_map.begin()->second->frame_bundle->getBundleId();
        const int max_idx = states_map.rbegin()->second->frame_bundle->getBundleId();
        optimize_point->EkfUpdate(dsw_, obs_dev, focal_length, min_idx, max_idx, huberA, huberB);
        // optimize_point->EkfUpdate(dsw_, obs_dev);

        return true;
    }
    // LOG(ERROR) << "miss schur point: " << optimize_point->local_obs_.size()
    //            << " key obs: " << optimize_point->obs_.size() << " pos: " << optimize_point->pos().norm()
    //            << " local status: " << (int)optimize_point->local_status_;

    return false;
}

int SchurVINS::RemovePointOutliers() {
    constexpr double MAX_REPROJECT_ERROR = 3.0;

    const int min_frame_idx = states_map.begin()->second->frame_bundle->getBundleId();
    const int max_frame_idx = states_map.rbegin()->second->frame_bundle->getBundleId();

    const int state_len = states_map.size() * 6;
    const int64_t curr_state_id = curr_state->id;

    int outlier_points_num = 0, total_points_num = 0;
    for (svo::LocalPointMap::iterator pit = local_pts.begin(); pit != local_pts.end(); ++pit) {
        const svo::PointPtr& curr_pt = pit->second;
        if (curr_pt->CheckStatus() == false) {
            // LOG(INFO) << "Solve2 pass invalid point: " << curr_pt->id()
            //           << " feature num : " << curr_pt->local_obs_.size() << " pos: " <<
            //           curr_pt->pos().transpose();
            continue;
        }
        double total_error = 0.0, num_obs = 0.0;
        const Eigen::Vector3d Pw = curr_pt->pos();

        // compute frame local observation
        for (svo::LocalFeatureMap::iterator fit = curr_pt->local_obs_.begin(); fit != curr_pt->local_obs_.end();
             ++fit) {
            const svo::LocalFeaturePtr& feature = fit->second;
            const int curr_frame_idx = feature->frame->bundleId();

            if (curr_frame_idx < min_frame_idx
                || curr_frame_idx > max_frame_idx)  // pass observations on schurvins local sliding window
                continue;

            if (feature->state.expired())
                continue;

            // const svo::AugStatePtr& ft_state = feature->state.lock();
            // const int state_idx = ft_state->index;
            // const svo::Transformation& T_imu_cam = feature->camera_id == 0 ? T_imu_cam0 : T_imu_cam1;
            // const Eigen::Matrix3d R_i_c = T_imu_cam.getRotationMatrix();
            // const Eigen::Quaterniond q_i_c = T_imu_cam.getEigenQuaternion();
            // const Eigen::Vector3d t_i_c = T_imu_cam.getPosition();
            // const Eigen::Matrix3d R_c_w = (ft_state->quat * q_i_c).toRotationMatrix().transpose();
            // const Eigen::Vector3d Pw = curr_pt->pos();
            // const Eigen::Vector3d Pi = ft_state->quat.inverse() * (Pw - ft_state->pos);
            const Eigen::Vector3d Pc = feature->frame->T_f_w_ * curr_pt->pos();
            // const double level_scale = (1 << feature->level);
            const double level_scale = 1;

            // calc residual
            const Eigen::Vector2d r = (feature->xyz.head<2>() - Pc.head<2>() / Pc.z()) * (focal_length / level_scale);
            total_error += r.norm();
            ++num_obs;
        }
        const double avg_error = total_error / num_obs;
        if (avg_error > MAX_REPROJECT_ERROR) {
            ++outlier_points_num;
            curr_pt->local_status_ = false;
            // curr_pt->last_structure_optim_ = -1;
        } else {
            curr_pt->local_status_ = true;
        }
    }

    // LOG(ERROR) << "outlier_points_num: " << outlier_points_num << "    ";

    return outlier_points_num;
}

int SchurVINS::RemoveOutliers(const svo::FrameBundle::Ptr frame_bundle) {
    constexpr double MAX_REPROJECT_ERROR = 4.0;
    double total_error = 0;
    const svo::AugStatePtr& state = states_map.rbegin()->second;

    int num_valid_feature = 0, num_invalid_feature = 0;
    for (const svo::FramePtr& frame : frame_bundle->frames_) {
        // const Eigen::Matrix3d R_i_c = frame->T_imu_cam().getRotationMatrix();
        // const Eigen::Quaterniond q_i_c = frame->T_imu_cam().getEigenQuaternion();
        // const Eigen::Vector3d t_i_c = frame->T_imu_cam().getPosition();
        // const Eigen::Matrix3d R_c_w = (state->quat * q_i_c).toRotationMatrix().transpose();

        // const Eigen::Matrix3d R_c_w = frame->T_f_w_.getRotationMatrix();
        // const Eigen::Vector3d t_c_w = frame->T_f_w_.getPosition();
        for (size_t i = 0; i < frame->numFeatures(); ++i) {
            const svo::PointPtr curr_pt = frame->landmark_vec_[i];
            const Eigen::Vector3d obs = frame->f_vec_.col(i);
            // const double level_scale = (1 << frame->level_vec_[i]);
            const double level_scale = 1;
            if (curr_pt && local_pts.find(curr_pt->id()) != local_pts.end()) {
                if (curr_pt->CheckStatus() == false)
                    continue;

                const Eigen::Vector3d Pc = frame->T_f_w_ * curr_pt->pos();
                const Eigen::Vector2d r
                    = (obs.head<2>() / obs.z() - Pc.head<2>() / Pc.z()) * (focal_length / level_scale);
                total_error += r.norm();

                if (r.norm() > MAX_REPROJECT_ERROR) {
                    // curr_pt->RemoveLocalObs(state->id, frame->getNFrameIndex());
                    frame->type_vec_[i] = svo::FeatureType::kOutlier;
                    frame->seed_ref_vec_[i].keyframe.reset();
                    frame->landmark_vec_[i] = nullptr;  // delete landmark observation
                    num_invalid_feature++;
                    // LOG(ERROR) << "RemoveOutliers residual: " << r.transpose()
                    //            << " camera id: " << frame->getNFrameIndex();
                } else {
                    num_valid_feature++;
                }
            }
        }
    }

    // LOG(INFO) << "RemoveOutliers total error: " << total_error << " num_total_feature: " << state->features.size()
    //           << " num_valid_feature: " << num_valid_feature << " num_invalid_feature: " << num_invalid_feature;

    return num_valid_feature;
}

}  // namespace schur_vins
