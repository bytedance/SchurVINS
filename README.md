# SchurVINS
## SchurVINS: Schur Complement-Based Lightweight Visual Inertial Navigation System
We propose a filter-based VINS framework named SchurVINS, which guarantees both high accuracy by building a complete residual model and low computational complexity with Schur complement. Technically, we first formulate the full residual model where Gradient, Hessian and observation covariance are explicitly modeled. Then Schur complement is employed to decompose the full model into ego-motion residual model and landmark residual model. Finally, Extended Kalman Filter (EKF) update is implemented in these two models with high efficiency. Experiments on EuRoC and TUM-VI datasets show that SchurVINS notably outperforms state-of-the-art (SOTA) methods in both accuracy and computational complexity. The main contributions include:
- An equivalent residual model is proposed to deal with hyper high-dimension observations, which consists of gradient, Hessian and the corresponding observation covariance. This method is of great generality in EKF systems.
- A lightweight EKF-based landmark solver is proposed to
estimate position of landmarks with high efficiency.
- A novel EKF-based VINS framework is developed to
achieve ego-motion and landmark estimation simultaneously with high accuracy and efficiency.

## 1. License

The code is licensed under GPLv3.

The SchurVINS is developed on SVO2.0(**[rpg_svo_pro_open](https://github.com/uzh-rpg/rpg_svo_pro_open)**), and thus its license is retained at the beginning of the related files.

**Related Publication:**  
Yunfei Fan, Tianyu Zhao, Guidong Wang. SchurVINS: Schur Complement-Based Lightweight Visual Inertial Navigation System. (Accepted by CVPR 2024).**[PDF](https://arxiv.org/pdf/2312.01616.pdf)**.  

## 2. Prerequisites
We have tested the codebase in **Ubuntu 18.04**.  
The following dependencies are needed:

### System dep.
```
# For Ubuntu 18.04 + Melodic
sudo apt-get install python-catkin-tools python-vcstool
sudo apt-get install libglew-dev libopencv-dev libyaml-cpp-dev 
```


### Ceres dep.

```
sudo apt-get install libblas-dev liblapack-dev libsuitesparse-dev
```

## 3. Build

Clone and build the repository:
```
cd ~/catkin_ws/src
git clone https://github.com/bytedance/SchurVINS.git
source ~/catkin_ws/devel/setup.bash
mkdir -p ~/catkin_ws/src/SchurVINS/results
mkdir -p ~/catkin_ws/src/SchurVINS/logs
vcs-import < ./SchurVINS/dependencies.yaml
touch minkindr/minkindr_python/CATKIN_IGNORE
catkin build
```

## 4. Run
We provide examples to run SchurVINS with [EuRoC dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets#downloads). 

Open three terminals, launch the schur_vins , rviz and play the bag file respectively. Take MH_01 for example 

```
roslaunch svo_ros euroc_vio_stereo.launch
rviz -d svo_ros/rviz_config.rviz
rosbag play YOUR_PATH_TO_DATASET/MH_01_easy.bag 
```
# 5. Reproducibility
There might be minor differences between the released version and the results in the paper. Please note that multi-thread performance has some randomness due to CPU utilization.
On EuRoC datasets, please note to skip the first few seconds of datasets with the following configurations for reproducibility.


|||||||||||
|----|------|----|----|------|----|----|------|----|----|
|**MH1**|**MH2**|**MH3**|**MH4**|**MH5**|**V11**|**V12**|**V13**|**V21**|**V22**|
|30s|30s|10s|13s|15s|0s|0s|2s|0s|0s|
|||||||||||

# 6. Security

If you discover a potential security issue in this project, or think you may
have discovered a security issue, we ask that you notify Bytedance Security via our [security center](https://security.bytedance.com/src) or [vulnerability reporting email](sec@bytedance.com).

# 7. Acknowledgement
This work incorporates the well-known SVO2.0 open-source code. We extend our gratitude to the authors of the software.
- [rpg_svo_pro_open](https://github.com/uzh-rpg/rpg_svo_pro_open)


# 8. Citation
If you found this code/work to be useful in your own research, please considering citing the following information. Additionally, please considering citing SVO2.0([rpg_svo_pro_open](https://github.com/uzh-rpg/rpg_svo_pro_open)) since SchurVINS is developed on it.
```bibtex
@inproceedings{fan2023schurvins,
  title={SchurVINS: Schur Complement-Based Lightweight Visual Inertial Navigation System},
  author={Yunfei Fan, Tianyu Zhao, Guidong Wang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```

# 9. We are Hiring!
Our team is hiring FTEs with background in Deep Learning, SLAM, and 3D Vision. We are based in Beijing and Shanghai. If you are interested, please send your resume to frank.01[AT]bytedance[DOT]com.
