# FROM osrf/ros:noetic-desktop-full # only for amd64
FROM ghcr.io/sloretz/ros:noetic-desktop
SHELL ["/bin/bash", "-c"]

# =========================================================
# =========================================================

# Are you are looking for how to use this docker file?
#   - https://docs.openvins.com/dev-docker.html
#   - https://docs.docker.com/get-started/
#   - http://wiki.ros.org/docker/Tutorials/Docker

# =========================================================
# =========================================================

# Dependencies we use, catkin tools is very good build system
# Also some helper utilities for fast in terminal edits (nano etc)
# Remove outdated ROS apt entries to avoid GPG failure on first update
RUN rm -f /etc/apt/sources.list.d/ros* /etc/apt/trusted.gpg.d/ros* || true
RUN apt-get update && apt-get install -y curl ca-certificates gnupg
RUN curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list
RUN apt-get update && apt-get install -y libeigen3-dev nano git
RUN apt-get update && apt-get install -y python3-catkin-tools python3-osrf-pycommon

# Ceres solver install and setup
RUN apt-get update && apt-get install -y cmake libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev libsuitesparse-dev libceres-dev
# ENV CERES_VERSION="2.0.0"
# RUN git clone https://ceres-solver.googlesource.com/ceres-solver && \
#     cd ceres-solver && \
#     git checkout tags/${CERES_VERSION} && \
#     mkdir build && cd build && \
#     cmake .. && \
#     make -j$(nproc) install && \
#     rm -rf ../../ceres-solver

# Seems this has Python 3.8 installed on it...
RUN apt-get update && apt-get install -y python3-dev python3-matplotlib python3-numpy python3-psutil python3-tk

# Install deps needed for clion remote debugging
# https://blog.jetbrains.com/clion/2020/01/using-docker-with-clion/
# RUN sed -i '6i\source "/catkin_ws/devel/setup.bash"\' /ros_entrypoint.sh
RUN apt-get update && apt-get install -y openssh-server build-essential gcc g++ \
    gdb clang cmake rsync tar python3 && apt-get clean
RUN ( \
    echo 'LogLevel DEBUG2'; \
    echo 'PermitRootLogin yes'; \
    echo 'PasswordAuthentication yes'; \
    echo 'Subsystem sftp /usr/lib/openssh/sftp-server'; \
  ) > /etc/ssh/sshd_config_test_clion \
  && mkdir /run/sshd
RUN useradd -m user && yes password | passwd user
RUN usermod -s /bin/bash user
CMD ["/usr/sbin/sshd", "-D", "-e", "-f", "/etc/ssh/sshd_config_test_clion"]

RUN apt-get update && apt-get install -y python3-vcstool vim clangd python3-catkin-tools jq unzip tmux
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc
RUN echo "source /catkin_ws/devel/setup.bash" >> /root/.bashrc
# RUN git clone https://github.com/dorian3d/DBoW2.git
RUN git clone https://gitclone.com/github.com/dorian3d/DBoW2.git
# RUN wget https://github.com/dorian3d/DBoW2/archive/refs/tags/v1.1-free.zip -O DBoW2.zip && unzip DBoW2.zip -d /DBoW2 && rm DBoW2.zip
RUN mkdir -p /catkin_ws/src/
WORKDIR /catkin_ws/src/
# RUN git clone https://github.com/bytedance/SchurVINS.git
RUN git clone https://github.com/byheng/SchurVINS.git -b aarch64

# RUN git clone https://gitclone.com/github.com/byheng/SchurVINS.git -b aarch64
RUN vcs-import < SchurVINS/dependencies.yaml
RUN touch /catkin_ws/src/minkindr/minkindr_python/CATKIN_IGNORE
RUN sed -i 's|GIT_REPOSITORY git@github.com:dorian3d/DBoW2.git|GIT_REPOSITORY /DBoW2|' /catkin_ws/src/dbow2_catkin/CMakeLists.txt
RUN mkdir -p /catkin_ws/src/SchurVINS/results && mkdir -p /catkin_ws/src/SchurVINS/logs

RUN apt install ros-noetic-pcl-ros python3-pip zip -y && pip install evo
WORKDIR /catkin_ws
RUN catkin config --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
# RUN source /opt/ros/noetic/setup.bash && catkin build
RUN jq -s 'add' $(find build -name compile_commands.json) > compile_commands.json
