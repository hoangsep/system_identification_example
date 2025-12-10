FROM osrf/ros:noetic-desktop-full

# Force Mesa software rendering (avoids host GPU issues inside container)
ENV LIBGL_ALWAYS_SOFTWARE=1

# 1. Install System Dependencies
# - ackermann-msgs: Required for car-like steering commands
# - ros-control: Required for the POLARIS simulator hardware interface
# - git/wget: For cloning repositories
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-tk \
    ros-noetic-ackermann-msgs \
    ros-noetic-ros-control \
    ros-noetic-ros-controllers \
    ros-noetic-gazebo-ros-control \
    ros-noetic-hector-gazebo-plugins \
    ros-noetic-joy \
    ros-noetic-jsk-rviz-plugins \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python Dependencies for SysID and MPC
# - pandas/numpy: Data processing
# - matplotlib: Plotting the RMSE and Errors
# - torch: Neural Networks for System Identification
# - casadi: The industry standard optimization solver for MPC (works great with Python)
# - scipy: General scientific computing
RUN pip3 install --no-cache-dir \
    numpy \
    pandas \
    matplotlib \
    scipy \
    scikit-learn \
    casadi \
    torch \
    torchvision

# 3. Setup Catkin Workspace
RUN mkdir -p /root/catkin_ws/src
WORKDIR /root/catkin_ws

# 4. Add source setup to bashrc so you don't have to type it every time
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc
RUN echo "source /root/catkin_ws/devel/setup.bash" >> /root/.bashrc

# 5. Build the empty workspace initially to verify setup
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

# Entrypoint
CMD ["bash"]
