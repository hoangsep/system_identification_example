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
    python3-dev \
    build-essential \
    cmake \
    gfortran \
    swig \
    libblas-dev \
    liblapack-dev \
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
    jinja2 \
    torch \
    torchvision

# 3. Install acados (C library + Python template) to a common path
ENV ACADOS_SOURCE_DIR=/opt/acados
RUN git clone --depth 1 --recurse-submodules https://github.com/acados/acados.git ${ACADOS_SOURCE_DIR} \
    && cd ${ACADOS_SOURCE_DIR} \
    && mkdir -p build \
    && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${ACADOS_SOURCE_DIR} -DACADOS_WITH_OPENMP=ON -DACADOS_WITH_QPOASES=ON \
    && make -j"$(nproc)" \
    && make install \
    && pip3 install -e ${ACADOS_SOURCE_DIR}/interfaces/acados_template
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ACADOS_SOURCE_DIR}/lib
ENV PYTHONPATH=${PYTHONPATH}:${ACADOS_SOURCE_DIR}/interfaces/acados_template

# 4. Setup Catkin Workspace
RUN mkdir -p /root/catkin_ws/src
WORKDIR /root/catkin_ws

# 5. Add source setup to bashrc so you don't have to type it every time
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc
RUN echo "source /root/catkin_ws/devel/setup.bash" >> /root/.bashrc

# 6. Build the empty workspace initially to verify setup
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

# Entrypoint
CMD ["bash"]
