FROM robocin/ardupilot-sitl-gazebo:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    && pip3 install MAVProxy pymavlink \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY ./control_drone_gazebo.py /root/ardu_ws/control_drone_gazebo.py
COPY ./nogps_control_drone_gazebo.py /root/ardu_ws/nogps_control_drone_gazebo.py
COPY ./ardupilot_gz_gazebo /root/ardu_ws/src/ardupilot_gz/ardupilot_gz_gazebo
COPY ./ardupilot_gz_bringup /root/ardu_ws/src/ardupilot_gz/ardupilot_gz_bringup
COPY ./iris_with_gimbal /root/ardu_ws/src/ardupilot_gz/ardupilot_gz_description/models/iris_with_gimbal
COPY ./aerial_ground /root/ardu_ws/src/ardupilot_gz/ardupilot_gz_description/models/aerial_ground
COPY ./optical_flow /root/ardu_ws/optical_flow
COPY ./optical_flow_ros2_subscriber.py /root/ardu_ws/optical_flow_ros2_subscriber.py
COPY ./constants.py /root/ardu_ws/constants.py

# Make the copied models resolvable. The world uses model://iris_with_gimbal and
# model://aerial_ground (incl. its albedo texture); those resolve only if the
# models' parent dir is on GZ_SIM_RESOURCE_PATH. No hook in this repo adds it
# (ardupilot_gz_gazebo only exports worlds/), so point at the source dir we just
# COPYed into — guaranteed present regardless of colcon install rules. Sourcing
# install/setup.bash at runtime prepends the package paths, keeping this as base.
ENV GZ_SIM_RESOURCE_PATH=/root/ardu_ws/src/ardupilot_gz/ardupilot_gz_description/models

RUN bash -c "source /opt/ros/humble/setup.bash && \
                colcon build --packages-select \
                ardupilot_gz_bringup \
                ardupilot_gz_description \
                ardupilot_gz_gazebo \
                ardupilot_gz_application \
                ardupilot_sitl_models"

RUN echo "source /root/ardu_ws/install/setup.bash" >> /root/.bashrc
