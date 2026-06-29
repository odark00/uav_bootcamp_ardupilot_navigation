FROM robocin/ardupilot-sitl-gazebo:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    && pip3 install MAVProxy pymavlink \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY ./control_drone_gazebo.py /root/ardu_ws/control_drone_gazebo.py
COPY ./copter.launch.py /root/ardu_ws/src/ardupilot_gz/ardupilot_gz_bringup/launch/copter.launch.py
COPY ./iris_camera.launch.py /root/ardu_ws/src/ardupilot_gz/ardupilot_gz_bringup/launch/robots/iris_camera.launch.py
COPY ./iris_with_camera /root/ardu_ws/src/ardupilot_gz/ardupilot_gz_description/models

RUN bash -c "source /opt/ros/humble/setup.bash && \
                colcon build --packages-select \
                ardupilot_gz_bringup \
                ardupilot_gz_description \
                ardupilot_gz_gazebo \
                ardupilot_gz_application \
                ardupilot_sitl_models"
