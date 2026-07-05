docker exec -it uav_bootcamp_ardupilot_navigation-ardupilot-sitl-1 bash -lc \
 "source /opt/ros/humble/setup.bash && \
 python control_drone_gazebo.py \
 --guided_nogps \
 --no-wind \
 --altitude 10; \
 exec bash"