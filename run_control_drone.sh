docker exec -it uav_bootcamp_ardupilot_navigation-ardupilot-sitl-1 bash -lc \
 "python control_drone_gazebo.py \
 --guided_nogps \
 --no-flow \
 --no-wind \
 --altitude 10; \
 exec bash"