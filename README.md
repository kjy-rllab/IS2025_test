# IS_RTS

ros2 run rccar_bringup RTS_online --mode train
ros2 run rccar_bringup RTS_project3 --mode val

ros2 topic pub --once /query message/msg/Query "{id: '0', team: 'RTS', map: 'map1', trial: 0, exit: false}"