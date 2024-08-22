import time

import numpy as np

from bunny_teleop.single_teleop_client import TeleopClient


def main():
    # This example assuming you are using xarm7 with ability hand
    dof = 17
    port_num = 5500
    server_address = "localhost"
    teleop_client = TeleopClient(port=port_num, cmd_dims=dof, host=server_address)

    # Init qpos
    init_qpos = [0] * 17
    init_qpos = np.array(init_qpos)
    robot_pose = np.zeros(7)
    joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7', 'thumb_q1', 'index_q1',
                   'middle_q1', 'ring_q1', 'pinky_q1', 'thumb_q2', 'index_q2', 'middle_q2', 'ring_q2', 'pinky_q2']

    # Teleop initialization
    teleop_client.send_init_config(
        robot_base_pose=robot_pose,
        init_qpos=init_qpos,
        joint_names=joint_names,
    )
    print(f"Begin teleoperation initialization")
    print(f"Place your two hands under your Apple Vision Pro. Make sure you all your fingers are in flatten pose.")

    teleop_client.wait_for_server_start()

    # Teleop start!
    np.set_printoptions(precision=3)
    while True:
        time.sleep(1.0 / 60)
        command = teleop_client.get_teleop_cmd()
        print(f"Receive teleop command for left and right robots: {command}")


if __name__ == '__main__':
    main()
