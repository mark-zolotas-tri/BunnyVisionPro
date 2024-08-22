import h5py
import glob
import os
import threading
import time
from enum import Enum, auto
from pathlib import Path
from pynput import keyboard
import numpy as np

from yourdfpy import URDF
from xarm7_ability import XArm7Ability
from bunny_teleop.single_teleop_client import TeleopClient

TASK_NAME = "single_grasp"
USE_REAL_HAND = True
USE_REAL_ARM = False

SAVE_DATA = False


class DataStatus(Enum):
    NOT_STARTED = auto()
    STARTED = auto()
    FINISHED = auto()
    SAVED = auto()
    ABORTED = auto()


# Demonstration collection
if SAVE_DATA:
    DATA_DIR = Path(__file__).parent.parent / "teleop_data" / TASK_NAME / "robot_data_raw"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    CURR_EPISODE_NUM, CURR_EPISODE = None, None

    def set_new_episode():
        global CURR_EPISODE_NUM, CURR_EPISODE, DATA_DIR
        OLD_EPISODES = sorted(glob.glob(str(DATA_DIR / "episode_*")), key=os.path.getmtime)
        CURR_EPISODE_NUM = int(OLD_EPISODES[-1].split("/")[-1][8:]) + 1 if OLD_EPISODES else 0
        CURR_EPISODE = f"episode_{CURR_EPISODE_NUM}"

        print("Current task:", TASK_NAME)
        print("Current episode:", CURR_EPISODE)

        print("Data will be saved in:", DATA_DIR / CURR_EPISODE)
        print("Press 's' to start saving data, 'q' to finish saving data.")

    set_new_episode()

    data_status = DataStatus.NOT_STARTED

    def on_press(key):
        global data_status
        try:
            if key.char == "n" and data_status == DataStatus.SAVED:
                print(">>> Start new episode")
                set_new_episode()
                data_status = DataStatus.NOT_STARTED
            if key.char == "s" and data_status == DataStatus.NOT_STARTED:
                print(">>> Start capturing data")
                data_status = DataStatus.STARTED
            if key.char == "q" and data_status == DataStatus.STARTED:
                print(">>> Finish capturing data")
                data_status = DataStatus.FINISHED
                print("Press 'n' to start a new episode.")
            if key.char == "a" and data_status == DataStatus.STARTED:
                print(">>> Abort this episode")
                data_status = DataStatus.ABORTED
                print("Press 'n' to start a new episode.")
            print(f"Pressed {key.char}")
        except AttributeError:
            pass  # Ignore special keys

    def start_listener():
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

    # Start the keyboard listener on a separate thread
    listener_thread = threading.Thread(target=start_listener)
    listener_thread.start()


def main():
    asset_path = (Path(__file__).parent.parent / "examples/assets").resolve()

    # Load a yourdfpy instance only for forward kinematics computation
    right_urdf_path = asset_path / "urdf/assembly/xarm7_ability/xarm7_ability_right_hand.urdf"
    right_robot_urdf = URDF.load(str(right_urdf_path))

    right_init_qpos = np.zeros(17)

    client = TeleopClient(port=5500, cmd_dims=(17), host="localhost")
    client.send_init_config(
        robot_base_pose=np.zeros(7), # ignorable...
        init_qpos=right_init_qpos,
        joint_names=right_robot_urdf.actuated_joint_names,
    )

    print("============= Waiting for server to start...")
    client.wait_for_server_start()
    print("============= Server started.")

    if USE_REAL_HAND or USE_REAL_ARM:
        use_servo_control = False

        # Create real robot interface for real control
        right_robot = XArm7Ability(
            use_arm=USE_REAL_ARM,
            use_hand=USE_REAL_HAND,
            arm_ip="192.168.1.xxx",
            hand_tty_index="usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0",
            is_right=True,
            use_servo_control=use_servo_control,
        )

        # Special handling the motion from initialization
        right_robot.reset()

    # Initialization of data collection
    data = {"action": [], "hand0": []}
    timestamps = []
    global data_status, CURR_EPISODE_NUM, CURR_EPISODE

    try:
        right_robot.hand.start_process()

        while True:
            if SAVE_DATA and data_status == DataStatus.STARTED:
                saving_data = True
            else:
                saving_data = False

            qpos_list = client.get_teleop_cmd()
            if saving_data:
                data["action"].append(qpos_list[0].tolist() + qpos_list[1].tolist())
                timestamps.append([time.time()])

            if USE_REAL_ARM or USE_REAL_HAND:
                if saving_data:
                    data["hand0"].append(right_robot.get_hand_data())
                    timestamps[-1].append(time.time())

                # Wait until next control signal, we only need to call this function for a single arm even with two
                right_robot.wait_until_next_control_signal()

                # Control hand
                if USE_REAL_HAND:
                    right_robot.control_hand_qpos(qpos_list[7:])

                # Right robot
                if USE_REAL_ARM:
                    right_robot.control_arm_qpos(qpos_list[0:7])

            if SAVE_DATA and data_status == DataStatus.FINISHED:
                os.makedirs(DATA_DIR / CURR_EPISODE, exist_ok=True)
                with h5py.File(DATA_DIR / CURR_EPISODE / "data.h5", "w") as f:
                    for key, value in data.items():
                        f.create_dataset(key, data=value)
                    f.create_dataset("timestamps", data=timestamps)
                np.save(DATA_DIR / CURR_EPISODE / "timestamps.npy", np.array(timestamps))
                print("Data saved in:", DATA_DIR / CURR_EPISODE)
                data_status = DataStatus.SAVED
                data = {"action": [], "hand0": []}
                timestamps = []

            if SAVE_DATA and data_status == DataStatus.ABORTED:
                data_status = DataStatus.SAVED
                data = {"action": [], "hand0": []}
                timestamps = []
                print("Episode aborted.")

    except KeyboardInterrupt:
        print("Keyboard interrupt, shutting down.\n")
        right_robot.stop()

        # Wait for the listener to finish
        listener_thread.join()


if __name__ == "__main__":
    main()
