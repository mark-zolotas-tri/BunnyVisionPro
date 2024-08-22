import pickle
import threading
import time
from typing import List

import numpy as np
import zmq
from tornado import ioloop
from zmq.eventloop import zmqstream

from bunny_teleop.init_config import (
    SingleInitializationConfig,
)


class TeleopClient:
    def __init__(self, port: int, cmd_dims: int, host="localhost"):
        # Create socket
        self.ctx = zmq.Context()
        if host == "localhost":
            sub_bind_to = f"tcp://localhost:{port}"
        else:
            sub_bind_to = f"tcp://{host}:{port}"
        self.init_bind_to = ":".join(
            sub_bind_to.split(":")[:-1] + [str(int(sub_bind_to.split(":")[-1]) + 1)]
        )
        self.sub_bind_to = sub_bind_to
        self.sub_socket = None

        self.cmd_dim = cmd_dims

        # Setup background IO loop
        self._loop = None
        self._started = threading.Event()
        self._stream = None
        self._thread = threading.Thread(target=self.run)
        self._thread.daemon = True
        self._thread.start()

        # Multi-thread variable
        self._lock = threading.Lock()
        self._shared_most_recent_teleop_cmd = np.zeros(cmd_dims)
        self._shared_most_recent_ee_pose = np.zeros(7)
        self._shared_server_started = False

    def send_init_config(
            self,
            *,
            robot_base_pose: np.ndarray,
            init_qpos: np.ndarray,
            joint_names: List[str],
    ):
        # TODO: check dof and init_qpos match
        init_socket = self.ctx.socket(zmq.REQ)
        init_socket.connect(self.init_bind_to)
        init_config = SingleInitializationConfig(
            robot_base_pose=robot_base_pose,
            init_qpos=init_qpos,
            joint_names=joint_names,
        )
        init_socket.send_json(init_config.to_dict())
        with self._lock:
            self._shared_server_started = False
        init_socket.close()
        
    def update_teleop_cmd(self, message):
        cmd = pickle.loads(message[0])
        target_qpos = cmd["target_qpos"]
        ee_pose = cmd["ee_pose"]
        if not self.started:
            print(f"Teleop Client: Teleop Server start, begin teleoperation now.")
            with self._lock:
                self._shared_server_started = True
                self._shared_most_recent_teleop_cmd[:] = target_qpos[:]
                self._shared_most_recent_ee_pose[:] = ee_pose[:]
        else:
            if not isinstance(target_qpos, np.ndarray) or not isinstance(
                    target_qpos, np.ndarray
            ):
                raise ValueError(
                    f"Teleop client: Invalid command: qpos dim: {target_qpos.shape}, cmd dim: {self.cmd_dim}"
                )
            with self._lock:
                self._shared_most_recent_teleop_cmd[:] = target_qpos[:]
                self._shared_most_recent_ee_pose[:] = ee_pose[:]

    def get_teleop_cmd(self):
        with self._lock:
            return self._shared_most_recent_teleop_cmd

    def get_ee_pose(self):
        with self._lock:
            return self._shared_most_recent_ee_pose

    @property
    def started(self):
        with self._lock:
            return self._shared_server_started

    def wait_for_server_start(self):
        try:
            while not self.started:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Keyboard interrupt, shutting down.\n")
            exit()

    def run(self):
        self._loop = ioloop.IOLoop()
        self._loop.initialize()
        self._loop.make_current()
        self.sub_socket = self.ctx.socket(zmq.SUB)
        self._stream = zmqstream.ZMQStream(
            self.sub_socket, io_loop=ioloop.IOLoop.current()
        )

        # Wait for server start
        self.sub_socket.connect(self.sub_bind_to)
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"")

        self._stream.on_recv(self.update_teleop_cmd)
        self._started.set()
        self._loop.start()
