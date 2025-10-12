from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from dphand.mujoco.mujoco_render import Viewer, OSViewer


class MujocoGymEnv(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 50}
    
    def __init__(
        self,
        xml_path: Path,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = float("inf"),
        image_size: int = 128,
        render_mode: Literal["rgb_array", "human"] = "rgb_array"
    ):
        # self.metadata["render_fps"] = int(np.round(1.0 / self.dt))

        super().__init__(xml_path.as_posix(), 
                         frame_skip=int(control_dt // physics_dt), 
                         observation_space=None, 
                         render_mode=render_mode
                        )
        
        self._control_dt = control_dt
        self._time_limit = time_limit
        self._random = np.random.RandomState(seed)
        
        self._image_size = image_size
        self._viewer = None
        self.render_mode = render_mode

        self._init_viewer()

    def _init_viewer(self):
        if self.render_mode == "human":
            self._viewer = Viewer( # 交互式渲染
                self.model,
                self.data,
                width=1280,
                height=960,
                img_obs_width=self._image_size,
                img_obs_height=self._image_size,
            )
        elif self.render_mode == "rgb_array":
            self._viewer = OSViewer( # 离屏渲染
                self.model,
                self.data,
                img_obs_width=self._image_size,
                img_obs_height=self._image_size,
            )


    def time_limit_exceeded(self) -> bool:
        return self.data.time >= self._time_limit

    # Accessors.
    @property
    def control_dt(self) -> float:
        return self._control_dt

    @property
    def physics_dt(self) -> float:
        return self.model.opt.timestep

    @property
    def random_state(self) -> np.random.RandomState:
        return self._random
    
    def render(self):
        if self.render_mode == "human":
            self._viewer.render()
        elif self.render_mode == "rgb_array":
            return self._viewer.render_rgb_cam("rgb_array", -1)

    def close(self):
        self._viewer.close()
        self._viewer = None