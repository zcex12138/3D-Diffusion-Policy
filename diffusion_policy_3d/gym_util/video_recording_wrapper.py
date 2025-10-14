import gymnasium as gym
import numpy as np
from termcolor import cprint


class SimpleVideoRecordingWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            steps_per_render=1,
            camera_id=1,
            render_size=(320, 240),
        ):
        """
        When file_path is None, don't record.
        """
        super().__init__(env)
        
        self.steps_per_render = steps_per_render
        self.camera_id = camera_id
        self.step_count = 0
        self.render_size = render_size

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.frames = list()

        frame = self.env.unwrapped._viewer.render_rgb_cam("rgb_array", self.camera_id, size=self.render_size)
        assert frame.dtype == np.uint8
        self.frames.append(frame)
        
        self.step_count = 1
        return obs
    
    def step(self, action):
        result = super().step(action)
        self.step_count += 1
        
        frame = self.env.unwrapped._viewer.render_rgb_cam("rgb_array", self.camera_id, size=self.render_size) if (self.step_count % self.steps_per_render == 0) else self.frames[-1]
        assert frame.dtype == np.uint8
        self.frames.append(frame)
        
        return result
    
    def get_video(self):
        video = np.stack(self.frames, axis=0) # (T, H, W, C)
        # to store as mp4 in wandb, we need (T, H, W, C) -> (T, C, H, W)
        video = video.transpose(0, 3, 1, 2)
        return video

