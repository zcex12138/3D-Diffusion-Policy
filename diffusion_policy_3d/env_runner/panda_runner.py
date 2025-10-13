import wandb
import numpy as np
import torch
import collections
import tqdm
from dphand.envs.pick_and_place_env import PickAndPlaceEnv
from diffusion_policy_3d.env import DphandEnvWrapper
from diffusion_policy_3d.gym_util_dphand.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util_dphand.video_recording_wrapper import SimpleVideoRecordingWrapper

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util
from termcolor import cprint

import visualizer

visualizer_3d = visualizer.RealTime3DVisualizer()
visualizer_3d.start_visualization(port=5000)

class DphandPandaRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=500,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 tqdm_interval_sec=5.0,
                 num_points=1024,
                 render_mode="rgb_array"
                 ):
        super().__init__(output_dir)

        def env_fn():
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    DphandEnvWrapper(
                        PickAndPlaceEnv(config="pick_cube_env_cfg" ,render_mode=render_mode,),
                        num_points=num_points, 
                        use_point_cloud=True
                        ),
                    steps_per_render=1,
                    camera_id=1,
                    ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )
        
        self.eval_episodes = eval_episodes
        self.env = env_fn()

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

    def run(self, policy: BasePolicy):
        device = policy.device
        dtype = policy.dtype

        all_traj_rewards = []
        all_success_rates = []
        env = self.env

        
        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Dphand Pointcloud Env", leave=False, mininterval=self.tqdm_interval_sec):
            
            # start rollout
            obs = env.reset()
            policy.reset()

            done = False
            traj_reward = 0
            is_success = False
            while not done:
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                with torch.no_grad():
                    obs_dict_input = {}
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                    action_dict = policy.predict_action(obs_dict_input)

                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action'].squeeze(0)

                obs, reward, done, info = env.step(action)
                env.render()
                visualizer_3d.update_point_cloud(obs['point_cloud'][0])

                traj_reward += reward
                done = np.all(done)
                # dphand环境可能没有success字段，需要根据实际情况调整
                if 'success' in info:
                    is_success = is_success or max(info['success'])
                else:
                    # 如果没有success字段，可以根据reward或其他指标判断成功
                    # 这里暂时设置为False，需要根据具体任务调整
                    is_success = False

            all_success_rates.append(is_success)
            all_traj_rewards.append(traj_reward)


        max_rewards = collections.defaultdict(list)
        log_data = dict()

        log_data['mean_traj_rewards'] = np.mean(all_traj_rewards)
        log_data['mean_success_rates'] = np.mean(all_success_rates)

        log_data['test_mean_score'] = np.mean(all_success_rates)
        
        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()
        

        videos = env.env.get_video()
        if len(videos.shape) == 5:
            videos = videos[:, 0]  # select first frame
        
        videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
        log_data[f'sim_video_eval'] = videos_wandb

        _ = env.reset()
        videos = None

        return log_data
