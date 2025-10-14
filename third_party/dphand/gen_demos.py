# bash scripts/metaworld/gen_demonstration_expert.sh reach 5
import argparse
import os
import zarr
import numpy as np
from dphand_env.envs.pick_and_place_env import PickAndPlaceEnv
from dphand_env.mujoco.wrappers import TeleopIntervention
from diffusion_policy_3d.env import DphandPointCloudEnvWrapper
from termcolor import cprint
import copy
import cv2


def main(args):
	save_dir = os.path.join(args.data_dir, 'panda_' + args.task_name + '.zarr')
	if os.path.exists(save_dir):
		cprint('Data already exists at {}'.format(save_dir), 'white')
		cprint("If you want to overwrite, delete the existing directory first.", "white")
		cprint("Do you want to overwrite? (y/n)", "red")
		# user_input = input()
		user_input = 'y'
		if user_input == 'y':
			cprint('Overwriting {}'.format(save_dir), 'red')
			os.system('rm -rf {}'.format(save_dir))
		else:
			cprint('Exiting', 'red')
			return
	os.makedirs(save_dir, exist_ok=True)
	# 创建环境
	env = PickAndPlaceEnv(config="pick_cube_env_cfg", render_mode="human")
	env = TeleopIntervention(env, ip="192.168.3.80", test=False, use_relative_pose=True)
	env = DphandPointCloudEnvWrapper(
		env=env,
		num_points=1024,
		use_point_cloud=False  # 生成点云
	)
	
	total_count = 0
	demo_count = 0
	success_count = 0
	front_image_arrays = []
	wrist_image_arrays = []

	depth_arrays = []
	state_arrays = []
	full_state_arrays = []
	action_arrays = []
	init_state_arrays = []
	episode_ends_arrays = []
    
	start = False
	# 初始化环境
	obs = env.reset()
	# loop over episodes
	while env.keyboard != "esc":
		# 这里替换成自己的策略
		action = np.random.uniform(
			low=env.action_space.low,
			high=env.action_space.high,
			size=env.action_space.shape
		)
		obs, reward, done, info = env.step(action) # action会由遥操作接口进行修改
		env.render()
		img = np.concatenate([obs['image']['front'], obs['image']['wrist']], axis=0)
		cv2.imshow('image', cv2.resize(img, (0,0), fx=3, fy=3))
		cv2.waitKey(1)

		if start == False and env.keyboard == "enter":
			start = True
			obs = env.reset()
			done = False
			ep_reward = 0.
			ep_success = False
			front_image_arrays_sub = []
			wrist_image_arrays_sub = []
			depth_arrays_sub = []
			state_arrays_sub = []
			full_state_arrays_sub = []
			action_arrays_sub = []
			total_count_sub = 0
			cprint("Start recording demos.", "green")
			print("Press 'r' to reset or Press 'enter' to save.\n")

		elif start == True:
			total_count_sub += 1
			front_image_arrays_sub.append(obs['image']['front'])
			wrist_image_arrays_sub.append(obs['image']['wrist'])
			depth_arrays_sub.append(obs['depth'])
			state_arrays_sub.append(obs['agent_pos'])
			full_state_arrays_sub.append(obs['full_state'])
			action_arrays_sub.append(info['action'])
			
			# 使用从env.step()返回的reward
			ep_reward += reward

			if env.keyboard == "enter" or info['success']:
				ep_success = ep_success or info['success']
				demo_count += 1
				success_count += info['success']
				total_count += total_count_sub
				init_state_arrays.extend(copy.deepcopy(info['init_state']))
				episode_ends_arrays.append(copy.deepcopy(total_count)) # the index of the last step of the episode    
				front_image_arrays.extend(copy.deepcopy(front_image_arrays_sub))
				wrist_image_arrays.extend(copy.deepcopy(wrist_image_arrays_sub))
				depth_arrays.extend(copy.deepcopy(depth_arrays_sub))
				state_arrays.extend(copy.deepcopy(state_arrays_sub))
				action_arrays.extend(copy.deepcopy(action_arrays_sub))
				full_state_arrays.extend(copy.deepcopy(full_state_arrays_sub))
				cprint(f"Got total {demo_count} demos ({success_count} successes).", "green")
				print("Press 'enter' to continue or Press 'esc' to exit.\n")
				obs = env.reset()
				start = False

			if env.keyboard == "+" or env.keyboard == "r":
				print("Re-start recording demos.\n")
				obs = env.reset()
				done = False
				ep_reward = 0.
				ep_success = False
				front_image_arrays_sub = []
				wrist_image_arrays_sub = []
				depth_arrays_sub = []
				state_arrays_sub = []
				full_state_arrays_sub = []
				action_arrays_sub = []
				total_count_sub = 0
				start = False
				continue

	# 修复：数据保存应该在while循环结束后
	# save data
	###############################
	# create zarr file
	zarr_root = zarr.group(save_dir)
	zarr_data = zarr_root.create_group('data')
	zarr_image = zarr_data.create_group('image')
	zarr_meta = zarr_root.create_group('meta')
	# save img, state, action arrays into data, and episode ends arrays into meta
	front_image_arrays = np.stack(front_image_arrays, axis=0)
	wrist_image_arrays = np.stack(wrist_image_arrays, axis=0)
	if front_image_arrays.shape[1] == 3: # make channel last
		front_image_arrays = np.transpose(front_image_arrays, (0,2,3,1))

	if wrist_image_arrays.shape[1] == 3: # make channel last
		wrist_image_arrays = np.transpose(wrist_image_arrays, (0,2,3,1))
	state_arrays = np.stack(state_arrays, axis=0)
	full_state_arrays = np.stack(full_state_arrays, axis=0)
	depth_arrays = np.stack(depth_arrays, axis=0)
	action_arrays = np.stack(action_arrays, axis=0)
	init_state_arrays = np.stack(init_state_arrays, axis=0)
	episode_ends_arrays = np.array(episode_ends_arrays)

	compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
	front_image_chunk_size = (100, front_image_arrays.shape[1], front_image_arrays.shape[2], front_image_arrays.shape[3])
	wrist_image_chunk_size = (100, wrist_image_arrays.shape[1], wrist_image_arrays.shape[2], wrist_image_arrays.shape[3])
	state_chunk_size = (100, state_arrays.shape[1])
	full_state_chunk_size = (100, full_state_arrays.shape[1])
	depth_chunk_size = (100, depth_arrays.shape[1], depth_arrays.shape[2])
	action_chunk_size = (100, action_arrays.shape[1])
	zarr_image.create_dataset('front', data=front_image_arrays, chunks=front_image_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
	zarr_image.create_dataset('wrist', data=wrist_image_arrays, chunks=wrist_image_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
	zarr_data.create_dataset('depth', data=depth_arrays, chunks=depth_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
	zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
	zarr_data.create_dataset('full_state', data=full_state_arrays, chunks=full_state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
	zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
	zarr_meta.create_dataset('init_state', data=init_state_arrays, dtype='float32', overwrite=True, compressor=compressor)
	zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)

	# print shape
	cprint(f'front_image shape: {front_image_arrays.shape}, range: [{np.min(front_image_arrays)}, {np.max(front_image_arrays)}]', 'green')
	cprint(f'wrist_image shape: {wrist_image_arrays.shape}, range: [{np.min(wrist_image_arrays)}, {np.max(wrist_image_arrays)}]', 'green')
	cprint(f'depth shape: {depth_arrays.shape}, range: [{np.min(depth_arrays)}, {np.max(depth_arrays)}]', 'green')
	cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
	cprint(f'full_state shape: {full_state_arrays.shape}, range: [{np.min(full_state_arrays)}, {np.max(full_state_arrays)}]', 'green')
	cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
	cprint(f'Saved zarr file to {save_dir}', 'light_blue')


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--task_name', type=str, default="pick_and_place_1014", choices=["pick_and_place"])
	parser.add_argument('--data_dir', type=str, default="/home/yhx/workspace/3D-Diffusion-Policy/data/1014" )
	args = parser.parse_args()
	main(args)