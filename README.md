# 添加相机
在 `third_party/dphand/dphand_env/assets/panda_dphand.xml` 中添加相机

# 渲染深度图
以 `diffusion_policy_3d/env/dphand/test_env.py` 为例

将 62 行修改为
```python
# depth = obs['depth'].copy()
_, depth = env.unwrapped._viewer.render_segment_depth(env.unwrapped.cam_ids["front"])
```
可以获取对应名称相机的深度图

# 关于环境的说明
- `MujocoEnv`: Gym中定义的Mujoco环境基类，结合了一些mujoco的基础api,实现了一些基础功能
- `MujocoGymEnv`: 继承自 `MujocoEnv`，主要添加了渲染功能，参考 `mujoco_render.py`
- `BaseEnv`: 继承自 `MujocoGymEnv`，主要有两点变化：1.支持从config文件中读取环境参数；2.支持运行多个物理步 # 可以考虑把这个环境和`DphandPandaEnv`合并，因为`dphand_env`已经弃用（这个环境是为之前没有franka panda的时候写的）
- `DphandPandaEnv`: 继承自 `BaseEnv`，主要添加了灵巧手和franka panda相关的属性，包括franka panda的阻抗控制器
- `PickAndPlaceEnv`: 继承自 `DphandPandaEnv`，面向任务级别的环境，例如pick and place

tactile sensor : depth<=0.5mm

generate zarr:/home/robot/Workspace/3D-Diffusion-Policy/third_party/dphand/gen_pointcloud.py

<!-- use uv to implement the xense package -->

then find methods to construct an encoder or two encoders for point cloud and tactile img


RESNET
