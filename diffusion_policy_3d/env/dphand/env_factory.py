"""
环境工厂模块
提供make_env()函数，通过配置文件创建环境和wrapper
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import gymnasium as gym

from dphand_env.envs.pick_and_place_env import PickAndPlaceEnv
from dphand_env.mujoco.wrappers import TeleopIntervention, Fix6DPoseWrapper
from diffusion_policy_3d.env.dphand.dphand_wrapper import (
    DphandPointCloudEnvWrapper,
    DphandImageEnvWrapper,
)
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper
from diffusion_policy_3d.gym_util.video_wrapper import VideoWrapper


# Wrapper注册表
WRAPPER_REGISTRY = {
    'TeleopIntervention': TeleopIntervention,
    'Fix6DPoseWrapper': Fix6DPoseWrapper,
    'DphandPointCloudEnvWrapper': DphandPointCloudEnvWrapper,
    'DphandImageEnvWrapper': DphandImageEnvWrapper,
    'MultiStepWrapper': MultiStepWrapper,
    'SimpleVideoRecordingWrapper': SimpleVideoRecordingWrapper,
    'VideoWrapper': VideoWrapper,
}

# 环境注册表
ENV_REGISTRY = {
    'PickAndPlaceEnv': PickAndPlaceEnv,
}


def make_env(cfg_path: str, 
              wrapper_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
              **override_kwargs) -> gym.Env:
    """
    根据配置文件创建环境
    
    Args:
        cfg_path: 配置文件路径（YAML格式）
        wrapper_overrides: 可选的wrapper参数覆盖字典，格式为 {wrapper_type: {param_name: value}}
            例如: {'DphandPointCloudEnvWrapper': {'use_point_cloud': True, 'num_points': 2048}}
            也可以通过 'enabled': False 来禁用某个wrapper
        **override_kwargs: 可选的覆盖参数，会覆盖配置文件中的对应值（仅限环境参数）
        
    Returns:
        配置好的环境实例
        
    配置文件格式示例:
        env:
          type: PickAndPlaceEnv
          task: pick_cube
          use_tactile_obs: false
          render_mode: human
          
        wrappers:
          - type: TeleopIntervention
            enabled: true
            params:
              ip: "192.168.3.80"
              test: true
              use_relative_pose: true
              
          - type: DphandPointCloudEnvWrapper
            enabled: true
            params:
              use_point_cloud: false
              num_points: 1024
              
    使用示例:
        # 覆盖wrapper参数
        env = make_env(
            'config.yaml',
            wrapper_overrides={
                'DphandPointCloudEnvWrapper': {
                    'use_point_cloud': True,
                    'num_points': 2048
                }
            }
        )
        
        # 禁用某个wrapper
        env = make_env(
            'config.yaml',
            wrapper_overrides={
                'TeleopIntervention': {
                    'enabled': False
                }
            }
        )
    """
    # 加载配置文件
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")
    
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # 获取环境配置
    env_cfg = cfg.get('env', {})
    env_type = env_cfg.get('type', 'PickAndPlaceEnv')
    
    if env_type not in ENV_REGISTRY:
        raise ValueError(f"未知的环境类型: {env_type}. 可用类型: {list(ENV_REGISTRY.keys())}")
    
    EnvClass = ENV_REGISTRY[env_type]
    
    # 准备环境初始化参数
    # 支持两种格式：env.params 或直接在 env 下
    if 'params' in env_cfg:
        env_params = env_cfg['params'].copy()
        # 从 env_cfg 中读取 use_tactile_obs（优先 env_cfg，其次 params）
        use_tactile_obs = env_cfg.get('use_tactile_obs', env_params.get('use_tactile_obs', False))
    else:
        # 直接从env_cfg中提取参数（排除type）
        env_params = {k: v for k, v in env_cfg.items() if k != 'type'}
        use_tactile_obs = env_params.get('use_tactile_obs', False)
    
    # 允许通过override_kwargs覆盖环境参数（包括 use_tactile_obs）
    if 'use_tactile_obs' in override_kwargs:
        use_tactile_obs = override_kwargs['use_tactile_obs']
    
    env_params.update({k: v for k, v in override_kwargs.items() if k in ['task', 'render_mode', 'seed']})
    
    # 从 env_params 中获取 task（如果被 override_kwargs 覆盖了，使用覆盖后的值）
    task = env_params.get('task')
    if not task:
        # 兼容旧的 config 格式：如果存在 config，尝试从中提取 task
        old_config = env_params.get('config')
        if old_config:
            # 移除 _tactile 和 _env_cfg 后缀，提取 task 名称
            task = old_config.replace('_tactile_env_cfg', '').replace('_env_cfg', '')
        else:
            raise ValueError("配置文件中必须指定 'task' 参数，或提供 'config' 参数（兼容模式）")
    
    # 根据 use_tactile_obs 构建最终的 config 名称
    if use_tactile_obs:
        config_name = f"{task}_tactile_env_cfg"
    else:
        config_name = f"{task}_env_cfg"
    
    env_params['config'] = config_name
    # 移除 task 和 use_tactile_obs，它们不是 PickAndPlaceEnv 的参数
    env_params.pop('task', None)
    env_params.pop('use_tactile_obs', None)
    
    # 创建基础环境
    env = EnvClass(**env_params)
    
    # 应用wrapper（按顺序）
    wrappers_cfg = cfg.get('wrappers', [])
    for wrapper_cfg in wrappers_cfg:
        wrapper_type = wrapper_cfg.get('type')
        
        # 检查是否通过wrapper_overrides禁用了该wrapper
        if wrapper_overrides and wrapper_type in wrapper_overrides:
            override_cfg = wrapper_overrides[wrapper_type]
            if override_cfg.get('enabled') is False:
                continue
        
        # 检查配置文件中的enabled设置
        if not wrapper_cfg.get('enabled', True):
            continue
        
        if wrapper_type not in WRAPPER_REGISTRY:
            raise ValueError(f"未知的wrapper类型: {wrapper_type}. 可用类型: {list(WRAPPER_REGISTRY.keys())}")
        
        WrapperClass = WRAPPER_REGISTRY[wrapper_type]
        wrapper_params = wrapper_cfg.get('params', {}).copy()
        
        # 自动将 use_tactile_obs 传递给支持触觉观测的 wrapper
        if wrapper_type in ['DphandPointCloudEnvWrapper', 'DphandImageEnvWrapper']:
            wrapper_params['use_tactile_obs'] = use_tactile_obs
        
        # 应用wrapper参数覆盖（如果提供）
        if wrapper_overrides and wrapper_type in wrapper_overrides:
            override_cfg = wrapper_overrides[wrapper_type].copy()
            # 移除配置项（如enabled），只保留wrapper的实际参数
            override_cfg.pop('enabled', None)
            wrapper_params.update(override_cfg)
        
        # 应用wrapper
        env = WrapperClass(env, **wrapper_params)
    
    return env


def make_env_from_dict(cfg_dict: Dict[str, Any], wrapper_overrides: Optional[Dict[str, Dict[str, Any]]] = None, **override_kwargs) -> gym.Env:
    """
    从字典配置创建环境（用于程序内配置）
    
    Args:
        cfg_dict: 配置字典
        wrapper_overrides: 可选的wrapper参数覆盖字典，格式为 {wrapper_type: {param_name: value}}
        **override_kwargs: 可选的覆盖参数（仅限环境参数）
        
    Returns:
        配置好的环境实例
    """
    # 获取环境配置
    env_cfg = cfg_dict.get('env', {})
    env_type = env_cfg.get('type', 'PickAndPlaceEnv')
    
    if env_type not in ENV_REGISTRY:
        raise ValueError(f"未知的环境类型: {env_type}. 可用类型: {list(ENV_REGISTRY.keys())}")
    
    EnvClass = ENV_REGISTRY[env_type]
    
    # 准备环境初始化参数
    # 支持两种格式：env.params 或直接在 env 下
    if 'params' in env_cfg:
        env_params = env_cfg['params'].copy()
    else:
        # 直接从env_cfg中提取参数（排除type）
        env_params = {k: v for k, v in env_cfg.items() if k != 'type'}
    
    env_params.update({k: v for k, v in override_kwargs.items() if k in ['config', 'render_mode', 'seed']})
    
    # 创建基础环境
    env = EnvClass(**env_params)
    
    # 应用wrapper
    wrappers_cfg = cfg_dict.get('wrappers', [])
    for wrapper_cfg in wrappers_cfg:
        wrapper_type = wrapper_cfg.get('type')
        
        # 检查是否通过wrapper_overrides禁用了该wrapper
        if wrapper_overrides and wrapper_type in wrapper_overrides:
            override_cfg = wrapper_overrides[wrapper_type]
            if override_cfg.get('enabled') is False:
                continue
        
        # 检查配置文件中的enabled设置
        if not wrapper_cfg.get('enabled', True):
            continue
        
        if wrapper_type not in WRAPPER_REGISTRY:
            raise ValueError(f"未知的wrapper类型: {wrapper_type}. 可用类型: {list(WRAPPER_REGISTRY.keys())}")
        
        WrapperClass = WRAPPER_REGISTRY[wrapper_type]
        wrapper_params = wrapper_cfg.get('params', {}).copy()
        
        # 应用wrapper参数覆盖（如果提供）
        if wrapper_overrides and wrapper_type in wrapper_overrides:
            override_cfg = wrapper_overrides[wrapper_type].copy()
            # 移除配置项（如enabled），只保留wrapper的实际参数
            override_cfg.pop('enabled', None)
            wrapper_params.update(override_cfg)
        
        env = WrapperClass(env, **wrapper_params)
    
    return env

