from gymnasium.envs.registration import register  # type: ignore[import-untyped]
import pathlib

CUR_PATH = pathlib.Path(__file__).parent

register(
    id="DphandPickCube-v0",
    entry_point="dphand.dphand_env:DphandPickCubeEnv",
    kwargs={"config_path": CUR_PATH / "../assets/configs/dphand_pick_cube_env_cfg.yaml",
            "image_obs": False},
)
