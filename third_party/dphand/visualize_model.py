import mujoco
import mujoco.viewer
from pynput import keyboard

model = mujoco.MjModel.from_xml_path("/home/robot/Workspace/3D-Diffusion-Policy/third_party/dphand/dphand_env/assets/panda_dphand_with_tip_camera.xml")  # 改成你的xml路径
data = mujoco.MjData(model)

def _start_escape_listener(viewer):
    def on_press(key):
        if key == keyboard.Key.esc:
            viewer.close()
            return False
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener

with mujoco.viewer.launch_passive(model, data) as viewer:
    esc_listener = _start_escape_listener(viewer)
    try:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
    finally:
        if esc_listener is not None:
            esc_listener.stop()
