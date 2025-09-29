# Install for dphand env
```
conda create -n dp3-yhx python=3.8

pip install uv

uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

cd 3D-Diffusion-Policy && pip install -e . && cd ..

uv pip install zarr==2.12.0 wandb ipdb gpustat omegaconf hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 diffusers==0.11.1 numba==0.56.4 moviepy imageio av matplotlib termcolor huggingface_hub==0.25.2 pynput open3d opencv-python
```

```
# install dphand env
cd third_party/dphand && pip install -e . && cd ..
# install dphand-teleop
git clone https://github.com/zcex12138/dphand-teleop.git
cd dphand-teleop && pip install -e . & cd ../..
# pytorch3d
cd third_party/pytorch3d_simplified && pip install -e . && cd ..
# visualizer for
cd visualizer && pip install -e .
```