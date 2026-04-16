# A Walk in the Park

Code to replicate [A Walk in the Park: Learning to Walk in 20 Minutes With Model-Free Reinforcement Learning](https://arxiv.org/abs/2208.07860), which contains code for training a simulated or real A1 quadrupedal robot to walk. Project page: https://sites.google.com/berkeley.edu/walk-in-the-park

## Installation

Create a clean environment and install the pinned dependencies:
```bash
conda env create -f environment-jax-cuda12.yml
conda activate park
```

If you want to reuse an existing `park` environment instead of recreating it,
first remove the conflicting packages and then reinstall the pinned stack:
```bash
python -m pip uninstall -y gym numpy jax jaxlib jax-cuda12-pjrt jax-cuda12-plugin tensorflow tensorflow-cpu tensorflow-probability
python -m pip install -r requirements.txt
```

To install the robot [SDK](https://github.com/unitreerobotics/unitree_legged_sdk), first install the dependencies in the README.md

To build, run: 
```bash
cd real/third_party/unitree_legged_sdk
mkdir build
cd build
cmake ..
make
``` 

Finally, copy the built `robot_interface.XXX.so` file to this directory.

## Training

Example command to run simulated training:

```bash
./scripts/run_train_cuda.sh \
    --env_name=A1Run-v0 \
    --utd_ratio=20 \
    --start_training=1000 \
    --max_steps=100000 \
    --config=configs/droq_config.py
```

To run training on the real robot, add `--real_robot=True`

Live preview for external viewing:
```bash
./scripts/run_train_cuda.sh \
    --env_name=A1Run-v0 \
    --utd_ratio=20 \
    --start_training=1000 \
    --max_steps=100000 \
    --config=configs/droq_config.py \
    --preview_stream=True \
    --preview_host=0.0.0.0 \
    --preview_port=8080
```

Then open `http://<your-machine-ip>:8080/` in a browser on another machine.

Notes:
- Do not export `JAX_PLATFORMS=gpu` with JAX 0.6.x. It expands to both `cuda`
  and `rocm` and can fail before training starts.
- The CUDA launcher prints the selected JAX backend at startup and fails fast if
  the process does not land on `cuda`.
