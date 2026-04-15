# A Walk in the Park

Code to replicate [A Walk in the Park: Learning to Walk in 20 Minutes With Model-Free Reinforcement Learning](https://arxiv.org/abs/2208.07860), adapted here for a simulated GO1 quadrupedal robot. Project page: https://sites.google.com/berkeley.edu/walk-in-the-park

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
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
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false python train_online.py --env_name=Go1Run-v0 \
                --utd_ratio=20 \
                --start_training=1000 \
                --max_steps=100000 \
                --config=configs/droq_config.py
```

The current branch is GO1 simulation-only.
