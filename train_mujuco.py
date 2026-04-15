#! /usr/bin/env python
"""Compatibility entrypoint for typoed script name.

Use `train_mujoco.py` for primary training.
"""

from absl import app

from train_mujoco import main


if __name__ == '__main__':
    app.run(main)
