# UNREAL / A3C / A3C-LSTM agent for MINOS

This is an implementation of the UNREAL agent described in [Reinforcement Learning with Unsupervised Auxiliary Tasks](https://arxiv.org/abs/1611.05397) by Jaderberg et al. 2016 for use with the [MINOS](https://github.com/minosworld/minos) multimodal indoor simulator.  This implementation is directly adaptated from https://github.com/miyosuda/unreal with a thin gym wrapper around the [MINOS](https://github.com/minosworld/minos) framework.  The original implementation README.md file is [here](README.unreal.md).  This code also implements the A3C and A3C-LSTM baseline agents (as ablations of the full UNREAL agent -- see `options.py` for relevant flags).

## Installation

Follow the [MINOS](https://github.com/minosworld/minos) installation instructions.  Confirm that the minos package is available by running `python3 -c 'import minos; print(minos)'`.

## Running experiments

Start experiments using invocations such as the following:

`python3 main.py --env_type indoor --env_name pointgoal_suncg_se`

`python3 main.py --env_type indoor --env_name objectgoal_suncg_mf`

`python3 main.py --env_type indoor --env_name roomgoal_mp3d_s`

Refer to `options.py` for available arguments that control the hyperparameters and agent architecture.
