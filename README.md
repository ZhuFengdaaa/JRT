# UNREAL / A3C / A3C-LSTM agent for MINOS

This is an implementation of the UNREAL agent described in [Reinforcement Learning with Unsupervised Auxiliary Tasks](https://arxiv.org/abs/1611.05397) by Jaderberg et al. 2016 for use with the [MINOS](https://github.com/minosworld/minos) multimodal indoor simulator.  This implementation is directly adaptated from https://github.com/miyosuda/unreal with a thin gym wrapper around the [MINOS](https://github.com/minosworld/minos) framework.  The original implementation README.md file is [here](README.unreal.md).  This code also implements the A3C and A3C-LSTM baseline agents (as ablations of the full UNREAL agent -- see `options.py` for relevant flags).

## Installation

Follow [OUR MINOS](https://github.com/zhufengdaaa/minos) installation instructions.  Confirm that the minos package is available by running `python3 -c 'import minos; print(minos)'`.

## Training baselines

Checkout git branch **minos/train** and **unreal/master**

Start experiments using invocations such as the following:

`python3 main.py --env_type indoor --env_name roomgoal_suncg_sf`

Refer to `options.py` for available arguments that control the hyperparameters and agent architecture.

## Adversarial Transfer

Refer to [adda](https://github.com/ZhuFengdaaa/adda)

## Policy Mimic

Checkout git branch **unreal/mimic** for policy mimic. 

Set paths in code and run:

`python3 main.py --env_type indoor --env_name roomgoal_mp3d_s`

## Testing

Checkout git branch **unreal/test** for testing. 

`python3 main.py --env_type indoor --env_name roomgoal_mp3d_sf`

