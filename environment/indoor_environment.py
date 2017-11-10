# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from datetime import datetime
import numpy as np
import os

from environment import environment
from sim.simulator.RoomSimulator import RoomSimulator
from helpers import sim_config

class IndoorEnvironment(environment.Environment):

  ACTION_LIST = [
    [1,0,0],
    [0,1,0],
    [0,0,1]
  ]

  @staticmethod
  def get_action_size(env_name):
    return len(IndoorEnvironment.ACTION_LIST)

  def __init__(self, env_name, thread_index):
    environment.Environment.__init__(self)
    
    self.last_state = None
    self.last_action = 0
    self.last_reward = 0

    simargs = sim_config.get(env_name)
    simargs['id'] = 'sim%02d' % thread_index
    simargs['logdir'] = os.path.join(IndoorEnvironment.get_log_dir(), simargs['id'])

    self._sim = RoomSimulator(simargs)
    self._sim_obs_space = self._sim.get_observation_space()
    self.reset()

  def reset(self):
    result = self._sim.reset()
    
    self._episode_info = result.get('episode_info')
    self._last_full_state = result.get('observation')
    obs = self._last_full_state['images']
    state = self._preprocess_frame(obs)
    self.last_state = state
    self.last_action = 0
    self.last_reward = 0

  def stop(self):
    if self._sim is not None:
        self._sim.close_game()

  def _preprocess_frame(self, image):
    if len(image.shape) == 3 and image.shape[0] == 1:  # assume gray
        image = np.dstack([image[0], image[0], image[0]])
    else:  # assume rgba
        image = image[0][:, :, :-1]
    image = image.astype(np.float32)
    image = image / 255.0
    return image

  def process(self, action):
    real_action = IndoorEnvironment.ACTION_LIST[action]

    full_state = self._sim.step(real_action)
    self._last_full_state = full_state  # Last observed state
    obs = full_state['images']
    reward = full_state['rewards']
    terminal = full_state['terminals']

    if not terminal:
      state = self._preprocess_frame(obs)
    else:
      state = self.last_state

    pixel_change = self._calc_pixel_change(state, self.last_state)
    self.last_state = state
    self.last_action = action
    self.last_reward = reward
    return state, reward, terminal, pixel_change
