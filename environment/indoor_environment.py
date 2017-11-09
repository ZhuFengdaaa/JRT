# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from environment import environment
from sim.simulator.RoomSimulator import RoomSimulator

class IndoorEnvironment(environment.Environment):

  ACTION_LIST = [
    [1,0,0],
    [0,1,0],
    [0,0,1]
  ]

  @staticmethod
  def get_action_size(env_name):
    return len(IndoorEnvironment.ACTION_LIST)
  
  def __init__(self, env_name):
    environment.Environment.__init__(self)
    
    self.last_state = None
    self.last_action = 0
    self.last_reward = 0

    self._sim = RoomSimulator(sim_args)
    self._sim_obs_space = self._sim.get_observation_space()

  def reset(self):
    result = self._sim.init_game()
    if result is None:
        result = self._sim.new_episode()
    # Force one observation
    #self._step([0]*self._sim.num_buttons)
    
    #self.last_state = self._preprocess_frame(obs)
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
    self._last_full_state = state  # Last observed state
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
