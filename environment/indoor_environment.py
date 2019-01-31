# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os

from PIL import Image
from environment import environment
from minos.lib.RoomSimulator import RoomSimulator
from minos.config import sim_config
import time
import json
from gym.envs.classic_control import rendering
import cv2
import time


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class IndoorEnvironment(environment.Environment):

  ACTION_LIST = [
    [1,0,0],
    [0,1,0],
    [0,0,1]
  ]

  @staticmethod
  def get_action_size(env_name):
    return len(IndoorEnvironment.ACTION_LIST)

  @staticmethod
  def get_objective_size(env_name):
    simargs = sim_config.get(env_name)
    return simargs.get('objective_size', 0)

  def __init__(self, env_name, env_args, thread_index):
    environment.Environment.__init__(self)
    self.i_episode = 0
    
    self.last_state = None
    self.last_action = 0
    self.last_reward = 0

    simargs = sim_config.get(env_name)
    simargs['id'] = 'sim%02d' % thread_index
    simargs['logdir'] = os.path.join(IndoorEnvironment.get_log_dir(), simargs['id'])
    self.viewer = rendering.SimpleImageViewer()

    # Merge in extra env args
    if env_args is not None:
      simargs.update(env_args)

    print(simargs)
    self._sim = RoomSimulator(simargs)
    self._sim_obs_space = self._sim.get_observation_space(simargs['outputs'])
    self.reset()

  def render(self, img):
    img = img[:, :, :-1]
    img = img.reshape((img.shape[1], img.shape[0], img.shape[2]))
    img = cv2.resize(img, (512,512), cv2.INTER_CUBIC);
    self.viewer.imshow(img)
    time.sleep(.1)

  def reset(self):
    result = self._sim.reset()
    
    self._episode_info = result.get('episode_info')
    self._last_full_state = result.get('observation')
    hd_image = np.array(self._last_full_state['observation']['sensors']['color']['data'], copy=True)
    resized_image = cv2.resize(self._last_full_state['observation']['sensors']['color']['data'], (84,84))
    self._last_full_state['observation']['sensors']['color']['data'] = resized_image
    self._last_full_state['observation']['sensors']['color']['hddata'] = hd_image
    obs = self._last_full_state['observation']['sensors']['color']['data']
    # self.render(obs)
    objective = self._last_full_state.get('measurements')
    state = { 'image': self._preprocess_frame(obs), 'objective': objective, "hdimage": hd_image }
    self.last_state = state
    self.last_action = 0
    self.last_reward = 0
    # self.i_episode = self.i_episode + 1
    # print("Saving episode {}".format(self.i_episode))
    # self.directory = "./{}".format(self.i_episode)
    # os.mkdir(self.directory)
    # with open(os.path.join(self.directory, "episode_info.txt"), "w") as outfile:
    #     json.dump(self._episode_info, outfile, indent=4, cls=NumpyEncoder)
    # self.i = 0

  def stop(self):
    if self._sim is not None:
        self._sim.close_game()

  def _preprocess_frame(self, image):
    if len(image.shape) == 2:  # assume gray
        image = np.dstack([image, image, image])
    else:  # assume rgba
        image = image[:, :, :-1]
    image = image.astype(np.float32)
    image = image / 255.0
    return image

  def process(self, action):
    real_action = IndoorEnvironment.ACTION_LIST[action]

    full_state = self._sim.step(real_action)
    hd_image = np.array(full_state['observation']['sensors']['color']['data'], copy=True)
    resized_image = cv2.resize(full_state['observation']['sensors']['color']['data'], (84,84))
    full_state['observation']['sensors']['color']['data'] = resized_image
    full_state['observation']['sensors']['color']['hddata'] = hd_image
    self._last_full_state = full_state  # Last observed state
    obs = full_state['observation']['sensors']['color']['data']
    # self.render(obs)
    # depth = full_state['observation']['sensors']['depth']['data']
    # Image.fromarray(obs.astype('uint8')).save(os.path.join(self.directory, 'color{}.png'.format(self.i)))
    # Image.fromarray(depth, 'L').save(os.path.join(self.directory, 'depth{}.png'.format(self.i)))
    # self.i+=1
    reward = full_state['rewards']
    terminal = full_state['terminals']
    success = full_state['success']
    objective = full_state.get('measurements')

    if not terminal:
      state = { 'image': self._preprocess_frame(obs), 'objective': objective, "hdimage": hd_image }
    else:
      state = self.last_state

    pixel_change = self._calc_pixel_change(state['image'], self.last_state['image'])
    self.last_state = state
    self.last_action = action
    self.last_reward = reward
    return state, reward, terminal, pixel_change, success

  def is_all_scheduled_episodes_done(self):
    return self._sim.is_all_scheduled_episodes_done()
