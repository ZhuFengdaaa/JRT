# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from environment.environment import Environment
from model.model import UnrealModel
from train.experience import ExperienceFrame
from options import get_options
import util
import cv2

# get command line args
flags = get_options("evaluate")

class Evaluate(object):
  def __init__(self):
    self.cnt1, self.cnt2=0,0
    self.action_size = Environment.get_action_size(flags.env_type, flags.env_name)
    self.objective_size = Environment.get_objective_size(flags.env_type, flags.env_name)
    self.global_network = UnrealModel(self.action_size,
                                      self.objective_size,
                                      -1,
                                      flags.use_lstm,
                                      flags.use_pixel_change,
                                      flags.use_value_replay,
                                      flags.use_reward_prediction,
                                      0.0,
                                      0.0,
                                      "/cpu:0",
                                      for_display=True)
    self.environment = Environment.create_environment(flags.env_type, flags.env_name,
                                                      env_args={'episode_schedule': flags.split,
                                                                'log_action_trace': flags.log_action_trace,
                                                                'seed': flags.seed,
                                                                # 'max_states_per_scene': flags.episodes_per_scene,
                                                                'episodes_per_scene_test': flags.episodes_per_scene})
    self.episode_reward = 0
    self.cnt_success = 0
    self.f = open("log.txt" ,"w")
    self.success = False

  def update(self, sess, update_iter):
    self.process(sess, update_iter)

  def is_done(self):
    return self.environment.is_all_scheduled_episodes_done()

  def choose_action(self, pi_values):
    return np.random.choice(range(len(pi_values)), p=pi_values)

  def process(self, sess, update_iter):
    last_action = self.environment.last_action
    last_reward = np.clip(self.environment.last_reward, -1, 1)
    last_action_reward = ExperienceFrame.concat_action_and_reward(last_action, self.action_size,
                                                                  last_reward, self.environment.last_state)
    
    if not flags.use_pixel_change:
      base_conv_output, pi_values, v_value = self.global_network.run_base_policy_and_value(sess,
                                                                         self.environment.last_state,
                                                                         last_action_reward)
      print(base_conv_output)
      print(base_conv_output.shape)
      exit(233)
    else:
      base_conv_output, pi_values, v_value, pc_q = self.global_network.run_base_policy_value_pc_q(sess,
                                                                                self.environment.last_state,
                                                                                last_action_reward)
      base_conv_output = base_conv_output.squeeze()
      base_conv_output = np.sum(base_conv_output, axis=2)
      base_conv_output = cv2.resize(base_conv_output, (128,128), cv2.INTER_CUBIC)
      base_conv_output=base_conv_output-np.min(base_conv_output)
      base_conv_output=base_conv_output/np.max(base_conv_output)
      base_conv_output = np.uint8(base_conv_output*255)
      base_conv_output = cv2.applyColorMap(base_conv_output, cv2.COLORMAP_JET)
      cv2.imwrite("color%d_%d.png" % (self.cnt1, self.cnt2), self.environment.last_state["hdimage"])
      cv2.imwrite("heatmap%d_%d.png" % (self.cnt1, self.cnt2), base_conv_output)
      self.cnt2+=1
      #exit(233)
    action = self.choose_action(pi_values)
    state, reward, terminal, pixel_change, success = self.environment.process(action)
    if success:
        self.cnt_success += 1
        self.success = True
    self.episode_reward += reward
  
    if terminal:
      if self.success:
        self.f.write("1\n")
      else:
        self.f.write("0\n")
      print(float(self.cnt_success) / 150)
      self.environment.reset()
      self.cnt1+=1
      self.cnt2=0
      self.episode_reward = 0
      self.success = False


def main(args):
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)  # avoid using all gpu memory
  sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
  sess.run(tf.global_variables_initializer())
  
  evaluate = Evaluate()
  saver = tf.train.Saver()
  # checkpoint = tf.train.get_checkpoint_state(flags.checkpoint_dir)
  # print(checkpoint, checkpoint.model_checkpoint_path)
  # if checkpoint and checkpoint.model_checkpoint_path:
  saver.restore(sess, flags.checkpoint_dir)
  print("checkpoint loaded:", flags.checkpoint_dir)
  if flags.adda_path != "":
      util.load_checkpoints(flags.adda_path, "target", "net_-1", sess=sess)
  # else:
  #   print("Could not find old checkpoint")

  update_iter=0
  while not evaluate.is_done():
    update_iter+=1
    evaluate.update(sess, update_iter)
  if flags.score_file is not None:
    with open(flags.score_file, "a") as score_file:
      score_file.write("%.2f\n" % float(evaluate.cnt_success/150*100))

    
if __name__ == '__main__':
  tf.app.run()
