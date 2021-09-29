import gym
import numpy as np
from statistics import mean

from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import pickle
import os

from RL_train import iterator, init_hyperparams

# Should make training go faster for large models
cudnn.benchmark = True  
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)




# Definition of hyper-parameters to tune with lower and upper value limits
# Usage -> "param" : (lower, upper)
parameters = {'inscale' : (0.0,0.5), 'hue' : (0.0,0.5), 'contrast' : (0.0,1.0), 'sat' : (0.0,1.0), 'bright' : (0.0,1.0), \
              'cutlength' : (0.0,24.0), 'cutholes' : (0.0,4.0), 'learning_rate' : (1e-5, 0.05)}
# Definition of metrics presented to the agent with:
# - lower value limit
# - upper value limits
# - for validation metrics, thresholdout noise scale
# - for validation metrics, corresponding train metrics
# Usage -> "metric" : (lower, upper [, noise scale, train metric])
metrics = {'val_acc' : (0.0,1.0, 1.0, 'acc'), 'val_loss' : (0.0,100.0, 10.0, 'loss'), 'acc' : (0.0,1.0), 'loss' : (0.0,100.0)}
val_metrics = {'val_acc' : 0.0, 'val_loss' : 0.0}
not_val_metrics = {'acc' : 0.0, 'loss' : 0.0}
# Metrics used for computing the reward of the agent with their respective coefficient factor
reward_coefs = {'val_acc' : 1000.0, 'acc' : 1.0}

save_filename = 'normal_64'
corruption = 0.0

# RL - Hyper-parameters default values for the warming train epochs
init_hyperparams = {}
# Tuned hyper-parameters
init_hyperparams['inscale'] = 0.0
init_hyperparams['hue'] = 0.0
init_hyperparams['contrast'] = 0.0
init_hyperparams['sat'] = 0.0
init_hyperparams['bright'] = 0.0
init_hyperparams['cutlength'] = 0.0
init_hyperparams['cutholes'] = 0.0
init_hyperparams['learning_rate'] = 0.05
# Fixed hyper-parameters
init_hyperparams['momentum'] = 0.9
init_hyperparams['percent_valid'] = 0.2
init_hyperparams['batch_size'] = 64
init_hyperparams['warmup_epochs'] = 5
init_hyperparams['patience'] = 60
init_hyperparams['max_epoch'] = 100


class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, parameters, metrics, val_metrics, not_val_metrics, reward_coefs, save_filename, corruption, do_thresholdout=True, test_mode=False):
    super(CustomEnv, self).__init__()

    # Whether it is an environment used to train or test the agent 
    self.test_mode = test_mode
    # Whether to do thresholdout or not
    self.do_thresholdout = do_thresholdout
    # Filename used in experiments
    self.save_filename = save_filename
    # Corruption level
    self.corruption = corruption
    # RL parameters
    self.parameters = parameters
    self.metrics = metrics
    self.val_metrics = val_metrics
    self.not_val_metrics = not_val_metrics
    self.reward_coefs = reward_coefs

    # Number of train steps between changing the hyper-parameters values
    self.train_steps = 100
    # Number of validation steps used to compute validation metrics
    self.valid_steps = 1
    # Progressively allows the agent to play more time
    self.epoch_schedule = [10, 15, 25, 35]
    self.game_per_epoch = 10
    self.n_games = 0
    # Number of steps between updates of validation metrics
    # Avoids reaching the upper limit of number of thresholdout validation metrics presented to the agent
    self.steps_between_thresholdouts = 10
    self.running_avg_lgt = 10
    # Number of step statistics visible from the agent
    self.window_size = 500
    self.reset_stat()
    self.load_of_datasets_and_interaction_spaces()

    # Definition of action and observation space for the agent
    self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(len(parameters),), dtype = np.float32)
    self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(len(metrics)+1, self.window_size), dtype = np.float32)
    
  



  def load_of_datasets_and_interaction_spaces(self):
    # Define the upper and lower bounds of actions and observations
    self.lower_limit_actions = np.array([self.parameters[key][0] for key in self.parameters])
    self.upper_limit_actions = np.array([self.parameters[key][1] for key in self.parameters])

    self.lower_limit_observations = [np.ones(self.window_size)*self.metrics[key][0] for key in self.metrics]
    self.lower_limit_observations.append(np.zeros(self.window_size))
    self.lower_limit_observations = np.array(self.lower_limit_observations)

    self.upper_limit_observations = [np.ones(self.window_size)*self.metrics[key][1] for key in self.metrics]
    self.upper_limit_observations.append(np.ones(self.window_size)*100000) # Upper bound depending on the maximum number of classifier training timestep
    self.upper_limit_observations = np.array(self.upper_limit_observations)





  def action_scaling(self, action):
    # Scaling of actions from (-1, +1) range to respective (upper, lower) ranges
    range_ = self.upper_limit_actions - self.lower_limit_actions
    return (action + 1.0) * 0.5 * range_ + self.lower_limit_actions

  def observation_scaling(self, obs):
    # Verification of observed values to be in respective (upper, lower) ranges
    obs = np.clip(obs, self.lower_limit_observations, self.upper_limit_observations)
    # Scaling of observations from respective (upper, lower) ranges to (0, 1) range
    range_ = self.upper_limit_observations - self.lower_limit_observations
    return (obs - self.lower_limit_observations) / range_





  def step(self, hparams_scaled):
    self.n_steps += 1
    hparams = self.action_scaling(hparams_scaled)
    hparams_dict = dict(self.parameters)
    i = 0
    for key in hparams_dict:
      hparams_dict[key] = float(hparams[i])
      i += 1

    # Writing of new hyper-parameters values
    with open('%s.ser' % self.save_filename, 'wb') as fp:
      pickle.dump(hparams_dict, fp)

    try:
      stats = next(self.DNN)
      for key in self.metrics:
        if np.isnan(np.sum(stats[key])):
          return self.observation_scaling(self.obs), 0.0, True, {}
        stats[key] = np.clip(stats[key], self.metrics[key][0], self.metrics[key][1])

      self.build_observation(stats)
      # Saving logs
      for key in self.metrics:
        self.logs[key].extend(stats[key])
      for key in self.parameters:
        self.logs[key].extend(stats[key])

      if not self.test_mode and self.n_games//self.game_per_epoch < len(self.epoch_schedule) and stats['current_epoch'] >= self.epoch_schedule[self.n_games//self.game_per_epoch]:
        self.n_games+=1
        done = True
      else:
        done = False
    except StopIteration:
      done = True
      if (self.test_mode):
        print("This was a test acc game for the agent")

    obs_scaled = self.observation_scaling(self.obs)
    reward = self.build_reward()
    self.logs["reward"].append(reward)

    return obs_scaled, reward, done, {}





  def reset(self):
    self.n_steps = 0
    # Initialisation of reward metrics
    self.last_reward_metrics = {}
    self.reward_metrics = {}
    for key in self.reward_coefs:
      self.last_reward_metrics[key] = self.reward_metrics[key] = 0.0

    self.DNN = iterator(self.train_steps, self.valid_steps, self.save_filename, self.corruption)
    stats = next(self.DNN)
    # We verify the values are not NaN and within the ranges
    for key in self.metrics:
      if np.isnan(np.sum(stats[key])):
        return self.reset()
      stats[key] = np.clip(stats[key], self.metrics[key][0], self.metrics[key][1])

    # Initialisation of running average
    self.running_avg = {}
    for key in self.metrics:
      self.running_avg[key] = [mean(stats[key])]*self.running_avg_lgt
    self.build_observation(stats, init=True)
    self.build_reward(init=True)
    
    return self.obs



      


  def thresholdout(self, train, valid, adjustment=1.0):
    # Compute values on the standard holdout
    tolerance = 0.01*adjustment
    threshold = 0.04*adjustment

    if abs(train-valid) < threshold + np.random.normal(0,tolerance):
        valid = train
    else:
        valid += np.random.normal(0,tolerance)
    
    return valid

        

  def update_running_avg(self, stats):
    for key in self.metrics:
      if len(stats[key]) > self.running_avg_lgt:
        stats[key] = stats[key][-self.running_avg_lgt:]
    for key in self.metrics:
      for i in range(self.running_avg_lgt-len(stats[key])):
        self.running_avg[key][i] = self.running_avg[key][i+len(stats[key])]
      for i in range(len(stats[key])):
        self.running_avg[key][self.running_avg_lgt-len(stats[key])+i] = stats[key][i]




  def build_observation(self, stats, init=False):
    # We do not update running average when the game starts as it is initialized in reset()
    if not init:
      self.update_running_avg(stats)

    for key in self.not_val_metrics:
      self.not_val_metrics[key] = mean(self.running_avg[key])
      if key in self.reward_metrics:
        self.reward_metrics[key] = self.not_val_metrics[key]

    if self.n_steps % self.steps_between_thresholdouts == 0:
        for key in self.val_metrics:
          self.val_metrics[key] = mean(self.running_avg[key])
          if key in self.reward_metrics:
            self.reward_metrics[key] = self.val_metrics[key]
        if self.do_thresholdout:
          for key in self.val_metrics:
            self.val_metrics[key] = self.thresholdout(self.not_val_metrics[self.metrics[key][3]], self.val_metrics[key], self.metrics[key][2])

    # If the game begins, we initialize the obs matrix
    if init:
      self.obs = np.zeros((len(metrics)+1, self.window_size), dtype=np.float32)

    # We update it with last obtained stats
    self.obs = np.roll(self.obs, -1, axis=1)

    i = 0
    for key in self.not_val_metrics:
      self.obs[i][-1] = self.not_val_metrics[key]
      i += 1
    for key in self.val_metrics:
      self.obs[i][-1] = self.val_metrics[key]
      i += 1
    self.obs[i][-1] = mean(stats['global_step'])





  def build_reward(self, init=False):
    reward = 0.0
    # Calculation of validation differences
    # Calculation of training differences
    # Update of last metrics
    if init or self.n_steps % self.steps_between_thresholdouts == 0:
      for key in self.reward_coefs:
        if key in self.val_metrics:
          reward += self.reward_coefs[key] * (self.reward_metrics[key] - self.last_reward_metrics[key])
          self.last_reward_metrics[key] = self.reward_metrics[key]
    if init or self.n_steps % self.steps_between_thresholdouts != 0:
      for key in self.reward_coefs:
        if not key in self.val_metrics:
          reward += self.reward_coefs[key] * (self.reward_metrics[key] - self.last_reward_metrics[key])
          self.last_reward_metrics[key] = self.reward_metrics[key]
    return reward
    




  def reset_stat(self):
    self.logs = {**self.metrics, **self.parameters}
    for key in self.logs:
      self.logs[key] = []
    self.logs["reward"] = []

  def get_stat(self):
    return self.logs







# Creation of RL environment that will be used for training
env = CustomEnv(parameters, metrics, val_metrics, not_val_metrics, reward_coefs, save_filename, corruption)
# Creation of action noise that will be added to agent output policy
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))  
# Creation of RL environment that will be used for evaluating the agent performance
valid_env = CustomEnv(parameters, metrics, val_metrics, not_val_metrics, reward_coefs, save_filename, corruption, test_mode=True)
policy_kwargs = dict(normalize_images=False)
eval_callback = EvalCallback(valid_env, verbose=1, deterministic=True, render=False, n_eval_episodes=2, eval_freq=1000, best_model_save_path="./Agents/%s/" % save_filename)
# Creation of the RL agent and start of the learning process
model = SAC("MlpPolicy", env, action_noise=action_noise, policy_kwargs=policy_kwargs, verbose=0)
model.learn(callback=eval_callback, total_timesteps=10000)








# Creation of RL environment that will be used for testing the performance of the agent
test_env = CustomEnv(parameters, metrics, val_metrics, not_val_metrics, reward_coefs, save_filename, corruption, test_mode=True)
# Load of the agent that yielding the best performance when evaluated
model = SAC.load("Agents/%s/best_model.zip" % save_filename)
# Test of the agent on 3 games
for ep in range(3):
  obs = test_env.reset()
  done = False
  while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = test_env.step(action[0])





# Creation of directory for logs
if not os.path.exists('Results'):
  os.makedirs('Results')
if not os.path.exists('Results/%s' % save_filename):
  os.makedirs('Results/%s' % save_filename)
# Save of training environment statistics
logs = env.get_stat()
with open('Results/%s/train.ser' % save_filename, 'wb') as fp:
  pickle.dump(logs, fp)
# Save of validation environment statistics
logs = valid_env.get_stat()
with open('Results/%s/valid.ser' % save_filename, 'wb') as fp:
  pickle.dump(logs, fp)
# Save of test environment statistics
logs = test_env.get_stat()
with open('Results/%s/test.ser' % save_filename, 'wb') as fp:
  pickle.dump(logs, fp)
# Save of training configuration
with open('Results/%s/config.txt' % save_filename, 'w') as fp:
  fp.write("parameters -> %s\n" % parameters)
  fp.write("metrics -> %s\n" % metrics)
  fp.write("val_metrics -> %s\n" % val_metrics)
  fp.write("not_val_metrics -> %s\n" % not_val_metrics)
  fp.write("reward_coefs -> %s\n" % reward_coefs)
  fp.write("corruption -> %s\n" % corruption)
  fp.write("train_steps -> %s\n" % test_env.train_steps)
  fp.write("valid_steps -> %s\n" % test_env.valid_steps)
  fp.write("epoch_schedule -> %s\n" % test_env.epoch_schedule)
  fp.write("game_per_epoch -> %s\n" % test_env.game_per_epoch)
  fp.write("steps_between_thresholdouts -> %s\n" % test_env.steps_between_thresholdouts)
  fp.write("running_avg_lgt -> %s\n" % test_env.running_avg_lgt)
  fp.write("window_size -> %s\n" % test_env.window_size)
  fp.write("init_hyperparams -> %s\n" % init_hyperparams)