# Basic idea

This repository is exploring the use of Reinforcement Learning to automaticaly optimize deep neural networks' hyper-parameters. 

# Reinforcement Learning

The library used for Reinforcement Learning (RL) is Stable Baselines 3. The proposed contribution simply performs hyper-parameter optimization by using RL selected values. To this end, a Soft Actor Critic (SAC) agent is trained to select, for one or multiple hyper-parameters, their respective values at each validation batch iteration. Even if the environement is currently tested on a single dataset for a single DNN, it was designed to be easily reusable for different configurations.

## CNN Experiment with RL

To train SAC on hyper-parameters tuning:
```
python RL_env.py
```
You can modify the metrics seen by the RL agent and the available hyper-parameters in the file.