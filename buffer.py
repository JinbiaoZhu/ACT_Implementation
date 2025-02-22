import os
import numpy as np
import torch


class ReplayBuffer(object):
    """
    Buffer to store environment transitions
    """
    def __init__(self, proprioception_shape, obs_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.proes = np.empty((capacity, *proprioception_shape), dtype=np.float32)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_proes = np.empty((capacity, *proprioception_shape), dtype=np.float32)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

        self.train_index, self.valid_index, self.test_index = [], [], []

    def add(self, pro, obs, action, reward, next_pro, next_obs, done):
        np.copyto(self.proes[self.idx], pro)
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_proes[self.idx], next_pro)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.dones[self.idx], done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
