import datetime
import io
import random
import traceback
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import IterableDataset


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open("wb") as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open("rb") as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


class ReplayBufferStorage:
    def __init__(self, data_specs, replay_dir, use_relabeling, is_demo_buffer=False):
        self._data_specs = data_specs
        self._replay_dir = replay_dir
        self._use_relabeling = use_relabeling
        self._is_demo_buffer = is_demo_buffer
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._preload()

    def __len__(self):
        return self._num_transitions

    def add(self, time_step):
        for spec in self._data_specs:
            value = time_step[spec.name]
            # Remove frame stacking
            if spec.name == "low_dim_obs":
                low_dim = spec.shape[0]
                value = value[..., -low_dim:]
            elif spec.name == "rgb_obs":
                rgb_dim = spec.shape[1]
                value = value[:, -rgb_dim:]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype, (
                spec.name,
                spec.shape,
                value.shape,
                spec.dtype,
                value.dtype,
            )
            self._current_episode[spec.name].append(value)
        if time_step.last():
            episode = dict()
            for spec in self._data_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            self._current_episode = defaultdict(list)
            if self._use_relabeling:
                episode = self._relabel_episode(episode)
            if self._is_demo_buffer:
                # If this is demo replay buffer, save only when it's successful
                if self._check_if_successful(episode):
                    self._store_episode(episode)
            else:
                self._store_episode(episode)

    def _relabel_episode(self, episode):
        if self._check_if_successful(episode):
            episode["demo"] = np.ones_like(episode["demo"])
        return episode

    def _check_if_successful(self, episode):
        reward = episode["reward"]
        return np.isclose(reward[-1], 1.0)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob("*.npz"):
            _, _, eps_len = fn.stem.split("_")
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        eps_fn = f"{ts}_{eps_idx}_{eps_len}.npz"
        save_episode(episode, self._replay_dir / eps_fn)


class ReplayBuffer(IterableDataset):
    def __init__(
        self,
        replay_dir,
        max_size,
        num_workers,
        nstep,
        discount,
        action_sequence,
        frame_stack,
        fetch_every,
        save_snapshot,
        fill_action="last_action",
    ):
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._action_sequence = action_sequence
        self._frame_stack = frame_stack
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot
        self._fill_action = fill_action

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob("*.npz"), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split("_")[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
        next_idx = idx + self._nstep - 1

        obs_idxs = list(
            map(
                lambda x: np.clip(x, 0, None),
                range((idx - 1) - self._frame_stack + 1, (idx - 1) + 1),
            )
        )
        obs_next_idxs = list(
            map(
                lambda x: np.clip(x, 0, None),
                range(next_idx - self._frame_stack + 1, next_idx + 1),
            )
        )

        # rgb_obs stacking -- channel-wise concat
        rgb_obs = np.concatenate(episode["rgb_obs"][obs_idxs], 1)
        next_rgb_obs = np.concatenate(episode["rgb_obs"][obs_next_idxs], 1)
        # low_dim_obs stacking -- last-dim-wise concat
        low_dim_obs = np.concatenate(episode["low_dim_obs"][obs_idxs], -1)
        next_low_dim_obs = np.concatenate(episode["low_dim_obs"][obs_next_idxs], -1)

        # Sampling action sequence
        action = episode["action"][idx : idx + self._action_sequence]
        if action.shape[0] < self._action_sequence:
            diff = self._action_sequence - action.shape[0]
            if self._fill_action == "last_action":
                action = np.concatenate([action] + [action[-1:]] * diff, 0)
            elif self._fill_action == "zero_action":
                action = np.concatenate(
                    [action, np.zeros((diff, action.shape[1]), dtype=action.dtype)], 0
                )
            else:
                raise ValueError(self._fill_action)
        assert action.shape[0] == self._action_sequence
        # Flatten
        # action = action.reshape(-1)

        reward = np.zeros_like(episode["reward"][idx])
        discount = np.ones_like(episode["discount"][idx])
        for i in range(self._nstep):
            step_reward = episode["reward"][idx + i]
            reward += discount * step_reward
            _discount = episode["discount"][idx + i]
            discount *= _discount * self._discount
        demo = episode["demo"][idx]
        return (
            rgb_obs,
            low_dim_obs,
            action,
            reward,
            discount,
            next_rgb_obs,
            next_low_dim_obs,
            demo,
        )

    def __iter__(self):
        while True:
            yield self._sample()


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(
    replay_dir,
    max_size,
    batch_size,
    num_workers,
    save_snapshot,
    nstep,
    discount,
    action_sequence,
    frame_stack,
    fill_action,
):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBuffer(
        replay_dir,
        max_size_per_worker,
        num_workers,
        nstep,
        discount,
        action_sequence,
        frame_stack,
        fetch_every=100,
        save_snapshot=save_snapshot,
        fill_action=fill_action,
    )

    loader = torch.utils.data.DataLoader(
        iterable,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        worker_init_fn=_worker_init_fn,
    )
    return loader
