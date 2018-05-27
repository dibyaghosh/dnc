from dnc.envs.picker import PickerEnv
from dnc.envs.lob import LobberEnv
from dnc.envs.ant import AntEnv
from dnc.envs.catch import CatchEnv

from dnc.envs.base import create_env_partitions

import numpy as np

_envs = {
    'pick': PickerEnv,
    'lob': LobberEnv,
    'catch': CatchEnv,
    'ant': AntEnv
}

_stochastic_params = {
    'pick': dict(goal_args=('noisy',(.6,.2),.1)),
    'lob': dict(box_center=(0,0), box_noise=0.4),
    'catch': dict(start_pos=(.1,1.7), start_noise=0.2),
    'ant': dict(angle_range=(0,2*np.pi)),
}

_deterministic_params = {
    'pick': dict(goal_args=('noisy',(.6,.2),0)),
    'lob': dict(box_center=(0,0), box_noise=0),
    'catch': dict(start_pos=(.1,1.7), start_noise=0),
    'ant': dict(angle_range=(-1e-4,1e-4)),
}

def create_stochastic(name):
    assert name in _stochastic_params
    return _envs[name](**_stochastic_params[name])

def create_deterministic(name):
    assert name in _deterministic_params
    return _envs[name](**_deterministic_params[name])

def test_env(env, n_rolls=5, n_steps=50):
    for i in range(n_rolls):
        env.reset()
        for t in range(n_steps):
            env.step(env.action_space.sample())
            env.render()
    env.render(close=True)