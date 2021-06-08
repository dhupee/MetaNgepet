from gym.envs.registration import register
from copy import deepcopy

register(
    id='live-forex v0',
    entry_point='gym_metalive.envs:ForexEnv',
    kwargs={
        'window_size': 24,
        'frame_bound': 24
    }
)
