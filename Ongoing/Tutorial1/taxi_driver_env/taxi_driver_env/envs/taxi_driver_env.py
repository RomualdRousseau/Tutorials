import numpy as np
import gymnasium as gym


class TaxiDriverEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3,), dtype=np.float64)
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        print("reset")
        return self._get_obs(), {}

    def step(self, action):
        print(f"step: {action}")
        return self._get_obs(), 0, False, False, {}

    def close(self):
        print("close")

    def _get_obs(self):
        return np.zeros(3)
