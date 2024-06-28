import random

import gymnasium as gym
import numpy as np
import pyray as pr

from tutorial1.constants import (
    APP_NAME,
    FRAME_RATE,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from tutorial1.scenes import trainer


class Tutorial1Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 5}  # type: ignore # noqa: RUF012

    def __init__(self, render_mode=None, agent_count=10):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.agent_count = agent_count
        self._gfx_initialized = False

        agent_space = gym.spaces.Dict(
            {
                "agent_vel": gym.spaces.Box(-1, 1, shape=(1, 1), dtype=np.float64),
                "agent_cam": gym.spaces.Box(0, 1, shape=(1, 16), dtype=np.float64),
            }
        )
        self.observation_space = gym.spaces.Sequence(agent_space)

        self.action_space = gym.spaces.Box(-1, 1, shape=(agent_count, 2), dtype=np.float64)

        trainer.spawn_agents(agent_count)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        trainer.reset_agents()
        trainer.reset()

        return self._get_obs(), self._get_info()

    def step(self, action):
        for i, agent in enumerate(trainer.get_agents()):
            throttle, wheel = action[i]
            agent.push_throttle(throttle)
            agent.turn_wheel(wheel)

        trainer.update(1 / FRAME_RATE)

        terminated = trainer.is_terminated()

        if self.render_mode == "human":
            if not self._gfx_initialized:
                self._gfx_init()
                self._gfx_initialized = True
            self._gfx_render()

        return self._get_obs(), 0, terminated, False, self._get_info()

    def close(self):
        if self.render_mode == "human" and self._gfx_initialized:
            self._gfx_close()

    def _get_obs(self):
        return [trainer.get_agent_obs(x) for x in trainer.get_agents()]

    def _get_info(self):
        return {"scores": [trainer.get_agent_score(x) for x in trainer.get_agents()]}

    def _gfx_init(self):
        pr.set_config_flags(pr.ConfigFlags.FLAG_MSAA_4X_HINT)
        pr.init_window(WINDOW_WIDTH, WINDOW_HEIGHT, APP_NAME)
        pr.set_target_fps(self.metadata["render_fps"])
        pr.hide_cursor()
        pr.init_audio_device()

    def _gfx_render(self):
        pr.begin_drawing()
        trainer.draw()
        pr.end_drawing()

    def _gfx_close(self):
        pr.close_window()
