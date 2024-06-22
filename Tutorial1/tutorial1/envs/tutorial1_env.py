import gymnasium as gym
import numpy as np
import pyray as pr

from tutorial1.constants import (
    APP_NAME,
    FRAME_RATE,
    VIRTUAL_WIDTH,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from tutorial1.entities import car
from tutorial1.entities.world import RAY_MAX_LEN
from tutorial1.scenes import trainer


class Tutorial1Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": FRAME_RATE}  # type: ignore # noqa: RUF012

    def __init__(self, render_mode=None, agent_count=10):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.agent_count = agent_count

        self._gfx_initialized = False

        self.observation_space = gym.spaces.Sequence(
            gym.spaces.Dict(
                {
                    "agent_pos": gym.spaces.Box(-VIRTUAL_WIDTH, VIRTUAL_WIDTH, shape=(2,), dtype=np.float64),
                    "agent_vel": gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float64),
                    "agent_cam": gym.spaces.Box(0, 1, shape=(10,), dtype=np.float64),
                }
            )
        )

        self.action_space = gym.spaces.Box(-1, 1, shape=(agent_count, 2), dtype=np.float64)

        trainer.add_agents(agent_count)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        trainer.reset()
        return self._get_obs(), {}

    def step(self, action):
        for i, agent in enumerate(trainer.get_agents()):
            throttle, wheel = action[i]
            agent.push_throttle(throttle)
            agent.turn_wheel(wheel)

        trainer.update(pr.get_frame_time())

        terminated = trainer._context.best_car is None

        if self.render_mode == "human":
            if not self._gfx_initialized:
                self._gfx_init()
                self._gfx_initialized = True
            self._gfx_render()

        return self._get_obs(), 0, terminated, False, {}

    def _get_obs(self):
        def get_agent_obs(agent: car.Car):
            return {
                "agent_pos": agent.pos,
                "agent_vel": agent.get_speed_in_kmh() / car.MAX_SPEED,
                "agent_cam": [1 - x.length / RAY_MAX_LEN for x in agent.camera],
            }

        return [get_agent_obs(x) for x in trainer.get_agents()]

    def close(self):
        if self.render_mode == "human" and self._gfx_initialized:
            self._gfx_close()

    def _gfx_init(self):
        pr.set_config_flags(pr.ConfigFlags.FLAG_MSAA_4X_HINT)
        pr.init_window(WINDOW_WIDTH, WINDOW_HEIGHT, APP_NAME)
        pr.set_target_fps(FRAME_RATE)
        pr.hide_cursor()
        pr.init_audio_device()

    def _gfx_render(self):
        pr.begin_drawing()
        trainer.draw()
        pr.end_drawing()

    def _gfx_close(self):
        pr.close_window()
