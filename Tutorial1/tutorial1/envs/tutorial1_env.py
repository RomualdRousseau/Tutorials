import pyray as pr
import numpy as np
import gymnasium as gym

from tutorial1.constants import FRAME_RATE, WINDOW_HEIGHT, WINDOW_WIDTH, APP_NAME
from tutorial1.scenes import gameloop


class Tutorial1Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, size=5):
        self.observation_space = gym.spaces.Dict(
            {
                "agent_pos": gym.spaces.Box(0, WINDOW_WIDTH, shape=(2,), dtype=np.float64),
                "agent_vel": gym.spaces.Box(-WINDOW_WIDTH, WINDOW_WIDTH, shape=(2,), dtype=np.float64),
            }
        )

        self.action_space = gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float64)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self._gfx_initialized = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        gameloop.reset()
        return self._get_obs(), {}

    def step(self, action):
        gameloop.update(pr.get_frame_time())

        if self.render_mode == "human":
            if not self._gfx_initialized:
                self._gfx_init()
                self._gfx_initialized = True
            self._gfx_render()

        return self._get_obs(), 0, False, False, {}

    def close(self):
        if self.render_mode == "human" and self._gfx_initialized:
            self._gfx_close()

    def _get_obs(self):
        return {
            "agent_pos": gameloop._context.player.pos,
            "agent_vel": gameloop._context.player.vel,
        }

    def _gfx_init(self):
        pr.set_config_flags(pr.ConfigFlags.FLAG_MSAA_4X_HINT)
        pr.init_window(WINDOW_WIDTH, WINDOW_HEIGHT, APP_NAME)
        pr.set_target_fps(FRAME_RATE)
        pr.hide_cursor()
        pr.init_audio_device()

    def _gfx_render(self):
        pr.begin_drawing()
        gameloop.draw()
        pr.end_drawing()

    def _gfx_close(self):
        pr.close_window()
