from gymnasium.envs.registration import register

register(
    id="tutorial1/Tutorial1-v1",
    entry_point="tutorial1.envs:Tutorial1Env",
)
