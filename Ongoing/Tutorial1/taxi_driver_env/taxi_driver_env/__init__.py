from gymnasium.envs.registration import register

register(
    id="taxi_driver_env/TaxiDriverEnv-v0",
    entry_point="taxi_driver_env.envs:TaxiDriverEnv",
)