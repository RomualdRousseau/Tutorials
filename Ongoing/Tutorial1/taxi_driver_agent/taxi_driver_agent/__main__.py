import gymnasium as gym


def main():
    env = gym.make("taxi_driver_env/TaxiDriverEnv-v0", render_mode="human")

    observation, info = env.reset()

    for _ in range(10):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        print(f"observation: {observation}")

        if terminated or truncated:
            observation, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
