import gymnasium as gym

import tutorial2.pyflow as pf


def get_agent_model():
    return pf.GeneticModel(
        [
            pf.GeneticDense(11, 64, activation="tanh", kernel_initializer="gorot"),
            pf.GeneticDense(64, 2, activation="tanh", kernel_initializer="gorot"),
        ]
    )


def main():
    env = gym.make("tutorial1/Tutorial1-v1", render_mode="human", agent_count=5)

    observation, info = env.reset()

    for _ in range(10000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
