import os
from typing import Optional

import gymnasium as gym
import numpy as np
from tqdm import trange

import tutorial2.pyflow as pf

AGENT_COUNT = 100
GAME_SEED = 5

BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt}"


def get_agent_model():
    return pf.Sequential(
        [
            pf.GeneticDense(11, 32, activation="linear", kernel_initializer="gorot"),
            pf.GeneticDense(32, 2, activation="tanh", kernel_initializer="gorot"),
        ],
        trainer=pf.GeneticTrainer(),
    )


class Agent:
    CK = np.array([0.25, 0.5, 0.25])

    def __init__(self, parent_model: Optional[pf.Sequential] = None, mutate: bool = False):
        self.fitness = -1.0

        self.model = get_agent_model() if parent_model is None else parent_model.clone()
        self.model.compile(optimizer="rmsprop")
        if mutate:
            self.model.fit(epochs=1, shuffle=False, verbose=False)

    def get_action(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        vel = observation["agent_vel"]
        cam = observation["agent_cam"]
        x = np.concat([vel, pf.conv1d(cam, Agent.CK)])
        y = self.model.predict(x)
        return y[0]

    def get_model(self) -> pf.Sequential:
        return self.model

    def get_fitness(self) -> float:
        return self.fitness

    def set_fitness(self, fitness: float) -> None:
        self.fitness = fitness


def main():
    if os.path.exists("agent_model.json"):
        base_model = get_agent_model()
        base_model.load("agent_model.json")
    else:
        base_model = None

    env = gym.make("tutorial1/Tutorial1-v1", render_mode="human", agent_count=AGENT_COUNT)

    agents = [
        Agent(base_model) for _ in trange(AGENT_COUNT, desc="Generating agents", ncols=120, bar_format=BAR_FORMAT)
    ]

    observation, info = env.reset(seed=GAME_SEED)

    while True:
        action = [a.get_action(observation[i]) for i, a in enumerate(agents)]

        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            for i, a in enumerate(agents):
                a.set_fitness(info["scores"][i] + reward)

            pool = pf.GeneticPool(agents)
            pool.sample()
            pool.normalize()
            pool.pool[0].model.save("agent_model.json")

            agents = [
                Agent(pool.select_parent().get_model(), True)
                for _ in trange(AGENT_COUNT, desc="Training agents", ncols=120, bar_format=BAR_FORMAT)
            ]

            observation, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
