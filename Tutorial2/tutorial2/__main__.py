import os
import signal
from typing import Optional

import fire
import gymnasium as gym
import numpy as np
from tqdm import trange

import tutorial2.pyflow as pf

BATCH_SIZE = 32
BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt}"

should_quit = False


def install_quit_handler():
    signal.signal(signal.SIGINT, handler_quit)


def handler_quit(signum, frame):
    global should_quit  # noqa PLW0603
    should_quit = True


def get_agent_model():
    return pf.Sequential(
        [
            pf.layers.GeneticDense(17, 32, activation="leaky_relu"),
            pf.layers.GeneticDense(32, 16, activation="leaky_relu"),
            pf.layers.GeneticDense(16, 2, activation="tanh"),
        ],
        trainer=pf.GeneticTrainer(rate=0.1, variance=0.1),
    )


class Agent:
    CK = np.array([0.25, 0.5, 0.25])

    def __init__(self, parent_model: Optional[pf.Sequential] = None, mutate: bool = False, timestep: int = 0):
        self.fitness = -1.0

        self.model = get_agent_model() if parent_model is None else parent_model.clone()
        self.model.compile(optimizer=pf.optimizers.rmsprop(lr=0.01))

        if mutate:
            if (timestep % BATCH_SIZE) == 0:
                self.model.zero_grad()
            self.model.fit(epochs=1, shuffle=False, verbose=False)

    def get_action(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        vel = observation["agent_vel"]
        cam = observation["agent_cam"]
        x = np.concat([vel, np.convolve(cam, Agent.CK, "same")])
        y = self.model.predict(x)
        return y[0]

    def get_model(self) -> pf.Sequential:
        return self.model

    def get_fitness(self) -> float:
        return self.fitness

    def set_fitness(self, fitness: float) -> None:
        self.fitness = fitness


def main(agent_count: int = 100, seed: int = 5, model_file: Optional[str] = None) -> None:
    """
    Welcome to the taxi driver simulation tutorial!

    In this tutorial, you will learn how to create your own Gymnasium agent. Gymnasium is a toolkit for developing and
    comparing reinforcement learning algorithms. Here, we will guide you through the steps to create a machine learning
    model that will train the car in the Taxi Driver Environment to drive safely between 2 locations in the city.
    """

    env = gym.make("tutorial1/Tutorial1-v1", render_mode="human", agent_count=agent_count)

    if os.path.exists("agent_model.json"):
        best_model = get_agent_model()
        best_model.load("agent_model.json")
    else:
        best_model = None

    agents = [Agent(best_model) for _ in trange(agent_count, desc="Spawning agents", ncols=120, bar_format=BAR_FORMAT)]

    observation, info = env.reset(seed=seed)

    timestep = 0

    while not should_quit:
        action = [a.get_action(observation[i]) for i, a in enumerate(agents)]

        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            for agent, score in zip(agents, info["scores"], strict=True):
                agent.set_fitness(score + reward)

            pool = pf.GeneticPool(agents)  # type: ignore
            pool.sample()
            pool.normalize()
            best_model = pool.best_parent().get_model()

            if info["spawn_location_changed"] and model_file is not None:
                best_model.save(model_file)

            agents = [
                Agent(pool.select_parent().get_model(), True, timestep)
                for _ in trange(agent_count, desc="Training agents", ncols=120, bar_format=BAR_FORMAT)
            ]

            observation, info = env.reset()

            timestep += 1

    if model_file is not None:
        best_model.save(model_file)

    env.close()


if __name__ == "__main__":
    install_quit_handler()
    fire.Fire(main)
