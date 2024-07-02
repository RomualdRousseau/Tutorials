import os
import time
from typing import Optional

import fire
import gymnasium as gym
import numpy as np
from tqdm import trange

import tutorial2.pyflow as pf

BATCH_SIZE = 32
BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt}"


class Agent:
    CK = np.array([0.25, 0.5, 0.25])

    def __init__(self, model: Optional[pf.Sequential] = None, mutate: bool = False, lr: float = 0.1) -> None:
        self.fitness = 0.0

        self.model = get_agent_model() if model is None else model.clone()
        self.model.compile(optimizer=pf.optimizers.sgd(momentum=0.9, lr=lr, nesterov=True))

        if mutate:
            self.model.fit(epochs=1, shuffle=False, verbose=False)

    def get_model(self) -> pf.Sequential:
        return self.model

    def get_fitness(self) -> float:
        return self.fitness

    def set_fitness(self, fitness: float) -> pf.GeneticIndividual:
        self.fitness = fitness
        return self

    def get_action(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        vel, cam = observation["agent_vel"], observation["agent_cam"]
        x = np.concat([vel, np.convolve(cam, Agent.CK, "same")])
        y = self.model.predict(x)
        return y[0]


def get_agent_model() -> pf.Sequential:
    return pf.Sequential(
        [
            pf.layers.GeneticDense(17, 32, activation="leaky_relu"),
            pf.layers.GeneticDense(32, 16, activation="leaky_relu"),
            pf.layers.GeneticDense(16, 2, activation="tanh"),
        ],
        trainer=pf.GeneticTrainer(rate=0.1, variance=1),
    )


def spawn_agents(
    agent_count: int, model_or_pool: Optional[pf.Sequential] | pf.GeneticPool, mutate: bool, lr: float = 0.1
) -> list[Agent]:
    get_model = lambda: (
        model_or_pool.select_parent().get_model() if isinstance(model_or_pool, pf.GeneticPool) else model_or_pool
    )
    return [
        Agent(get_model(), mutate, lr)
        for _ in trange(agent_count, desc="Spawning agents", ncols=120, bar_format=BAR_FORMAT)
    ]


def main(
    agent_count: int = 100,
    render_fps: Optional[int] = None,
    seed: int = 5,
    learning_rate: float = 0.1,
    mode: str = "training",
    model_file: Optional[str] = None,
    duration: float = 15.0,
) -> None:
    """
    Welcome to the taxi driver simulation tutorial!

    In this tutorial, you will learn how to create your own Gymnasium agent. Gymnasium is a toolkit for developing and
    comparing reinforcement learning algorithms. Here, we will guide you through the steps to create a machine learning
    model that will train the car in the Taxi Driver Environment to drive safely between 2 locations in the city.

    Params:
    -------
    agent_count: Number of agents to run in the simulation.
    render_fps: Set the frame per second.
    seed: Initialize the random generators and make the simulation reproductible.
    learning_rate: Set the learning rate. First training at 0.1 and then 0.01.
    mode: Set the mode; 'training' or 'validation'.
    model_file: Load the model file to initialize the agents.
    duration: Duration in minutes of the simulation.
    """

    assert agent_count > 0
    assert seed >= 0
    assert mode in ("training", "validation")
    assert mode == "training" or mode == "validation" and model_file is not None
    assert duration > 0

    env = gym.make("tutorial1/Tutorial1-v1", agent_count=agent_count, render_mode="human", render_fps=render_fps)

    if model_file is not None and os.path.exists(model_file):
        best_model = get_agent_model()
        best_model.load(model_file)
    else:
        best_model = None

    agents = spawn_agents(agent_count, best_model, False, learning_rate)
    observation, info = env.reset(seed=seed)

    t_end = time.monotonic() + 60 * duration
    while time.monotonic() < t_end:
        action = [agent.get_action(obs) for agent, obs in zip(agents, observation, strict=True)]

        observation, _, terminated, truncated, info = env.step(action)

        best_model = max(zip(agents, info["scores"], strict=True), key=lambda x: x[1])[0].model

        if terminated or truncated:
            if mode == "training":
                pool = pf.GeneticPool(
                    [agent.set_fitness(score) for agent, score in zip(agents, info["scores"], strict=True)]
                )
                pool.sample()
                pool.normalize()
                agents = spawn_agents(agent_count, pool, True, learning_rate)
            else:
                agents = spawn_agents(agent_count, best_model, False, learning_rate)

            observation, info = env.reset()

    if mode == "training" and model_file is not None and best_model is not None:
        best_model.save(model_file)

    env.close()


if __name__ == "__main__":
    fire.Fire(main)
