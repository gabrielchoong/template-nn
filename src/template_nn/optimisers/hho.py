# author: Ali Asghar Heidari, Hossam Faris
# link: https://aliasgharheidari.com/HHO.html
# date accessed: 2025-01-27

import math
import random
import time
from typing import Callable, List

import numpy as np

from template_nn.optimisers.levy import Levy
from template_nn.optimisers.solution import Solution


class HHO:
    """
    Usage:
    from template_nn.optimisers.hho import HHO

    hho_config: dict { # provide your config in dict format }
    hho = HHO(hho_config)
    hho.optimise()
    """

    def __init__(self, tabular: dict) -> None:
        """

        :param tabular:
        """
        self.objective_function: Callable | List[Callable] = tabular["objective_function"]
        self.lower_bound: int = tabular["lower_bound"]
        self.upper_bound: int = tabular["upper_bound"]
        self.dimension: int = tabular["dimension"]
        self.search_agents_num: int = tabular["search_agents_num"]
        self.max_iterations: int = tabular["max_iterations"]

    def optimise(self) -> Solution:
        """

        :return:
        """
        if not isinstance(self.lower_bound, list):
            self.lower_bound = [self.lower_bound for _ in range(self.dimension)]
            self.upper_bound = [self.upper_bound for _ in range(self.dimension)]

        self.lower_bound = np.asarray(self.lower_bound)
        self.upper_bound = np.asarray(self.upper_bound)

        # initial hawk position
        Hawks = np.asarray(
            [i * (self.upper_bound - self.lower_bound) + self.lower_bound
             for i in np.random.uniform(
                low=0,
                high=1,
                size=(self.search_agents_num, self.dimension))
             ]
        )

        rabbit_location = np.zeros(self.dimension)

        rabbit_energy = float("inf")

        convergence_curve = np.zeros(self.max_iterations)

        s = Solution()

        print(f"HHO is now tackling \"{self.objective_function.__name__}\"")

        timerStart = time.time()
        s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

        t = 0

        while t < self.max_iterations:

            for i in range(0, self.search_agents_num):

                # check boundaries
                Hawks[i, :] = np.clip(
                    Hawks[i, :],
                    self.lower_bound,
                    self.upper_bound
                )

                # fitness of locations
                fitness = self.objective_function(Hawks[i, :])

                if fitness < rabbit_energy:
                    rabbit_energy = fitness
                    rabbit_location = Hawks[i, :].copy()

            E1 = 2 * (1 - (t / self.max_iterations))

            for i in range(0, self.search_agents_num):

                E0 = 2 * random.random() - 1
                escaping_energy = E1 * E0

                # exploration
                if abs(escaping_energy) >= 1:

                    # hawks random perch based on 2 strategies
                    q = random.random()
                    random_hawk_index = math.floor(
                        self.search_agents_num * random.random())

                    Hawk_random = Hawks[random_hawk_index, :]

                    if q < 0.5:
                        # perch based on other family members
                        Hawks[i, :] = Hawk_random - random.random() * \
                                      abs(Hawk_random - 2 * random.random() * Hawks[i, :])
                    else:
                        # perch on nearby tree branch
                        Hawks[i, :] = (rabbit_location - Hawks.mean(0)) - \
                                      random.random() * ((self.upper_bound - self.lower_bound) *
                                                         random.random() + self.lower_bound)

                # exploitation
                # if abs(escaping_energy) < 1:
                else:

                    # 4 strategies to attack the rabbit
                    r = random.random()

                    # phase 1: surprise pounce (7 kills)
                    if r >= 0.5 and abs(escaping_energy) < 0.5:
                        Hawks[i, :] = rabbit_location - escaping_energy * abs(rabbit_location - Hawks[i, :])

                    if r >= 0.5 and abs(escaping_energy) >= 0.5:
                        jump_strength = 2 * (1 - random.random())
                        Hawks[i, :] = (rabbit_location - Hawks[i, :]) - escaping_energy * abs(
                            jump_strength * rabbit_location - Hawks[i, :]
                        )

                    # phase 2: team rapid dives (leapfrog movement)
                    if r < 0.5 and abs(escaping_energy) >= 0.5:
                        jump_strength = 2 * (1 - random.random())
                        X1 = rabbit_location - escaping_energy * abs(jump_strength * rabbit_location - Hawks[i, :])
                        X1 = np.clip(X1, self.lower_bound, self.upper_bound)

                        if self.objective_function(X1) < fitness:
                            Hawks[i, :] = X1.copy()
                        else:
                            X2 = rabbit_location - escaping_energy * abs(
                                jump_strength * rabbit_location - Hawks[i, :]
                            ) + np.multiply(
                                np.random.rand(self.dimension), Levy(self.dimension)
                            )
                            X2 = np.clip(X2, self.lower_bound, self.upper_bound)
                            if self.objective_function(X2) < fitness:
                                Hawks[i, :] = X2.copy()

                    if r < 0.5 and abs(escaping_energy) < 0.5:
                        jump_strength = 2 * (1 - random.random())
                        X1 = rabbit_location - escaping_energy * abs(jump_strength * rabbit_location - Hawks.mean(0))
                        X1 = np.clip(X1, self.lower_bound, self.upper_bound)

                        if self.objective_function(X1) < fitness:
                            Hawks[i, :] = X1.copy()
                        else:
                            X2 = rabbit_location - escaping_energy * abs(
                                jump_strength * rabbit_location - Hawks.mean(0)
                            ) + np.multiply(
                                np.random.rand(self.dimension), Levy(self.dimension)
                            )

                            X2 = np.clip(X2, self.lower_bound, self.upper_bound)

                            if self.objective_function(X2) < fitness:
                                Hawks[i, :] = X2.copy()

            convergence_curve[t] = rabbit_energy
            if t % 1 == 0:
                print(['At iteration ' + str(t) + ' the best fitness is ' + str(rabbit_energy)])
            t = t + 1

        timerEnd = time.time()
        s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
        s.executionTime = timerEnd - timerStart
        s.convergenceCurve = convergence_curve
        s.optimizer = "HHO"
        s.objname = self.objective_function.__name__
        s.best = rabbit_location
        s.bestIndividual = rabbit_location

        return s
