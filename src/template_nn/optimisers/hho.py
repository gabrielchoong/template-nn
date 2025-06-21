# author: Ali Asghar Heidari, Hossam Faris
# link: https://aliasgharheidari.com/HHO.html
# date accessed: 2025-01-27

import random
import time
from typing import Tuple
import warnings

import numpy as np
import pandas as pd

from .._utils.hho_operations import exploration, exploitation
from .._utils.model_compose import get_params
from .._utils.solution import Solution


class HHO:
    """
    Usage:
    from template_nn.optimisers.hho import HHO

    hho_config: dict { # provide your config in dict format }
    hho = HHO(hho_config)
    hho.optimise()
    """

    def __init__(self, tabular: dict | pd.DataFrame) -> None:
        warnings.warn(
            "The HHO optimiser is experimental and is deprecated as of version 0.1.4 and will be removed in 0.1.6.",
            DeprecationWarning,
            stacklevel=2
        )
        """
        :param tabular: A dict or pd.DataFrame input
        """

        keys = ("objective_function", "lower_bound", "upper_bound", "dimension", "search_agents_num", "max_iterations")

        self.objective_function, self.lower_bound, self.upper_bound, self.dimension, self.search_agents_num, self.max_iterations = \
            get_params(tabular, keys)

    def _initialise(self) -> Tuple:

        # --- Initialisation / Preparation Phase ---
        # ensure lower and upper bound are of correct dimension
        if not isinstance(self.lower_bound, list):
            self.lower_bound = [self.lower_bound for _ in range(self.dimension)]
            self.upper_bound = [self.upper_bound for _ in range(self.dimension)]

        # convert bounds for vectorised operations
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

        # initialise rabbit (best solution) and other variables
        rabbit_location = np.zeros(self.dimension)
        rabbit_energy = float("inf")
        convergence_curve = np.zeros(self.max_iterations)
        fitness = self.objective_function(Hawks)

        return Hawks, rabbit_location, rabbit_energy, convergence_curve, fitness

    def optimise(self) -> Solution:
        """

        :return:
        """

        # --- Initialisation ---
        Hawks, rabbit_location, rabbit_energy, convergence_curve, fitness = self._initialise()
        s = Solution()

        print(f"HHO is now tackling \"{self.objective_function.__name__}\"")

        timerStart = time.time()
        s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

        for t in range(self.max_iterations):

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

            # --- Core Algorithm ---
            for i in range(0, self.search_agents_num):

                E0 = 2 * random.random() - 1
                escaping_energy = E1 * E0

                # exploration
                if abs(escaping_energy) >= 1:
                    Hawks = exploration(i, self.search_agents_num, Hawks, rabbit_location, self.lower_bound,
                                        self.upper_bound)
                # exploitation
                else:
                    Hawks = exploitation(i, escaping_energy, Hawks, rabbit_location, self.objective_function,
                                         fitness, self.dimension, self.lower_bound, self.upper_bound)

            # store convergence data
            convergence_curve[t] = rabbit_energy
            if t % 1 == 0:
                print(f"At iteration {str(t)} the best fitness is {str(rabbit_energy)}")

        # --- Optimisation Complete ---
        timerEnd = time.time()
        s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
        s.executionTime = timerEnd - timerStart
        s.convergenceCurve = convergence_curve
        s.optimizer = "HHO"
        s.objname = self.objective_function.__name__
        s.best = rabbit_location
        s.bestIndividual = rabbit_location

        return s
