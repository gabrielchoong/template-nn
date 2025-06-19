import math
import random

import numpy as np

from template_nn._utils.levy import Levy


def exploration(i,
                search_agent_num,
                Hawks,
                rabbit_location,
                lower_bound,
                upper_bound
                ) -> np.ndarray:
    q = random.random()
    random_hawk_index = math.floor(
        search_agent_num * random.random()
    )

    Hawk_random = Hawks[random_hawk_index, :]

    if q < 0.5:

        Hawks[i, :] = Hawk_random - random.random() * \
                      abs(Hawk_random - 2 * random.random() * Hawks[i, :])

    else:

        Hawks[i, :] = (rabbit_location - Hawks.mean()) - \
                      random.random() * ((upper_bound - lower_bound) *
                                         random.random() + lower_bound)

    return Hawks


def exploitation(i,
                 escaping_energy,
                 Hawks,
                 rabbit_location,
                 objective_function,
                 fitness,
                 dimension,
                 lower_bound,
                 upper_bound,
                 ) -> np.ndarray:
    r = random.random()

    if r >= 0.5:

        if abs(escaping_energy) >= 0.5:
            jump_strength = 2 * (1 - random.random())
            Hawks[i, :] = (rabbit_location - Hawks[i, :]) - escaping_energy * abs(
                jump_strength * rabbit_location - Hawks[i, :]
            )

        else:
            Hawks[i, :] = rabbit_location - escaping_energy * abs(rabbit_location - Hawks[i, :])

    else:

        if abs(escaping_energy) >= 0.5:
            jump_strength = 2 * (1 - random.random())
            X1 = rabbit_location - escaping_energy * abs(
                jump_strength * rabbit_location - Hawks[i, :]
            )
            X1 = np.clip(X1, lower_bound, upper_bound)

            if objective_function(X1) < fitness:
                Hawks[i, :] = X1.copy()
            else:
                X2 = rabbit_location - escaping_energy * abs(
                    jump_strength * rabbit_location - Hawks[i, :]
                ) + np.multiply(
                    np.random.rand(dimension), Levy(dimension)
                )
                X2 = np.clip(X2, lower_bound, upper_bound)
                if objective_function(X2) < fitness:
                    Hawks[i, :] = X2.copy()

        else:
            jump_strength = 2 * (1 - random.random())
            X1 = rabbit_location - escaping_energy * abs(
                jump_strength * rabbit_location - Hawks.mean(0)
            )
            X1 = np.clip(X1, lower_bound, upper_bound)

            if objective_function(X1) < fitness:
                Hawks[i, :] = X1.copy()
            else:
                X2 = rabbit_location - escaping_energy * abs(
                    jump_strength * rabbit_location - Hawks.mean(0)
                ) + np.multiply(
                    np.random.rand(dimension), Levy(dimension)
                )
                X2 = np.clip(X2, lower_bound, upper_bound)

                if objective_function(X2) < fitness:
                    Hawks[i, :] = X2.copy()

    return Hawks
