import warnings
from typing import Iterable, Sized, List

import pandas as pd
from torch import nn

keys = ("input_size", "output_size", "hidden_sizes", "activation_functions")


def keys_val(tabular: dict | pd.DataFrame, _keys: tuple) -> None:
    """
    :param tabular: A dict or pd.DataFrame.
    :param _keys: A tuple of required keys.
    """

    if not all(key in tabular for key in _keys):
        raise ValueError(f"Tabular data must contain keys {_keys}")


def get_params(tabular: dict | pd.DataFrame) -> tuple[int, int, Iterable[int], Iterable[nn.Module]]:
    """
    Destructures a tabular input.
    :param tabular: A dict or pd.DataFrame input.
    :return: A tuple containing (int, int, Sized, Iterable[nn.Module]).
    Unpack the values in the order of: input_size, output_size, hidden_sizes, activation_function
    """

    input_size = output_size = hidden_sizes = activation_functions = None

    keys_val(tabular, keys)

    if isinstance(tabular, dict):
        # unpack values from dictionary
        input_size, output_size = tabular["input_size"], tabular["output_size"]

        hidden_sizes, activation_functions = tabular["hidden_sizes"], tabular["activation_functions"]

    if isinstance(tabular, pd.DataFrame):
        # pandas.DataFrame uses `numpy.int64` by default
        # Call the `.item()` method to convert it back to `int`
        input_size, output_size = tabular["input_size"].iloc[0].item(), tabular["output_size"].iloc[0].item()

        # list-like objects don't need to be converted
        hidden_sizes, activation_functions = tabular["hidden_sizes"].iloc[0], tabular["activation_functions"].iloc[0]

    return input_size, output_size, hidden_sizes, activation_functions


def warn_hidden_layer(hidden_layer_num: int) -> None:
    """
    The variable `hidden_layer_num` should be inferred automatically.
    :param hidden_layer_num: An integer value indicating the number of hidden layers.

    Conventionally, a neural network is considered deep if it has at least two hidden layers.
    Others specify a four-hidden-layer neural network is considered deep.
    However, it is safe to assume a neural network is considered shallow if it has at most 2 hidden layers.

    References:
    1. Oladyshkin, S., Praditia, T., Kroeker, I., Mohammadi, F., Nowak, W., & Otte, S. (2023).
    The deep arbitrary polynomial chaos neural network or how Deep Artificial Neural Networks
    could benefit from data-driven homogeneous chaos theory. *Neural Networks*, 166, 85–104.
    https://doi.org/10.1016/j.neunet.2023.06.036

    2. Ross, A., Leroux, N., De Riz, A., Marković, D., Sanz-Hernández, D., Trastoy, J.,
    Bortolotti, P., Querlioz, D., Martins, L., Benetti, L., Claro, M. S., Anacleto, P.,
    Schulman, A., Taris, T., Bégueret, J.-B., Saïghi, S., Jenkins, A., Ferreira, R.,
    Vincent, A. F., … Grollier, J. (2023). Multilayer spintronic neural networks with
    radiofrequency connections. *Nature Nanotechnology*, 18(11), 1273–1280.
    https://doi.org/10.1038/s41565-023-01452-w
    """

    ###############################################################
    # if anyone managed to "push" the warning messages to the top #
    # please open a pr for it, thanks.                            #
    ###############################################################
    if hidden_layer_num >= 3:
        warnings.warn(
            "*** Deep Neural Network Detected ***\n"
            "The network is considered deep (>= 3 hidden layers).\n\n"
            "Consider using model templates from the 'deep' directory\n"
            "for better architecture options suited for deeper models.\n",
            UserWarning
        )
    else:
        warnings.warn(
            "*** Shallow Neural Network Detected ***\n"
            "A shallow neural network (<= 2 hidden layers) is being used.\n\n"
            "If you need more complexity or better performance, consider\n"
            "switching to a deeper architecture (>= 3 hidden layers).\n",
            UserWarning
        )


def sized_to_list(sized: Sized) -> List:
    """
    The `validate_args` function changed the type of `hidden_sizes` to Sized.
    Use this function to change it back to a List.
    :param sized: An iterable of sized items.
    :return: A list of sized items.
    """

    return [sized] if not isinstance(sized, list) else sized
