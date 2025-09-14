from typing import Iterable


def is_valid_keys(
    tabular: dict[str, int | list[int] | list[str]], keys: tuple[str]
) -> None:
    if not all(key in tabular for key in keys):
        raise ValueError(f"Tabular data must contain keys {keys}")


def is_dict(tabular: dict, keys: tuple) -> list:
    params = []
    for key in keys:
        params.append(tabular[key])
    return params


def is_positive_int(number: int) -> None:
    if not isinstance(number, int) or number <= 0:
        raise ValueError(f"Expected positive integer, got {number} instead.")


def is_iterable(numbers: Iterable[int] | list[tuple[int, int]]) -> None:
    if not isinstance(numbers, Iterable):
        raise ValueError(f"Expected iterable structure, got {numbers} instead.")


def has_activation_functions(activation_functions: Iterable[str]) -> None:
    if not activation_functions:
        raise ValueError("No activation functions were provided.")


def activation_functions_check(
    activation_functions: list[str], hidden_sizes: list[int]
) -> None:
    if len(activation_functions) != len(hidden_sizes):
        raise ValueError(
            "Number of activation functions does not match number of hidden node counts."
            + f"Expected {activation_functions} of hidden nodes, but got {hidden_sizes} instead."
        )
