import torch
import torch.nn as nn

from ..args_val import is_df, is_dict, is_valid_keys


class BaseNetwork(nn.Module):

    def __init__(self, tabular, model_keys, visualise) -> None:
        super().__init__()
        self.tabular = tabular
        self.model_keys = model_keys
        self.visualise = visualise
        self.model = self._build_model(*self._get_params(tabular, model_keys))

        print(self) if visualise else None

    def forward(self, x: torch.Tensor) -> nn.Module:
        return self.model(x)

    def _build_model(self, *args, **kwargs):
        raise NotImplementedError("Define how model is built here")

    def _create_layers(self, *args, **kwargs):
        raise NotImplementedError("Define layer structure here")

    def _get_params(self, tabular, model_keys) -> list:
        """
        Destructures a tabular input.
        :param keys: A tuple containing keys for specific use case.
        :param tabular: A dict or pd.DataFrame input.
        :return: A tuple containing values relevant to the `keys` list.
        """
        is_valid_keys(tabular, model_keys)

        return (is_dict(tabular, model_keys)
                if isinstance(tabular, dict) else is_df(tabular, model_keys))

    def optimise(self):
        pass
