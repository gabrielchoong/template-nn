import pandas as pd
import numpy as np
from template_nn.args_val import is_df


def test_is_df_returns_expected_values():
    df = pd.DataFrame([{"a": np.int32(1), "b": "value"}])
    assert is_df(df, ("a", "b")) == [1, "value"]
