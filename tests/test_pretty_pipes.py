from typing import Callable, List
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from blog_post_code.pretty_pipes.pipes import (
    convert_to_snake_case,
    convert_label_to_snake_case,
)


@pytest.fixture()
def data():
    return pd.DataFrame(
        {
            "id": ["uuid1", "uuid2", "uuid3", "uuid4"],
            "label": ["large hat", "Small_hat", "Large hat", "small Hat "],
            "size": [200, 100, 350, 500],
        }
    )


@pytest.mark.parametrize(
    ("converter", "columns"),
    [
        (convert_label_to_snake_case, ["label"]), 
        (convert_to_snake_case, ["label"])
    ],
)
def test_snake_case_conversion(
    data, converter: Callable[[pd.DataFrame, List[str]], pd.DataFrame], columns: List[str]
):
    """Test that whitespace and case variations are handled."""

    expected = pd.DataFrame(
        {
            "id": {0: "uuid1", 1: "uuid2", 2: "uuid3", 3: "uuid4"},
            "label": {0: "large_hat", 1: "small_hat", 2: "large_hat", 3: "small_hat"},
            "size": {0: 200, 1: 100, 2: 350, 3: 500},
        }
    )

    converted = converter(data, columns)

    assert_frame_equal(converted, expected)
