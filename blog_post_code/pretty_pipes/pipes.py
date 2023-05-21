from typing import List
import pandas as pd


def convert_label_to_snake_case(dataframe, columns: List[str]):

    for column in columns:
        dataframe[column] = (
            dataframe[column]
            .str.strip()
            .str.lower()
            .str.replace(r"\W+", "_", regex=True)
        )

    return dataframe


def convert_to_snake_case(dataframe: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Convert values in string columns in a dataframe to snake case, handling extraneous spacing and capitalization."""
    assert set(columns).issubset(
        set(dataframe.columns)
    ), "At least one column is missing."

    assert all(
        pd.api.types.is_string_dtype(dataframe[column]) for column in columns
    ), "All columns must be of string dtype."

    return dataframe.assign(
        **{
            column: (
                dataframe[column]
                .str.strip()
                .str.lower()
                .str.replace(r"\W+", "_", regex=True)
            )
            for column in columns
        }
    )


if __name__ == "__main__":

    raw = pd.DataFrame(
        {
            "id": ["uuid1", "uuid2", "uuid3", "uuid4"],
            "label": ["large hat", "Small_hat", "Large hat", "small Hat "],
            "size": [200, 100, 350, 500],
        }
    )
    print("Run as a standalone...")
    print(convert_to_snake_case(raw, ["label"]))
    print("Run within a pipe...")
    print(raw.pipe(convert_to_snake_case, ["label"]))
