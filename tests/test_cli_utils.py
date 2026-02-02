import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from src.utils.cli_utils import parse_file_indexes

directory = "datasets/1.openloop"

csv_files = [
    f
    for f in os.listdir(directory)
    if f.endswith(".csv") and os.path.isfile(os.path.join(directory, f))
]
no_of_files = len(csv_files)


@pytest.mark.parametrize(
    "raw_input,expected",
    [
        ([], list(range(no_of_files))),
        (["3"], [3]),
        (["100"], list(range(no_of_files))),
        (["0-2"], [0, 1, 2]),
        (["0   - 2"], [0, 1, 2]),
        (["0-99"], list(range(no_of_files))),
        (["1", "3"], [1, 3]),
        (["10-2"], list(range(no_of_files))),
        (["a", "b"], list(range(no_of_files))),
        (["-1", "100"], list(range(no_of_files))),
    ],
)
def test_parse_file_indexes(raw_input, expected):
    result = parse_file_indexes(raw_input, no_of_files)
    assert result == expected
