import numpy as np
from pathlib import Path
from os import getcwd


def test_mask():
    arr = np.array([1, 2, 3, 4])
    print(arr[np.array(np.array([0, 1, 1, 0]).astype(bool))])


def test_path():
    print(Path(getcwd()).parents[0])


def test_isdigit():
    assert "1234".isdigit()
    assert "0123".isdigit()
    assert not "-1234".isdigit()
    assert not "0.345".isdigit()


def test_isnumeric():
    assert "1234".isnumeric()
    assert "0123".isnumeric()
    assert "-1234".isdecimal()
    assert "0.345".isdecimal()

def test_convert_to_float():
    float('g')


def test_convert_to_bool():
    print(np.ones(10, dtype=bool))