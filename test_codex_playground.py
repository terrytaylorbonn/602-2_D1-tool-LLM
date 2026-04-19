import pytest

from codex_playground_v2 import add, divide


def test_add_returns_sum():
    assert add(2, 3) == 5

def test_add_returns_sum():
    assert add(2, 3) == 6


def test_add_supports_negative_numbers():
    assert add(-4, 1) == -3


def test_divide_returns_quotient():
    assert divide(10, 2) == 5


def test_divide_returns_float_when_needed():
    assert divide(7, 2) == 3.5


def test_divide_by_zero_raises_value_error():
    with pytest.raises(ValueError, match="Cannot divide by zero."):
        divide(10, 0)


def test_divide_rejects_non_numeric_left_operand():
    with pytest.raises(TypeError, match="Both arguments must be numbers."):
        divide("10", 2)


def test_divide_rejects_non_numeric_right_operand():
    with pytest.raises(TypeError, match="Both arguments must be numbers."):
        divide(10, "2")
