# codex_playground_v2_refactor.py


def add(a, b):
    """Return the sum of two values."""
    return a + b


def divide(a, b):
    """Return the result of dividing a by b."""
    return a / b


def main():
    add_result = add(2, 3)
    divide_result = divide(10, 2)

    print("add(2,3) =", add_result)
    print("divide(10,2) =", divide_result)


if __name__ == "__main__":
    main()
