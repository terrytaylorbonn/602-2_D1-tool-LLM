#codex_playground_v2.py
def add(a, b):
    return a + b


def divide(a, b):
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Both arguments must be numbers.")
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


def main():
    print("add(2,3) =", add(2, 3))
    print("divide(10,2) =", divide(10, 2))


if __name__ == "__main__":
    main()
