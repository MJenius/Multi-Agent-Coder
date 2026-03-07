"""
Main entry point for the sample application.
"""

from src.utils import calculate_total


class Item:
    def __init__(self, name: str, price: float):
        self.name = name
        self.price = price


def main():
    items = [
        Item("Widget", 9.99),
        Item("Gadget", 24.99),
        Item("Doohickey", 4.50),
    ]

    total = calculate_total(items)
    print(f"Total: ${total:.2f}")

    # BUG: this crashes on an empty list
    empty_total = calculate_total([])
    print(f"Empty total: ${empty_total:.2f}")


if __name__ == "__main__":
    main()
