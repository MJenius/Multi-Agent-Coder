"""
Utility functions for the sample project.
"""


def calculate_total(items):
    """Calculate the total price from a list of items.

    Each item is expected to have a `.price` attribute.
    """
    return sum(item.price for item in items)


def format_receipt(items, total):
    """Format a receipt string for the given items."""
    lines = [f"  {item.name}: ${item.price:.2f}" for item in items]
    lines.append(f"  TOTAL: ${total:.2f}")
    return "\n".join(lines)
