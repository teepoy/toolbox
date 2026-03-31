"""
Filter Rows Extension

Filter rows based on column conditions.
"""


def transform(data, params):
    """
    Filter rows based on conditions.

    Params:
        column: Column name to filter on
        operator: Comparison operator (eq, ne, gt, lt, gte, lte, contains)
        value: Value to compare against
    """
    if not data:
        return data

    column = params.get("column")
    operator = params.get("operator", "eq")
    value = params.get("value")

    if not column:
        return data

    operators = {
        "eq": lambda a, b: a == b,
        "ne": lambda a, b: a != b,
        "gt": lambda a, b: (a or 0) > (b or 0),
        "lt": lambda a, b: (a or 0) < (b or 0),
        "gte": lambda a, b: (a or 0) >= (b or 0),
        "lte": lambda a, b: (a or 0) <= (b or 0),
        "contains": lambda a, b: str(b).lower() in str(a).lower() if a else False,
    }

    compare = operators.get(operator, operators["eq"])

    return [row for row in data if compare(row.get(column), value)]
