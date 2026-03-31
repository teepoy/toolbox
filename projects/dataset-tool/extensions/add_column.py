"""
Add Column Extension

Add computed columns based on existing columns.
"""


def transform(data, params):
    """
    Add a new computed column.

    Params:
        new_column: Name of the new column
        operation: Operation type (sum, multiply, concat, uppercase)
        source_columns: List of columns to use as sources
    """
    if not data:
        return data

    new_column = params.get("new_column")
    operation = params.get("operation", "sum")
    source_columns = params.get("source_columns", [])

    if not new_column:
        return data

    operations = {
        "sum": lambda row, cols: sum((row.get(c, 0) or 0) for c in cols),
        "multiply": lambda row, cols: __import__("functools").reduce(
            lambda a, b: a * b, [(row.get(c, 1) or 1) for c in cols], 1
        ),
        "concat": lambda row, cols: " ".join(str(row.get(c, "")) for c in cols),
        "uppercase": lambda row, cols: str(
            row.get(cols[0] if cols else "", "")
        ).upper(),
        "lowercase": lambda row, cols: str(
            row.get(cols[0] if cols else "", "")
        ).lower(),
        "length": lambda row, cols: len(str(row.get(cols[0] if cols else "", ""))),
    }

    compute = operations.get(operation, operations["sum"])

    result = []
    for row in data:
        new_row = row.copy()
        try:
            new_row[new_column] = compute(row, source_columns)
        except:
            new_row[new_column] = None
        result.append(new_row)

    return result
