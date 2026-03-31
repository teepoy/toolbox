"""
Sort Extension

Sort rows by specified columns.
"""


def transform(data, params):
    """
    Sort rows by columns.

    Params:
        columns: List of columns to sort by (in order)
        directions: List of directions - 'asc' or 'desc' (default: 'asc')
    """
    if not data:
        return data

    columns = params.get("columns", [])
    directions = params.get("directions", ["asc"] * len(columns))

    if not columns:
        columns = [list(data[0].keys())[0]] if data else []

    while len(directions) < len(columns):
        directions.append("asc")

    def sort_key(row):
        return tuple(
            (row.get(col) or "")
            if directions[i] == "asc"
            else -1 * (row.get(col) or 0)
            if isinstance(row.get(col), (int, float))
            else ""
            for i, col in enumerate(columns)
        )

    return sorted(data, key=sort_key)
