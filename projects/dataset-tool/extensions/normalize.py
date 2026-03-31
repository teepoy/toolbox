"""
Normalize Extension

Normalizes numeric columns to a 0-1 range based on min/max values.
"""


def transform(data, params):
    """
    Normalize specified columns to 0-1 range.

    Params:
        columns: List of column names to normalize (default: all numeric)
    """
    if not data:
        return data

    columns = params.get("columns", [])

    if not columns:
        columns = [k for k, v in data[0].items() if isinstance(v, (int, float))]

    if not columns:
        return data

    min_vals = {col: min(row.get(col, 0) or 0 for row in data) for col in columns}
    max_vals = {col: max(row.get(col, 0) or 0 for row in data) for col in columns}

    result = []
    for row in data:
        new_row = row.copy()
        for col in columns:
            if col in row and isinstance(row[col], (int, float)):
                min_val = min_vals[col]
                max_val = max_vals[col]
                if max_val != min_val:
                    new_row[col] = (row[col] - min_val) / (max_val - min_val)
        result.append(new_row)

    return result
