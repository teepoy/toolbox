"""
Deduplicate Extension

Remove duplicate rows based on specified columns.
"""


def transform(data, params):
    """
    Remove duplicate rows.

    Params:
        columns: List of columns to check for duplicates (default: all)
        keep: Which to keep - 'first' or 'last' (default: 'first')
    """
    if not data:
        return data

    columns = params.get("columns")
    keep = params.get("keep", "first")

    if not columns:
        columns = list(data[0].keys()) if data else []

    seen = {}
    result = []

    rows = data if keep == "first" else reversed(data)

    for row in rows:
        key = tuple(row.get(c) for c in columns)
        if keep == "first":
            if key not in seen:
                seen[key] = True
                result.append(row)
        else:
            if key not in seen:
                seen[key] = row
            else:
                seen[key] = row

    return result if keep == "first" else list(reversed(list(seen.values())))
