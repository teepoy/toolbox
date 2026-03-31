"""
Dataset Tool Extension System

This directory contains Python extensions that can transform and process datasets.

EXTENSION REGISTRATION:
Each extension should define a 'transform(data, params)' function that:
- Takes 'data': List of dictionaries representing rows
- Takes 'params': Dictionary of parameters from the API call
- Returns: Modified list of dictionaries

SAMPLE EXTENSIONS:

1. normalize.py - Normalize numeric columns to 0-1 range
2. filter_rows.py - Filter rows based on conditions
3. add_column.py - Add computed columns
4. deduplicate.py - Remove duplicate rows
"""


def get_extension_metadata():
    """Return metadata about this extension."""
    return {
        "name": "template",
        "version": "1.0",
        "description": "Template extension showing the extension API",
        "parameters": {"sample_param": "Description of parameter"},
    }


def transform(data, params):
    """
    Transform the dataset.

    Args:
        data: List of row dictionaries
        params: Parameters from API request body

    Returns:
        List of row dictionaries (modified)
    """
    # Your transformation logic here
    return data
