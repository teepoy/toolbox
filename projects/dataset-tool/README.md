# Dataset Tool

A web-based dataset view and modification tool with a Rust backend and Python extension system.

## Architecture

```
dataset-tool/
├── backend/          # Rust Actix-web API server
│   ├── src/
│   │   └── main.rs   # API endpoints, SQLite database
│   └── Cargo.toml
├── extensions/       # Python transformation extensions
│   ├── normalize.py
│   ├── filter_rows.py
│   ├── add_column.py
│   ├── deduplicate.py
│   └── sort.py
└── frontend/         # Web UI (HTML/CSS/JS)
    └── index.html
```

## Features

- **Dataset Management**: Create, view, edit, and delete datasets
- **Inline Editing**: Modify cell values directly in the table
- **Python Extensions**: Apply transformations via pluggable Python scripts
- **Dark Theme UI**: Modern, responsive interface

## Running the Tool

```bash
# Start the backend (runs on http://localhost:8080)
cd backend
cargo run

# Access the frontend
open http://localhost:8080
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/datasets` | List all datasets |
| POST | `/api/datasets` | Create dataset |
| GET | `/api/datasets/{id}` | Get dataset with data |
| PUT | `/api/datasets/{id}/rows` | Update rows |
| DELETE | `/api/datasets/{id}` | Delete dataset |
| GET | `/api/extensions` | List available extensions |
| POST | `/api/datasets/{id}/extensions/{name}` | Run extension |

## Python Extensions

Place `.py` files in `~/.local/share/dataset-tool/extensions/`.

Each extension must define a `transform(data, params)` function:

```python
def transform(data, params):
    """
    Args:
        data: List of row dictionaries
        params: Parameters from API request
    Returns:
        Modified list of row dictionaries
    """
    return data
```

### Available Extensions

- **normalize**: Normalize numeric columns to 0-1 range
- **filter_rows**: Filter rows based on conditions
- **add_column**: Add computed columns
- **deduplicate**: Remove duplicate rows
- **sort**: Sort by specified columns

## Tech Stack

- **Backend**: Rust + Actix-web + SQLite
- **Extensions**: Python 3
- **Frontend**: Vanilla HTML/CSS/JS
