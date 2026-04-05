# Terminal UI

The TUI is built into the CLI as a subcommand:

```bash
dman-cli tui
```

## What it gives you

- dataset list view
- dataset detail view
- tabbed views for info, images, categories, and schema
- keyboard navigation for quick catalog browsing

## Why it exists

The TUI is useful when the CLI is too narrow for exploration but opening the web/API stack would be overkill.

## Typical use

1. initialize the catalog
2. add or import datasets
3. open `dman-cli tui`
4. browse datasets interactively

## Current scope

The TUI is focused on browsing and inspection. Dataset mutation still happens primarily through the CLI and SDK.
