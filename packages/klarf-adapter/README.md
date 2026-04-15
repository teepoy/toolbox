## klarf-adapter

`klarf-adapter` is a small Python library and CLI for reading and generating KLARF wafer defect files.

### Install with uv

```bash
uv sync --dev
```

### Library usage

```python
from klarf_adapter import load, dumps

document = load("sample.klarf")
print(document.to_dict())
print(dumps(document))
```

The parser models KLARF as an ordered sequence of records:

- `KlarfStatement` for standard `Keyword value...;` statements
- `KlarfTable` for `ClassLookup` and `SampleTestPlan`
- `KlarfDefectRecordSpec` for defect field definitions
- `KlarfDefectList` for defect entries

### CLI

Read a KLARF file into YAML:

```bash
uv run klarf-adapter read sample.klarf
```

Generate KLARF from YAML or JSON:

```bash
uv run klarf-adapter write sample.yaml -o sample.klarf
```

### Test

```bash
uv run pytest
```
