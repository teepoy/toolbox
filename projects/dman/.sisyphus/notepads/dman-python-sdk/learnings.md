## [2026-04-06] Task: sdk-init — Codebase inventory

### Package layout
- pyproject.toml: `name = "dman"`, `python-source = "python"`, maturin-backed
- python/dman/__init__.py — pure Python, only `main()` CLI entrypoint today
- Rust .so compiled as `dman` (maturin uses filename not Rust fn name)
- PyO3 module fn is `fn dman_python(...)` in lib.rs — irrelevant to import name

### Fix for `import dman` → add re-exports to python/dman/__init__.py:
```python
from .dman import load_dataset, create_dataset, update_dataset
from .dman import DmanDataset, DmanDatasetBuilder, DmanDatasetUpdater
```
No Rust changes needed at all.

### Rust SDK public surface (from loader.rs + builder.rs)
DmanDataset methods:
- `__len__()` → int
- `__getitem__(idx)` → dict (id, name, metadata, assets: list[dict], annotations: list[dict])
- `samples()` → list[dict] (id, name, dataset_id, metadata, assets: list[dict])
- `get_sample(name)` → dict | None  (id, name, dataset_id, metadata, assets: list[dict])
- `annotations(sample_name, asset_name=None)` → list[dict]
- `images()` → list[str]
- `to_torch_dataset()`, `to_hf_dataset()`
- `.name` (getter), `.dataset_id` (getter)
- `sample_count()`, `asset_count()`

DmanDatasetBuilder methods:
- `add_sample(name, metadata=None)` → int (index)
- `add_asset(sample_name, asset_type, file_path, width=None, height=None, metadata=None)`
- `add_image(path, metadata=None)` → int
- `add_annotation(sample_name, category, bbox=None, segmentation=None, keypoints=None, metadata=None, asset_name=None)`
- `set_category(name, supercategory=None)`
- `build()` → DmanDataset

DmanDatasetUpdater methods:
- `add_sample(name, metadata=None)`
- `add_asset(sample_name, asset_type, file_path, width=None, height=None, metadata=None)`
- `add_annotation(sample_id, category, bbox=None, asset_id=None, metadata=None)`
- `remove_sample(sample_id)`
- `apply()`

### Internal dict shapes returned by Rust SDK

Sample dict keys: id(int), name(str), dataset_id(int), metadata(str — JSON or ""), assets(list[Asset dict])
Asset dict keys: id(int), sample_id(int), asset_type(str), file_name(str), file_path(str), width(int|None), height(int|None), metadata(str — JSON or "")
Annotation dict keys: id(int), sample_id(int), asset_id(int|None), category_id(int|None), bbox(str — JSON "{x,y,width,height}" or ""), segmentation(str — JSON or ""), keypoints(str — JSON or ""), metadata(str — JSON or "")

### BBox JSON format in DB
`{"x": ..., "y": ..., "width": ..., "height": ...}` (confirmed from tests)
