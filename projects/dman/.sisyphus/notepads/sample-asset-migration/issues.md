# Issues — sample-asset-migration

## [2026-04-05] Known issues to watch

- BBox JSON serialization inconsistency: format.rs uses `{x,y,width,height}`, builder.rs uses `{x,y,w,h}` — fix in T25
- `DatasetService::get_info()` returns `image_count` — must become `sample_count`/`asset_count`/`annotation_count`
- `DatasetOps` (829 lines) iterates images — full rewrite needed in T8
- Python plugin `PyImgData`/`PyAnnData` are bbox-only, single-image-centric — redesign in T16
