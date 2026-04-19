# Decisions — sample-asset-migration

## [2026-04-05] Key design decisions (confirmed by user)

1. **No backward compatibility** — clean break, no actual users yet
2. **Annotation dual-level**: `sample_id NOT NULL` + optional `asset_id`
3. **Python plugin contract**: Full redesign (not incremental)
4. **Format adapters**: Update directly (no compatibility shim)
5. **Asset type**: Rust `AssetType` enum serialized to TEXT column
6. **Storage paths**: Rename `images/` → `assets/`
7. **Single-image ergonomics**: Importers auto-create 1:1 Sample→Asset via helper
8. **Virtual dataset filters**: Filter on samples (assets come along)
9. **BBox JSON**: Standardize to `{x, y, width, height}` everywhere
10. **DB migration strategy**: Replace single existing M::up with new combined schema (clean break)
