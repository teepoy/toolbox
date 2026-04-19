## [2026-04-05] Plan initialized

### Execution order (from plan)
1. Phase 1: Write dedicated docs page (lifecycle + plugin placement + discovery)
2. Phase 2: Document Python provider contract precisely (detect, import_dataset, export_dataset)
3. Phase 3: Add parquet multi-image example (materialization pattern)
4. Phase 4: Harden runtime validation (missing keys, bad bbox, unknown image refs)
5. Phase 5: Add tests (discovery, importer contract, CLI integration, export)
6. Phase 6: Clarify extension boundaries (future MCP/gRPC layering — docs only)

### Scope constraints
- Do NOT redesign the plugin system
- Do NOT add MCP/gRPC transport unless implementation exists
- Document and harden the CURRENT path first
