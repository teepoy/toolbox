# API Reference

dman-server exposes a read-only REST API for use by external ML services and Python training pipelines.

The full machine-readable spec is located at [`docs/api/openapi.yaml`](../../api/openapi.yaml) (OpenAPI 3.1).
A proto3 stub for future gRPC support is at [`docs/api/proto/dman.proto`](../../api/proto/dman.proto).

## Base URL

```
http://<host>:<port>
```

By default the server binds to `127.0.0.1:3000`.

## Endpoints

### Datasets

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/datasets` | List datasets (supports `?page` / `?per_page`) |
| `GET` | `/api/datasets/{name}` | Get a single dataset |
| `GET` | `/api/datasets/{name}/samples` | List samples (supports `?category` filter) |
| `GET` | `/api/datasets/{name}/samples/{sample_id}` | Get a single sample |
| `GET` | `/api/datasets/{name}/assets/{id}` | Get a single asset |
| `GET` | `/api/datasets/{name}/categories` | List categories |
| `GET` | `/api/datasets/{name}/stats` | Dataset statistics |
| `GET` | `/api/datasets/{name}/embeddings` | List embeddings (`?model=` optional) |
| `GET` | `/api/datasets/{name}/predictions` | List predictions (`?model_version=` optional) |
| `GET` | `/api/datasets/{name}/patches` | List patches |
| `GET` | `/api/datasets/{name}/export` | Download dataset as zip (`?format=yolo\|coco\|huggingface`) |

### Assets

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/assets/{id}/embeddings` | List embeddings for an asset (`?model=` optional) |
| `GET` | `/api/assets/{id}/patches` | List patches for an asset |
| `GET` | `/api/assets/{id}/annotations` | List annotations for an asset |

### Virtual Datasets

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/virtual-datasets` | List stored virtual datasets |
| `GET` | `/api/virtual-datasets/{name}` | Get a stored virtual dataset definition |
| `GET` | `/api/virtual-datasets/{name}/samples` | Evaluate and list samples for a stored VDS |
| `POST` | `/api/virtual-datasets/evaluate` | Ad-hoc DSL evaluation (body: `VirtualDatasetDef` JSON) |

### Media

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/images/{dataset_id}/{filename}` | Serve a raw asset file (ETag + Cache-Control headers) |

### Health

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check — returns `{"status":"ok"}` |
