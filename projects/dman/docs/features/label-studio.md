# Label Studio Integration

`dman` can import from and export to Label Studio, so annotation work can move between a local dataset catalog and a labeling project.

## Import from Label Studio

```bash
dman-cli label-studio import \
  http://localhost:8080 \
  your-api-key \
  1 \
  my-labeled-dataset
```

## Export to Label Studio

```bash
dman-cli label-studio export \
  http://localhost:8080 \
  your-api-key \
  1 \
  my-labeled-dataset \
  --port 9090
```

## What happens on export

`dman` generates image URLs that Label Studio can open. Those URLs point at the local dman image-serving route:

```text
http://localhost:<port>/images/<dataset_id>/<filename>
```

That means the dataset images must be reachable from the machine running Label Studio.

## Good fit

Use this when:

- you annotate in Label Studio but want a local dataset catalog
- you want to export a dataset from dman for more labeling work
- you want to bridge between built-in/custom formats and annotation workflows
