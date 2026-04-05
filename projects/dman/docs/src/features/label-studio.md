# Label Studio Integration

`dman` can import from and export to Label Studio, so annotation work can move between a local dataset catalog and a labeling project.

## Data model mapping

Each Label Studio task maps to one **Sample** containing one **Asset** (image type). This is a 1:1 mapping: one task = one sample + one image asset. Annotations on the task become dman annotations attached to that asset.

## Import from Label Studio

```bash
dman-cli label-studio import \
  http://localhost:8080 \
  your-api-key \
  1 \
  my-labeled-dataset
```

Each imported task becomes a sample. The task image becomes an image asset under that sample. All annotation results are stored as dman annotations linked to the asset.

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

`dman` generates asset URLs that Label Studio can open:

```text
http://localhost:<port>/images/<dataset_id>/<filename>
```

The static file route `/images/{dataset_id}/{filename}` serves image assets directly. The dataset assets must be reachable from the machine running Label Studio.

## Good fit

Use this when:

- you annotate in Label Studio but want a local dataset catalog
- you want to export a dataset from dman for more labeling work
- you want to bridge between built-in/custom formats and annotation workflows
