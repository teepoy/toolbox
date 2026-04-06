# Python SDK

The Python SDK is for workflows where CLI commands are too rigid and you want direct control over dataset creation or mutation.

## Main entry points

```python
import dman

builder = dman.create_dataset("demo")
dataset = dman.load_dataset("demo")
updater = dman.update_dataset("demo")
```

## Common workflows

### Create a dataset

```python
builder = dman.create_dataset("pets")
builder.add_sample("scene_001")
builder.add_asset("scene_001", "image", "/tmp/cat.jpg")
builder.add_annotation("scene_001", "cat", bbox=[10.0, 20.0, 30.0, 40.0])
builder.set_category("cat", supercategory="animal")
dataset = builder.build()
```

### Load a dataset

```python
dataset = dman.load_dataset("pets")
print(len(dataset))
print(dataset[0])
```

### Update a dataset

```python
updater = dman.update_dataset("pets")
updater.add_sample("scene_002")
updater.add_asset("scene_002", "image", "/tmp/dog.jpg")
updater.apply()
```

## Why use the SDK

- build datasets from scripts or notebooks
- integrate preprocessing pipelines directly
- create datasets from nonstandard sources before exporting them
- mix programmatic workflows with the CLI catalog
