# Python SDK

The Python SDK is for workflows where CLI commands are too rigid and you want direct control over dataset creation or mutation.

## Main entry points

```python
import dman_python

builder = dman_python.create_dataset("demo")
dataset = dman_python.load_dataset("demo")
updater = dman_python.update_dataset("demo")
```

## Common workflows

### Create a dataset

```python
builder = dman_python.create_dataset("pets")
image_id = builder.add_image("/tmp/cat.jpg")
builder.add_annotation(image_id, "cat", [10.0, 20.0, 30.0, 40.0])
builder.set_category("cat", supercategory="animal")
dataset = builder.build()
```

### Load a dataset

```python
dataset = dman_python.load_dataset("pets")
print(len(dataset))
print(dataset[0])
```

### Update a dataset

```python
updater = dman_python.update_dataset("pets")
updater.add_image("/tmp/dog.jpg")
updater.apply()
```

## Why use the SDK

- build datasets from scripts or notebooks
- integrate preprocessing pipelines directly
- create datasets from nonstandard sources before exporting them
- mix programmatic workflows with the CLI catalog
