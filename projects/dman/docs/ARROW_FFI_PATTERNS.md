# Arrow Zero-Copy FFI: Production OSS Patterns (Rust→Python via PyO3)

## Summary Table

| Project | Pattern | FFI Method | Arrow Crate | PyO3 Feature | Key Type |
|---------|---------|-----------|-------------|-------------|----------|
| **datafusion-python** | `RecordBatch` → `PyCapsule` stream | `FFI_ArrowArrayStream` (C Data Interface) | `arrow-58` | `pyarrow` | `FFI_ArrowArrayStream::new()` |
| **delta-rs** | Stream wrapper → `PyRecordBatchReader` | `pyo3_arrow::PyRecordBatchReader` | `arrow-57` | `arrow` | `pyo3-arrow-0.14.0` |
| **lance** | `RecordBatchReader` → PyArrow wrapped | `PyArrowType<RecordBatchReader>` | `arrow-57` | `pyarrow` | `IntoPyArrow` trait |
| **arro3** | Minimal library for Arrow PyCapsule | `Arrow PyCapsule Interface v2` | `arrow-58` | `arrow` | Zero-copy bridge |

---

## Pattern 1: Arrow C Data Interface Stream (datafusion-python)

**Use Case**: Streaming large datasets efficiently without materialization

### Rust Implementation

**File**: [`crates/core/src/dataframe.rs`](https://github.com/apache/datafusion-python/blob/ff15648c5dca6b41d3f6146c6c36c97e605f8561/crates/core/src/dataframe.rs#L1107-L1146)

```rust
use std::ffi::{CStr, CString};
use std::ptr::NonNull;
use arrow::ffi::FFI_ArrowSchema;
use arrow::ffi_stream::FFI_ArrowArrayStream;
use arrow::pyarrow::ToPyArrow;
use pyo3::types::PyCapsule;
use pyo3::prelude::*;

// The `__arrow_c_stream__` method implements Arrow PyCapsule Interface v2
#[pyo3(signature = (requested_schema=None))]
fn __arrow_c_stream__<'py>(
    &'py self,
    py: Python<'py>,
    requested_schema: Option<Bound<'py, PyCapsule>>,
) -> PyDataFusionResult<Bound<'py, PyCapsule>> {
    // 1. Execute DataFusion query to get partitioned streams
    let df = self.df.as_ref().clone();
    let streams = spawn_future(py, async move {
        df.execute_stream_partitioned().await
    })?;

    // 2. Get schema and handle requested schema negotiation
    let mut schema: Schema = self.df.schema().to_owned().as_arrow().clone();
    let mut projection: Option<SchemaRef> = None;

    if let Some(schema_capsule) = requested_schema {
        // Parse requested schema from PyCapsule
        let data: NonNull<FFI_ArrowSchema> = schema_capsule
            .pointer_checked(Some(c"arrow_schema"))?
            .cast();
        let schema_ptr = unsafe { data.as_ref() };
        let desired_schema = Schema::try_from(schema_ptr)?;

        schema = project_schema(schema, desired_schema)?;
        projection = Some(Arc::new(schema.clone()));
    }

    let schema_ref = Arc::new(schema.clone());

    // 3. Wrap in RecordBatchReader for Arrow C stream interface
    let reader = PartitionedDataFrameStreamReader {
        streams,
        schema: schema_ref,
        projection,
        current: 0,
    };
    let reader: Box<dyn RecordBatchReader + Send> = Box::new(reader);

    // 4. Convert to Arrow stream and wrap in PyCapsule
    let stream = FFI_ArrowArrayStream::new(reader);
    let name = CString::new(ARROW_ARRAY_STREAM_NAME.to_bytes()).unwrap();
    let capsule = PyCapsule::new(py, stream, Some(name))?;
    Ok(capsule)
}
```

### Key Traits/Types

- **`FFI_ArrowArrayStream`**: Arrow-rs FFI structure that implements C Data Interface
- **`RecordBatchReader`**: Trait that yields `RecordBatch` items lazily
- **`PyCapsule`**: PyO3's wrapper for opaque C pointers (zero-copy container)

### Python-Side Consumption

```python
import pyarrow as pa
import datafusion

df = datafusion.sql("SELECT * FROM table")

# Method 1: Direct PyCapsule consumption
table = pa.RecordBatchReader._import_from_c_capsule(
    df.__arrow_c_stream__()
)
batches = table.to_batches()

# Method 2: Automatic via table_from_pydict/pylist (modern PyArrow)
table = pa.table(df)
```

### Cargo Dependencies

```toml
arrow = { version = "58" }
arrow-array = { version = "58" }
datafusion-python-util = { ... }
pyo3 = { version = "0.20+" }
```

---

## Pattern 2: pyo3-arrow Wrapper (delta-rs)

**Use Case**: Lightweight, modular FFI with support for multiple Python Arrow libraries

### Rust Implementation

**File**: [`python/src/query.rs`](https://github.com/delta-io/delta-rs/blob/0bb2d6bc7d058c10870c4e275827639b172e813f/python/src/query.rs#L61-L72)

```rust
use pyo3_arrow::PyRecordBatchReader;
use arrow_cast::cast;
use arrow_schema::{DataType, Schema, SchemaRef};

/// Execute SQL query and return a PyRecordBatchReader
pub fn execute(&self, py: Python, sql: &str) -> PyResult<PyRecordBatchReader> {
    let stream = py.detach(|| {
        rt().block_on(async {
            let df = self.ctx.sql(sql).await?;
            df.execute_stream().await
        })
        .map_err(PythonError::from)
    })?;

    // Convert async stream to sync RecordBatchReader
    let stream = convert_stream_to_reader(stream);
    // Wrap in pyo3_arrow type for automatic FFI export
    Ok(PyRecordBatchReader::new(stream))
}
```

**File**: [`python/src/reader.rs`](https://github.com/delta-io/delta-rs/blob/0bb2d6bc7d058c10870c4e275827639b172e813f/python/src/reader.rs#L97-L107)

```rust
/// Converts async SendableRecordBatchStream to sync RecordBatchReader
pub(crate) fn convert_stream_to_reader(
    stream: SendableRecordBatchStream,
) -> Box<dyn RecordBatchReader + Send> {
    let (schema, cast_targets, needs_cast) = view_type_contract(&stream.schema());
    Box::new(StreamToReaderAdapter {
        schema,
        cast_targets,
        needs_cast,
        stream,
    })
}
```

### Key Aspects

- **`pyo3_arrow::PyRecordBatchReader`**: Automatically implements `__arrow_c_stream__`
- **No manual PyCapsule handling**: pyo3-arrow handles all FFI protocol details
- **View type normalization**: Handles Utf8View ↔ Utf8 conversions for compatibility

### Python-Side Consumption

```python
import pyarrow as pa
from deltalake import PyQueryBuilder

qb = PyQueryBuilder()
qb.register("my_table", delta_table)

# Returns PyRecordBatchReader directly—PyArrow understands it
reader = qb.execute(
    "SELECT * FROM my_table WHERE year = 2024"
)

# Zero-copy conversion
table = pa.RecordBatchReader._import_from_c_capsule(
    reader.__arrow_c_stream__()
)
```

### Cargo Dependencies

```toml
pyo3-arrow = { version = "0.14.0", default-features = false }
arrow-schema = { version = "58" }
arrow-cast = { version = "58" }
pyo3 = { version = "0.28" }
```

---

## Pattern 3: PyArrowType Wrapper (lance)

**Use Case**: Direct wrapping of Rust types to PyArrow without intermediate abstraction

### Rust Implementation

**File**: [`python/src/scanner.rs`](https://github.com/lancedb/lance/blob/d630106da5a238b3adfb8c5dea3b3921f3519945/python/src/scanner.rs#L147-L158)

```rust
use arrow::pyarrow::*;
use arrow_array::RecordBatchReader;
use pyo3::prelude::*;

#[pyclass(name = "_Scanner", module = "_lib")]
pub struct Scanner {
    scanner: Arc<LanceScanner>,
}

#[pymethods]
impl Scanner {
    /// Returns PyArrow-compatible reader using Arrow PyCapsule Interface
    fn to_pyarrow(
        self_: PyRef<'_, Self>,
    ) -> PyResult<PyArrowType<Box<dyn RecordBatchReader + Send>>> {
        let scanner = self_.scanner.clone();
        let reader = rt()
            .spawn(Some(self_.py()), async move {
                LanceReader::try_new(scanner).await
            })?
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        Ok(PyArrowType(Box::new(reader)))
    }
}
```

**File**: [`python/src/dataset.rs`](https://github.com/lancedb/lance/blob/d630106da5a238b3adfb8c5dea3b3921f3519945/python/src/dataset.rs#L2829-L2832)

```rust
use arrow::pyarrow::IntoPyArrow;
use arrow_array::RecordBatchReader;

fn to_stream_reader<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    // ... build and execute query ...

    let dataset_stream = DatasetRecordBatchStream::new(stream);
    let reader: Box<dyn RecordBatchReader + Send> =
        Box::new(LanceReader::from_stream(dataset_stream));

    // Automatic FFI export via IntoPyArrow trait
    reader.into_pyarrow(py)
}
```

### Key Traits/Types

- **`PyArrowType<T>`**: Generic wrapper that auto-implements `__arrow_c_stream__` for `RecordBatchReader`
- **`ToPyArrow` trait**: Converts Arrow types to PyArrow objects
- **`IntoPyArrow` trait**: Consumes Rust type and exports to PyArrow

### Python-Side Consumption

```python
import pyarrow as pa
import lance

ds = lance.dataset("path/to/data.lance")
scanner = ds.scanner()

# to_pyarrow() returns a RecordBatchReader
reader = scanner.to_pyarrow()

# Consume as table
table = pa.RecordBatchReader._import_from_c_capsule(
    reader.__arrow_c_stream__()
)

# Or directly as pandas
df = table.to_pandas()
```

### Cargo Dependencies

```toml
arrow = { version = "57.0.0", features = ["pyarrow"] }
arrow-array = "57.0.0"
arrow-schema = "57.0.0"
pyo3 = { version = "0.26" }
```

---

## Pattern 4: Direct RecordBatch Materialization (datafusion-python collect)

**Use Case**: Small result sets where full materialization is acceptable

### Rust Implementation

**File**: [`crates/core/src/dataframe.rs`](https://github.com/apache/datafusion-python/blob/ff15648c5dca6b41d3f6146c6c36c97e605f8561/crates/core/src/dataframe.rs#L647-L653)

```rust
use arrow::pyarrow::ToPyArrow;
use pyo3::prelude::*;

/// Eagerly collect all result batches and convert to PyArrow objects
fn collect<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let batches = wait_for_future(py, self.df.as_ref().clone().collect())?
        .map_err(PyDataFusionError::from)?;

    // Zero-copy conversion of each batch via Arrow C Data Interface
    batches.into_iter()
        .map(|rb| rb.to_pyarrow(py))
        .collect()
}
```

### Key Points

- **`to_pyarrow(py)` method**: Built into `arrow-rs` `RecordBatch` type
- **Each batch converted independently**: Enables parallel processing
- **No streaming overhead**: Simple for small datasets

### Python-Side Consumption

```python
import pyarrow as pa
import datafusion

df = datafusion.sql("SELECT * FROM tiny_table")

# collect() returns list of pyarrow.RecordBatch objects
batches = df.collect()

# Combine into table
table = pa.Table.from_batches(batches)
```

---

## Pattern 5: Array-Level FFI (pyarrow_util.rs)

**Use Case**: Individual scalar/array conversions for UDFs and expressions

### Rust Implementation

**File**: [`crates/core/src/pyarrow_util.rs`](https://github.com/apache/datafusion-python/blob/ff15648c5dca6b41d3f6146c6c36c97e605f8561/crates/core/src/pyarrow_util.rs#L56-L163)

```rust
use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use pyo3::{Bound, PyAny, PyResult, Python};

/// Extract Arrow array from ANY Python object supporting Arrow PyCapsule Interface
fn pyobj_extract_scalar_via_capsule(
    value: &Bound<'_, PyAny>,
    as_list_array: bool,
) -> PyResult<PyScalarValue> {
    // Zero-copy extraction via Arrow PyCapsule
    let array_data = ArrayData::from_pyarrow_bound(value)?;
    let array = make_array(array_data);

    array_to_scalar_value(array, as_list_array)
}

impl FromPyArrow for PyScalarValue {
    fn from_pyarrow_bound(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = value.py();

        // Support multiple Python Arrow libraries via PyCapsule Interface
        if let Ok(pa) = py.import("pyarrow") {
            let scalar_type = pa.getattr("Scalar")?;
            if value.is_instance(&scalar_type)? {
                // Extract PyArrow scalar
                let factory = py.import("pyarrow")?.getattr("array")?;
                let args = PyList::new(py, [value])?;
                let array = factory.call1((args, typ))?;
                return pyobj_extract_scalar_via_capsule(&array, false);
            }
        }

        // Also support nanoarrow and arro3 via same interface
        if let Ok(arro3) = py.import("arro3") {
            let scalar_type = arro3.getattr("core")?.getattr("Scalar")?;
            if value.is_instance(&scalar_type)? {
                return pyobj_extract_scalar_via_capsule(value, false);
            }
        }

        // Fallback: check __arrow_c_array__ protocol
        if value.hasattr("__arrow_c_array__")? {
            let array_data = ArrayData::from_pyarrow_bound(value)?;
            let array = make_array(array_data);
            return array_to_scalar_value(array, true);
        }

        Err(...)
    }
}

pub fn scalar_to_pyarrow<'py>(
    scalar: &ScalarValue,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    // Convert Rust scalar → Array → PyArrow
    let array = scalar.to_array().map_err(PyDataFusionError::from)?;
    let pyarray = array.to_data().to_pyarrow(py)?;
    let pyscalar = pyarray.call_method1("__getitem__", (0,))?;
    Ok(pyscalar)
}
```

### Key Concepts

- **Multi-library support**: Detects PyArrow, arro3, nanoarrow automatically
- **Arrow C Data Interface**: `__arrow_c_array__` protocol for universal interop
- **Fallback chains**: Tries multiple methods to find compatible representation

---

## FFI Boundary Comparison Matrix

| Aspect | datafusion-python | delta-rs | lance |
|--------|------------------|----------|-------|
| **Streaming Type** | `FFI_ArrowArrayStream` + `PyCapsule` | `pyo3_arrow::PyRecordBatchReader` | `PyArrowType<RecordBatchReader>` |
| **Materialization** | None (lazy stream) | Wrapped (adapter pattern) | Lazy via `RecordBatchReader` |
| **Schema Negotiation** | ✓ (requested_schema param) | ✓ (via pyo3-arrow) | ✓ (implicit) |
| **Multi-library Support** | ✓ (via Arrow PyCapsule v2) | ✓ (pyo3-arrow abstracts it) | ✓ (via PyArrowType) |
| **Crate Complexity** | Medium (manual FFI management) | Low (pyo3-arrow abstraction) | Low (PyArrowType wrapper) |
| **Arrow Version** | 58 | 58 (delta), pyo3-arrow=0.14 | 57 + lance-arrow |

---

## Recommended Implementation Path for Your SDK

### Option A: Lightweight (Similar to delta-rs)

**Best for**: Stateless query results, small team

```rust
use pyo3_arrow::{PyRecordBatchReader, PyRecordBatch};

#[pyfunction]
pub fn query_dataset(path: &str) -> PyResult<PyRecordBatchReader> {
    let reader = load_and_query(path)?;
    Ok(PyRecordBatchReader::new(reader))
}
```

**Cargo.toml**:
```toml
pyo3-arrow = "0.17"
arrow = "58"
pyo3 = "0.28"
```

### Option B: Full Control (Similar to datafusion-python)

**Best for**: Complex schema negotiation, performance-critical paths

```rust
use arrow::ffi_stream::FFI_ArrowArrayStream;
use pyo3::types::PyCapsule;
use std::ffi::CString;

#[pymethods]
impl MyDataset {
    fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyCapsule>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let reader = self.create_reader(requested_schema)?;
        let stream = FFI_ArrowArrayStream::new(reader);
        let name = CString::new("arrow_array_stream")?;
        PyCapsule::new(py, stream, Some(name))
    }
}
```

**Cargo.toml**:
```toml
arrow = "58"
arrow-schema = "58"
arrow-array = "58"
pyo3 = "0.28"
```

---

## References & Permalinks

- **datafusion-python** (`__arrow_c_stream__`): https://github.com/apache/datafusion-python/blob/ff15648c5dca6b41d3f6146c6c36c97e605f8561/crates/core/src/dataframe.rs#L1107-L1146
- **delta-rs** (pyo3-arrow): https://github.com/delta-io/delta-rs/blob/0bb2d6bc7d058c10870c4e275827639b172e813f/python/src/query.rs#L61-L72
- **lance** (PyArrowType): https://github.com/lancedb/lance/blob/d630106da5a238b3adfb8c5dea3b3921f3519945/python/src/scanner.rs#L147-L158
- **pyo3-arrow crate**: https://docs.rs/pyo3-arrow/latest/pyo3_arrow/
- **Arrow PyCapsule Interface v2**: https://arrow.apache.org/docs/format/CDataInterface.html
