# Arrow FFI Implementation Quick Start

## 3-Minute Overview

Your goal: **Export Arrow data from Rust→Python zero-copy via PyO3**

Three production patterns found:

1. **pyo3-arrow** (Recommended for new projects) ⭐
2. **Manual FFI_ArrowArrayStream** (Maximum control)
3. **PyArrowType wrapper** (If using arrow-rs)

---

## Pattern 1: pyo3-arrow (RECOMMENDED)

### Cargo.toml
```toml
[dependencies]
pyo3 = "0.28"
pyo3-arrow = "0.17"
arrow = "58"
arrow-array = "58"
```

### Rust Code
```rust
use pyo3::prelude::*;
use pyo3_arrow::PyRecordBatchReader;
use arrow::array::RecordBatchReader;

#[pyfunction]
pub fn get_data(py: Python, path: &str) -> PyResult<PyRecordBatchReader> {
    // 1. Create your RecordBatchReader however you want
    let reader: Box<dyn RecordBatchReader + Send> =
        load_data_and_create_reader(path)?;

    // 2. Wrap it—automatic __arrow_c_stream__ implementation!
    Ok(PyRecordBatchReader::new(reader))
}
```

### Python Code
```python
import pyarrow as pa
import my_extension

# Returns PyRecordBatchReader automatically
reader = my_extension.get_data("data.parquet")

# Consume as PyArrow
table = pa.RecordBatchReader._import_from_c_capsule(
    reader.__arrow_c_stream__()
)
# OR works with arro3, nanoarrow, etc.!
```

### ✓ Why This Works
- ✓ Automatic `__arrow_c_stream__()` implementation
- ✓ Works with PyArrow, arro3, nanoarrow seamlessly
- ✓ Handles view type normalization
- ✓ 3 lines of Rust code
- ✓ Zero-copy streaming

**Time to implement**: 30 minutes

---

## Pattern 2: Manual FFI (Maximum Control)

For complex schema negotiation or performance tuning.

### Cargo.toml
```toml
[dependencies]
pyo3 = "0.28"
arrow = "58"
arrow-schema = "58"
arrow-array = "58"
```

### Rust Code
```rust
use std::ffi::CString;
use arrow::ffi_stream::FFI_ArrowArrayStream;
use arrow::array::RecordBatchReader;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

#[pyclass]
pub struct Dataset {
    data: Vec<u8>,
}

#[pymethods]
impl Dataset {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        Ok(Dataset { data: load_file(path)? })
    }

    /// Implements Arrow PyCapsule Interface v2
    fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyCapsule>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        // 1. Create your RecordBatchReader
        let reader: Box<dyn RecordBatchReader + Send> =
            self.create_reader()?;

        // 2. Convert to FFI stream
        let stream = FFI_ArrowArrayStream::new(reader);

        // 3. Wrap in PyCapsule with proper name
        let name = CString::new("arrow_array_stream")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        PyCapsule::new(py, stream, Some(name))
    }
}
```

### Python Code
```python
import pyarrow as pa
import my_extension

dataset = my_extension.Dataset("data.parquet")

# Direct PyCapsule consumption
reader = pa.RecordBatchReader._import_from_c_capsule(
    dataset.__arrow_c_stream__()
)
```

### ✓ Why This Works
- ✓ Full control over schema, filtering, casting
- ✓ Can implement schema negotiation (requested_schema parameter)
- ✓ Direct FFI for performance tuning
- ✗ More complex error handling

**Time to implement**: 2 hours

---

## Pattern 3: PyArrowType (If using arrow-rs features)

Simple if `arrow = { version = "58", features = ["pyarrow"] }`

### Cargo.toml
```toml
[dependencies]
pyo3 = "0.28"
arrow = { version = "58", features = ["pyarrow"] }
```

### Rust Code
```rust
use arrow::pyarrow::IntoPyArrow;
use arrow::array::RecordBatchReader;
use pyo3::prelude::*;

#[pyclass]
pub struct Scanner;

#[pymethods]
impl Scanner {
    fn to_pyarrow<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let reader: Box<dyn RecordBatchReader + Send> = self.scan()?;
        reader.into_pyarrow(py)
    }
}
```

### ✓ Why This Works
- ✓ Leverages arrow-rs built-in support
- ✓ Very clean API
- ✗ Requires PyArrow feature (adds ~50MB to wheels)

**Time to implement**: 15 minutes

---

## Decision Tree

```
Do you need schema negotiation?
├─ NO
│  └─ Can use arrow-rs "pyarrow" feature?
│     ├─ YES → Pattern 3 (PyArrowType) ⏱ 15 min
│     └─ NO  → Pattern 1 (pyo3-arrow) ⏱ 30 min ⭐
│
└─ YES
   └─ Performance-critical streaming?
      ├─ YES → Pattern 2 (Manual FFI) ⏱ 2 hours
      └─ NO  → Pattern 1 (pyo3-arrow) ⏱ 30 min ⭐
```

**Recommendation**: Start with Pattern 1 (pyo3-arrow). It covers 95% of use cases in 30 minutes.

---

## Critical Implementation Checklist

- [ ] **RecordBatchReader trait**: Your data source must implement `arrow::array::RecordBatchReader` or wrap it
- [ ] **Lazy evaluation**: Use streaming, not eager materialization (unless <100MB)
- [ ] **Error handling**: Use PyResult everywhere, let `?` operator convert Rust errors to Python
- [ ] **GIL release**: If CPU-intensive, release GIL with `py.detach()` or `py.allow_threads()`
- [ ] **Schema consistency**: Ensure schema doesn't change mid-stream
- [ ] **Memory safety**: RecordBatchReader's lifetime must outlive PyCapsule

### Test Your Implementation
```python
import pyarrow as pa

# Test 1: Basic consumption
table = pa.table(my_reader)
assert table.num_rows > 0

# Test 2: arro3 compatibility
import arro3.compute as acp
# If your data works with arro3, you have true zero-copy ✓

# Test 3: Batch streaming
for batch in my_reader:
    assert isinstance(batch, pa.RecordBatch)
```

---

## Common Pitfalls

### ❌ Pitfall 1: Materializing entire dataset
```rust
// DON'T DO THIS
let all_data = collect_entire_dataset()?;  // ← Uses all RAM
Ok(PyRecordBatchReader::new(all_data))
```

**Fix**: Stream data in chunks via `RecordBatchReader` iterator

### ❌ Pitfall 2: Holding Python GIL during I/O
```rust
// DON'T DO THIS
fn read_data(py: Python) -> PyResult<Data> {
    let data = load_from_disk()?;  // ← Blocks GIL, freezes Python
    Ok(data)
}
```

**Fix**: Use `py.detach()` or Tokio with `block_in_place`
```rust
let data = py.detach(|| {
    std::fs::read("file.bin")  // ← GIL released
})?;
```

### ❌ Pitfall 3: Static lifetimes
```rust
// DON'T DO THIS
pub fn __arrow_c_stream__(
    &self,
    py: Python,
    requested_schema: Option<Bound<PyCapsule>>,  // ← Wrong lifetime
) -> PyResult<Bound<PyCapsule>> {
    // ...
}
```

**Fix**: Use `<'py>` lifetime parameter
```rust
pub fn __arrow_c_stream__<'py>(
    &'py self,
    py: Python<'py>,
    requested_schema: Option<Bound<'py, PyCapsule>>,  // ← Correct
) -> PyResult<Bound<'py, PyCapsule>> {
    // ...
}
```

---

## Debugging Tips

### Issue: "arrow_array_stream" capsule not recognized
```python
# Check if PyCapsule has correct name
capsule = obj.__arrow_c_stream__()
print(capsule)  # Should show "PyCapsule: arrow_array_stream"
```

**Fix**: Ensure `CString::new("arrow_array_stream")` matches exactly

### Issue: RecordBatch schema mismatch
```rust
// Your RecordBatchReader returns different schema each batch?
let batch1_schema = batch1.schema().clone();
let batch2_schema = batch2.schema().clone();
assert_eq!(batch1_schema, batch2_schema);  // Must match!
```

### Issue: Hangs when consuming stream
```python
# GIL is still held? Try:
import psutil
p = psutil.Process()
print(p.num_threads())  # Should be > 1 if GIL released
```

---

## Performance Tips

1. **Batch size tuning**: Larger batches (64K-1M rows) reduce FFI overhead
   ```rust
   const BATCH_SIZE: usize = 65536;  // Experiment with this
   ```

2. **Schema projection**: Support filtering columns at query time
   ```rust
   if let Some(requested_schema) = requested_schema {
       // Project to only requested columns before FFI
   }
   ```

3. **Copy-on-write**: Arrow uses CoW by default—reuse buffers when possible
   ```rust
   // Good: Arrow automatically shares underlying buffers
   let arr = make_array(data);
   ```

4. **View types**: Use Utf8View/BinaryView for string-heavy data
   ```rust
   // Modern Arrow defaults—better performance on strings
   ```

---

## References

| Resource | Link |
|----------|------|
| pyo3-arrow docs | https://docs.rs/pyo3-arrow/ |
| Arrow FFI spec | https://arrow.apache.org/docs/format/CDataInterface.html |
| delta-rs example | https://github.com/delta-io/delta-rs/blob/main/python/src/query.rs |
| datafusion-python | https://github.com/apache/datafusion-python/blob/main/crates/core/src/dataframe.rs |
| lance (PyArrowType) | https://github.com/lancedb/lance/blob/main/python/src/scanner.rs |

---

## Next Steps

1. Choose pattern (recommendation: **Pattern 1 - pyo3-arrow**)
2. Create `src/lib.rs` with PyO3 module
3. Define data source struct + `__arrow_c_stream__()`
4. Test with: `python -c "import my_lib; print(my_lib.get_data()).__arrow_c_stream__()"`
5. Consume in Python with PyArrow/arro3/nanoarrow

**Good luck!** 🚀 The hardest part is behind you—you now have production patterns to reference.
