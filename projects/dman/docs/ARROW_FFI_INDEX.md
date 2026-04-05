# Arrow Zero-Copy FFI Documentation Index

## 📚 Documentation Overview

This directory contains comprehensive research and implementation guides for Arrow↔Python zero-copy data exchange via PyO3. All patterns are based on production code from major OSS projects.

---

## 📖 Documents

### 1. **ARROW_FFI_QUICK_START.md** ⭐ START HERE
**Duration**: 5 minutes | **Audience**: Developers ready to implement

Fast-track guide with three implementation patterns:
- **Pattern 1: pyo3-arrow** (30 min, recommended)
- **Pattern 2: Manual FFI** (2 hours, maximum control)
- **Pattern 3: PyArrowType** (15 min, if using arrow-rs features)

**Includes**:
- Ready-to-copy code snippets
- Decision tree for choosing your pattern
- Common pitfalls & fixes
- Debugging tips
- Performance tuning

**Start here if**: You want working code in the next hour.

---

### 2. **ARROW_FFI_RESEARCH_SUMMARY.md**
**Duration**: 10 minutes | **Audience**: Decision makers, architects

Executive summary of real production implementations:

**Projects Analyzed**:
1. **datafusion-python** — Arrow C Data Interface + PyCapsule (most control)
2. **delta-rs** — pyo3-arrow wrapper (lightweight)
3. **lance** — PyArrowType generic wrapper (clean)
4. **arro3** — Minimal PyCapsule Interface v2 bridge

**Includes**:
- Key findings from each project
- FFI comparison matrix
- Dependency recommendations
- Schema negotiation patterns
- Async→Sync bridge patterns
- Multi-library support strategy
- Real-world observations

**Start here if**: You need to understand tradeoffs and choose an approach.

---

### 3. **ARROW_FFI_PATTERNS.md**
**Duration**: 30 minutes | **Audience**: Implementers, deep-dive readers

Detailed production code patterns with full implementation examples.

**Sections**:
- **Pattern 1**: Arrow C Data Interface Stream (datafusion-python)
- **Pattern 2**: pyo3-arrow Wrapper (delta-rs)
- **Pattern 3**: PyArrowType Wrapper (lance)
- **Pattern 4**: Direct RecordBatch Materialization
- **Pattern 5**: Array-Level FFI
- **FFI Boundary Comparison Matrix**
- **Recommended Implementation Paths**

**Each pattern includes**:
- Full Rust code with annotations
- Exact GitHub permalink to source
- Python consumption example
- Cargo.toml dependencies
- Key concepts explanation

**Start here if**: You want to understand the mechanics deeply.

---

## 🎯 Quick Decision Guide

### Q1: Do you have existing RecordBatchReader?
- **YES** → Use Pattern 1 (pyo3-arrow) — just wrap it!
- **NO** → Write one or use engine like DataFusion/DuckDB

### Q2: Do you need schema negotiation (column filtering)?
- **YES** → Pattern 2 (Manual FFI) OR Pattern 1 with pyo3-arrow
- **NO** → Any pattern works; choose by complexity

### Q3: Is PyArrow dependency acceptable?
- **YES** → Pattern 3 (PyArrowType + arrow "pyarrow" feature)
- **NO** → Pattern 1 (pyo3-arrow) or Pattern 2 (manual)

### Q4: How much time do you have?
- **< 1 hour** → Pattern 1 (pyo3-arrow) ⏱️ 30 min
- **1-2 hours** → Pattern 3 (PyArrowType) ⏱️ 15 min + Polish
- **2+ hours** → Pattern 2 (Manual FFI) ⏱️ 2 hours + Testing

---

## 🚀 Implementation Checklist

### Phase 1: Setup (10 minutes)
- [ ] Read ARROW_FFI_QUICK_START.md
- [ ] Choose pattern based on decision tree
- [ ] Create Cargo.toml with dependencies

### Phase 2: Core Implementation (30-120 minutes)
- [ ] Create `__arrow_c_stream__()` method (or wrapper)
- [ ] Implement RecordBatchReader trait
- [ ] Handle error cases with PyResult

### Phase 3: Integration (30 minutes)
- [ ] Test with PyArrow consumption
- [ ] Test with arro3/nanoarrow (optional)
- [ ] Release GIL during I/O (`py.detach()`)

### Phase 4: Optimization (30+ minutes, optional)
- [ ] Tune batch sizes
- [ ] Implement schema negotiation
- [ ] Profile memory usage

---

## 📊 Dependency Comparison

| Aspect | Pattern 1 (pyo3-arrow) | Pattern 2 (Manual) | Pattern 3 (PyArrowType) |
|--------|------------------------|-------------------|------------------------|
| Setup time | 30 min | 2 hours | 15 min |
| Arrow version | 58 | 58 | 58 |
| PyO3 version | 0.28 | 0.28 | 0.28 |
| Multi-library support | ✓ | ✓ | PyArrow only |
| Schema negotiation | ✓ | ✓ | Limited |
| Code complexity | Low | Medium | Very low |
| PyArrow dependency | No | No | Yes (~50MB) |

---

## 🔗 Key Resources

### Official Specifications
- **Arrow PyCapsule Interface v2**: https://arrow.apache.org/docs/format/CDataInterface.html
- **pyo3-arrow crate**: https://docs.rs/pyo3-arrow/latest/pyo3_arrow/

### Production Examples (with permalinks)
- **datafusion-python**: https://github.com/apache/datafusion-python/blob/ff15648c5dca6b41d3f6146c6c36c97e605f8561/crates/core/src/dataframe.rs#L1107-L1146
- **delta-rs**: https://github.com/delta-io/delta-rs/blob/0bb2d6bc7d058c10870c4e275827639b172e813f/python/src/query.rs#L61-L72
- **lance**: https://github.com/lancedb/lance/blob/d630106da5a238b3adfb8c5dea3b3921f3519945/python/src/scanner.rs#L147-L158

---

## ⚡ Common Questions

### Q: How is this zero-copy?
The Arrow C Data Interface (PyCapsule) passes **pointers to memory buffers**, not copies. Both Rust and Python see the same underlying bytes.

### Q: Which pattern is production-ready?
All three patterns are production-ready:
- **pyo3-arrow**: Used by delta-rs (Delta Lake)
- **Manual FFI**: Used by apache/datafusion-python
- **PyArrowType**: Used by lancedb/lance

### Q: Do I need to optimize batch sizes?
Not initially. Arrow defaults to reasonable batches (~64KB rows). Tune later if profiling shows it helps.

### Q: What if I have async code?
Use `py.detach()` to release GIL, or `tokio::task::block_in_place` to block async runtime. See delta-rs example in Pattern 2.

### Q: Will this work with arro3 and nanoarrow?
Yes! The Arrow PyCapsule Interface is universal. Any Python library supporting v2 works automatically.

### Q: How much does Arrow C Data Interface cost?
Minimal overhead (<1ms per stream). The FFI calls are just pointer arithmetic; no data copying occurs.

---

## 📝 Implementation Tips

1. **Start with Pattern 1 (pyo3-arrow)** if unsure—covers 95% of use cases
2. **RecordBatchReader must be stable**—schema can't change mid-stream
3. **Release the GIL**—use `py.detach()` during I/O
4. **Test with multiple Python libraries**—PyArrow + arro3 confirms zero-copy
5. **Profile before optimizing**—batch sizes rarely matter unless >1B rows

---

## 🐛 Support & Debugging

### Issue: "arrow_array_stream" not recognized
See **ARROW_FFI_QUICK_START.md → Debugging Tips → Issue 1**

### Issue: RecordBatch schema mismatch
See **ARROW_FFI_QUICK_START.md → Debugging Tips → Issue 2**

### Issue: Hangs or freezes
See **ARROW_FFI_QUICK_START.md → Pitfalls → Pitfall 2** (GIL management)

### Issue: Need complex schema negotiation
See **ARROW_FFI_PATTERNS.md → Pattern 2** (Manual FFI with requested_schema)

---

## 📌 Document Metadata

| Document | Lines | Topics | Audience |
|----------|-------|--------|----------|
| ARROW_FFI_QUICK_START.md | 359 | 3 patterns, pitfalls, checklist | Implementers |
| ARROW_FFI_RESEARCH_SUMMARY.md | 287 | Research findings, decisions | Architects |
| ARROW_FFI_PATTERNS.md | 492 | Deep-dive patterns, permalinks | Advanced users |
| **Total** | **1,138** | **Comprehensive FFI reference** | All levels |

---

## 🎓 Learning Path

1. **5 min**: Skim this INDEX (you are here)
2. **10 min**: Read ARROW_FFI_RESEARCH_SUMMARY.md
3. **5 min**: Run decision tree in ARROW_FFI_QUICK_START.md
4. **30 min**: Copy code from chosen pattern
5. **30 min**: Test and iterate
6. **Reference**: Use ARROW_FFI_PATTERNS.md for deep questions

**Total time**: ~80 minutes from zero to working implementation

---

## ✨ Key Takeaways

- **Three production patterns exist**; pyo3-arrow is recommended for new projects
- **Arrow C Data Interface** enables true zero-copy data exchange
- **RecordBatchReader trait** is the gateway; stream everything
- **Multi-library support** (PyArrow/arro3/nanoarrow) works out-of-the-box
- **Schema negotiation** allows Python to request specific columns before streaming

---

**Ready to start?** → Open **ARROW_FFI_QUICK_START.md** next! 🚀
