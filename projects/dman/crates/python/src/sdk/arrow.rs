//! Arrow RecordBatch builders for zero-copy FFI to Python via pyo3-arrow.
//!
//! Each function converts a slice of internal row types into an Arrow RecordBatch
//! that can be exported to Python as a pyarrow.RecordBatch with no serialization.

#[cfg(feature = "python")]
pub(crate) use python_impl::*;

#[cfg(feature = "python")]
pub mod python_impl {
    use std::sync::Arc;

    use arrow_array::builder::{Float64Builder, Int64Builder, LargeListBuilder, StringBuilder};
    use arrow_array::RecordBatch;
    use arrow_schema::{DataType, Field, Schema};

    use super::super::loader::python_impl::{AnnotationRow, AssetRow, SampleRow};

    // ─── CategoryRow (defined here, not in loader) ──────────────────────────

    #[derive(Debug, Clone)]
    pub(crate) struct CategoryRow {
        pub id: i64,
        pub dataset_id: i64,
        pub name: String,
        pub supercategory: Option<String>,
    }

    // ─── samples ────────────────────────────────────────────────────────────

    pub(crate) fn samples_to_record_batch(
        rows: &[SampleRow],
    ) -> Result<RecordBatch, arrow_schema::ArrowError> {
        let mut id_b = Int64Builder::with_capacity(rows.len());
        let mut dataset_id_b = Int64Builder::with_capacity(rows.len());
        let mut name_b = StringBuilder::with_capacity(rows.len(), rows.len() * 32);
        let mut metadata_b = StringBuilder::with_capacity(rows.len(), rows.len() * 64);
        let mut created_at_b = StringBuilder::with_capacity(rows.len(), rows.len() * 24);

        for r in rows {
            id_b.append_value(r.id);
            dataset_id_b.append_value(r.dataset_id);
            name_b.append_value(&r.name);
            match &r.metadata {
                Some(m) => metadata_b.append_value(m),
                None => metadata_b.append_null(),
            }
            match &r.created_at {
                Some(ts) => created_at_b.append_value(ts),
                None => created_at_b.append_null(),
            }
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("dataset_id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("metadata", DataType::Utf8, true),
            Field::new("created_at", DataType::Utf8, true),
        ]));

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(id_b.finish()),
                Arc::new(dataset_id_b.finish()),
                Arc::new(name_b.finish()),
                Arc::new(metadata_b.finish()),
                Arc::new(created_at_b.finish()),
            ],
        )
    }

    // ─── assets ─────────────────────────────────────────────────────────────

    pub(crate) fn assets_to_record_batch(
        rows: &[AssetRow],
    ) -> Result<RecordBatch, arrow_schema::ArrowError> {
        let cap = rows.len();
        let mut id_b = Int64Builder::with_capacity(cap);
        let mut sample_id_b = Int64Builder::with_capacity(cap);
        let mut asset_type_b = StringBuilder::with_capacity(cap, cap * 8);
        let mut file_name_b = StringBuilder::with_capacity(cap, cap * 32);
        let mut file_path_b = StringBuilder::with_capacity(cap, cap * 64);
        let mut width_b = Int64Builder::with_capacity(cap);
        let mut height_b = Int64Builder::with_capacity(cap);
        let mut hash_b = StringBuilder::with_capacity(cap, cap * 64);
        let mut metadata_b = StringBuilder::with_capacity(cap, cap * 64);

        for r in rows {
            id_b.append_value(r.id);
            sample_id_b.append_value(r.sample_id);
            asset_type_b.append_value(&r.asset_type);
            file_name_b.append_value(&r.file_name);
            file_path_b.append_value(&r.file_path);
            match r.width {
                Some(w) => width_b.append_value(w),
                None => width_b.append_null(),
            }
            match r.height {
                Some(h) => height_b.append_value(h),
                None => height_b.append_null(),
            }
            match &r.hash {
                Some(h) => hash_b.append_value(h),
                None => hash_b.append_null(),
            }
            match &r.metadata {
                Some(m) => metadata_b.append_value(m),
                None => metadata_b.append_null(),
            }
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("sample_id", DataType::Int64, false),
            Field::new("asset_type", DataType::Utf8, false),
            Field::new("file_name", DataType::Utf8, false),
            Field::new("file_path", DataType::Utf8, false),
            Field::new("width", DataType::Int64, true),
            Field::new("height", DataType::Int64, true),
            Field::new("hash", DataType::Utf8, true),
            Field::new("metadata", DataType::Utf8, true),
        ]));

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(id_b.finish()),
                Arc::new(sample_id_b.finish()),
                Arc::new(asset_type_b.finish()),
                Arc::new(file_name_b.finish()),
                Arc::new(file_path_b.finish()),
                Arc::new(width_b.finish()),
                Arc::new(height_b.finish()),
                Arc::new(hash_b.finish()),
                Arc::new(metadata_b.finish()),
            ],
        )
    }

    // ─── annotations ────────────────────────────────────────────────────────

    /// Parse a JSON bbox string `{"x":..,"y":..,"width":..,"height":..}` into (x, y, w, h).
    fn parse_bbox(json: &str) -> Option<(f64, f64, f64, f64)> {
        let v: serde_json::Value = serde_json::from_str(json).ok()?;
        let obj = v.as_object()?;
        let x = obj.get("x").and_then(|v| v.as_f64())?;
        let y = obj.get("y").and_then(|v| v.as_f64())?;
        let w = obj.get("width").and_then(|v| v.as_f64())?;
        let h = obj.get("height").and_then(|v| v.as_f64())?;
        Some((x, y, w, h))
    }

    /// Parse a JSON array of floats `[1.0, 2.0, ...]`.
    fn parse_f64_list(json: &str) -> Option<Vec<f64>> {
        let v: serde_json::Value = serde_json::from_str(json).ok()?;
        let arr = v.as_array()?;
        arr.iter().map(|v| v.as_f64()).collect()
    }

    /// Parse a JSON array of arrays of floats `[[1.0, 2.0], [3.0, 4.0]]`.
    fn parse_nested_f64_list(json: &str) -> Option<Vec<Vec<f64>>> {
        let v: serde_json::Value = serde_json::from_str(json).ok()?;
        let outer = v.as_array()?;
        outer
            .iter()
            .map(|inner| {
                inner
                    .as_array()
                    .and_then(|arr| arr.iter().map(|v| v.as_f64()).collect())
            })
            .collect()
    }

    pub(crate) fn annotations_to_record_batch(
        rows: &[AnnotationRow],
    ) -> Result<RecordBatch, arrow_schema::ArrowError> {
        let cap = rows.len();
        let mut id_b = Int64Builder::with_capacity(cap);
        let mut sample_id_b = Int64Builder::with_capacity(cap);
        let mut asset_id_b = Int64Builder::with_capacity(cap);
        let mut category_id_b = Int64Builder::with_capacity(cap);
        let mut bbox_x_b = Float64Builder::with_capacity(cap);
        let mut bbox_y_b = Float64Builder::with_capacity(cap);
        let mut bbox_w_b = Float64Builder::with_capacity(cap);
        let mut bbox_h_b = Float64Builder::with_capacity(cap);
        let mut seg_b = LargeListBuilder::new(LargeListBuilder::new(Float64Builder::new()));
        let mut kp_b = LargeListBuilder::new(Float64Builder::new());
        let mut metadata_b = StringBuilder::with_capacity(cap, cap * 64);

        for r in rows {
            id_b.append_value(r.id);
            sample_id_b.append_value(r.sample_id);

            match r.asset_id {
                Some(v) => asset_id_b.append_value(v),
                None => asset_id_b.append_null(),
            }
            match r.category_id {
                Some(v) => category_id_b.append_value(v),
                None => category_id_b.append_null(),
            }

            match r.bbox.as_deref().and_then(parse_bbox) {
                Some((x, y, w, h)) => {
                    bbox_x_b.append_value(x);
                    bbox_y_b.append_value(y);
                    bbox_w_b.append_value(w);
                    bbox_h_b.append_value(h);
                }
                None => {
                    bbox_x_b.append_null();
                    bbox_y_b.append_null();
                    bbox_w_b.append_null();
                    bbox_h_b.append_null();
                }
            }

            // segmentation: JSON [[f64]] → LargeList<LargeList<Float64>>
            match r.segmentation.as_deref().and_then(parse_nested_f64_list) {
                Some(polygons) => {
                    for polygon in &polygons {
                        for &val in polygon {
                            seg_b.values().values().append_value(val);
                        }
                        seg_b.values().append(true);
                    }
                    seg_b.append(true);
                }
                None => {
                    seg_b.append(false);
                }
            }

            // keypoints: JSON [f64] → LargeList<Float64>
            match r.keypoints.as_deref().and_then(parse_f64_list) {
                Some(pts) => {
                    for &val in &pts {
                        kp_b.values().append_value(val);
                    }
                    kp_b.append(true);
                }
                None => {
                    kp_b.append(false);
                }
            }

            match &r.metadata {
                Some(m) => metadata_b.append_value(m),
                None => metadata_b.append_null(),
            }
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("sample_id", DataType::Int64, false),
            Field::new("asset_id", DataType::Int64, true),
            Field::new("category_id", DataType::Int64, true),
            Field::new("bbox_x", DataType::Float64, true),
            Field::new("bbox_y", DataType::Float64, true),
            Field::new("bbox_w", DataType::Float64, true),
            Field::new("bbox_h", DataType::Float64, true),
            Field::new(
                "segmentation",
                DataType::LargeList(Arc::new(Field::new(
                    "item",
                    DataType::LargeList(Arc::new(Field::new("item", DataType::Float64, true))),
                    true,
                ))),
                true,
            ),
            Field::new(
                "keypoints",
                DataType::LargeList(Arc::new(Field::new("item", DataType::Float64, true))),
                true,
            ),
            Field::new("metadata", DataType::Utf8, true),
        ]));

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(id_b.finish()),
                Arc::new(sample_id_b.finish()),
                Arc::new(asset_id_b.finish()),
                Arc::new(category_id_b.finish()),
                Arc::new(bbox_x_b.finish()),
                Arc::new(bbox_y_b.finish()),
                Arc::new(bbox_w_b.finish()),
                Arc::new(bbox_h_b.finish()),
                Arc::new(seg_b.finish()),
                Arc::new(kp_b.finish()),
                Arc::new(metadata_b.finish()),
            ],
        )
    }

    // ─── categories ─────────────────────────────────────────────────────────

    pub(crate) fn categories_to_record_batch(
        rows: &[CategoryRow],
    ) -> Result<RecordBatch, arrow_schema::ArrowError> {
        let cap = rows.len();
        let mut id_b = Int64Builder::with_capacity(cap);
        let mut dataset_id_b = Int64Builder::with_capacity(cap);
        let mut name_b = StringBuilder::with_capacity(cap, cap * 32);
        let mut supercategory_b = StringBuilder::with_capacity(cap, cap * 32);

        for r in rows {
            id_b.append_value(r.id);
            dataset_id_b.append_value(r.dataset_id);
            name_b.append_value(&r.name);
            match &r.supercategory {
                Some(sc) => supercategory_b.append_value(sc),
                None => supercategory_b.append_null(),
            }
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("dataset_id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("supercategory", DataType::Utf8, true),
        ]));

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(id_b.finish()),
                Arc::new(dataset_id_b.finish()),
                Arc::new(name_b.finish()),
                Arc::new(supercategory_b.finish()),
            ],
        )
    }

    // ─── Tests ──────────────────────────────────────────────────────────────

    #[cfg(test)]
    mod tests {
        use super::*;
        use arrow_array::cast::AsArray;
        use arrow_array::Array;
        use arrow_schema::DataType;

        // ── helpers ──────────────────────────────────────────────────────

        /// Assert that a RecordBatch field at `idx` has the expected name,
        /// data-type, and nullability.
        fn assert_field(
            batch: &RecordBatch,
            idx: usize,
            name: &str,
            dt: &DataType,
            nullable: bool,
        ) {
            let schema = batch.schema();
            let field = schema.field(idx);
            assert_eq!(field.name(), name, "field {idx} name mismatch");
            assert_eq!(field.data_type(), dt, "field {idx} ({name}) type mismatch");
            assert_eq!(
                field.is_nullable(),
                nullable,
                "field {idx} ({name}) nullable mismatch"
            );
        }

        // ── samples ─────────────────────────────────────────────────────

        #[test]
        fn samples_empty_batch_has_correct_schema() {
            let batch = samples_to_record_batch(&[]).unwrap();
            assert_eq!(batch.num_rows(), 0);
            assert_eq!(batch.num_columns(), 5);
            assert_field(&batch, 0, "id", &DataType::Int64, false);
            assert_field(&batch, 1, "dataset_id", &DataType::Int64, false);
            assert_field(&batch, 2, "name", &DataType::Utf8, false);
            assert_field(&batch, 3, "metadata", &DataType::Utf8, true);
            assert_field(&batch, 4, "created_at", &DataType::Utf8, true);
        }

        #[test]
        fn samples_round_trip_values_and_nulls() {
            let rows = vec![
                SampleRow {
                    id: 1,
                    dataset_id: 10,
                    name: "frame001".into(),
                    metadata: Some(r#"{"key":"val"}"#.into()),
                    created_at: Some("2025-01-01T00:00:00".into()),
                },
                SampleRow {
                    id: 2,
                    dataset_id: 10,
                    name: "frame002".into(),
                    metadata: None,
                    created_at: None,
                },
            ];
            let batch = samples_to_record_batch(&rows).unwrap();
            assert_eq!(batch.num_rows(), 2);

            // Non-null columns: check values
            let ids = batch
                .column(0)
                .as_primitive::<arrow_array::types::Int64Type>();
            assert_eq!(ids.value(0), 1);
            assert_eq!(ids.value(1), 2);

            let names = batch.column(2).as_string::<i32>();
            assert_eq!(names.value(0), "frame001");
            assert_eq!(names.value(1), "frame002");

            // Nullable columns: check null pattern
            assert!(!batch.column(3).is_null(0)); // metadata present
            assert!(batch.column(3).is_null(1)); // metadata absent
            assert!(!batch.column(4).is_null(0)); // created_at present
            assert!(batch.column(4).is_null(1)); // created_at absent
        }

        // ── assets ──────────────────────────────────────────────────────

        #[test]
        fn assets_empty_batch_has_correct_schema() {
            let batch = assets_to_record_batch(&[]).unwrap();
            assert_eq!(batch.num_rows(), 0);
            assert_eq!(batch.num_columns(), 9);
            assert_field(&batch, 0, "id", &DataType::Int64, false);
            assert_field(&batch, 1, "sample_id", &DataType::Int64, false);
            assert_field(&batch, 2, "asset_type", &DataType::Utf8, false);
            assert_field(&batch, 3, "file_name", &DataType::Utf8, false);
            assert_field(&batch, 4, "file_path", &DataType::Utf8, false);
            assert_field(&batch, 5, "width", &DataType::Int64, true);
            assert_field(&batch, 6, "height", &DataType::Int64, true);
            assert_field(&batch, 7, "hash", &DataType::Utf8, true);
            assert_field(&batch, 8, "metadata", &DataType::Utf8, true);
        }

        #[test]
        fn assets_round_trip_with_nullable_fields() {
            let rows = vec![
                AssetRow {
                    id: 1,
                    sample_id: 1,
                    asset_type: "image".into(),
                    file_name: "img.jpg".into(),
                    file_path: "/data/img.jpg".into(),
                    width: Some(640),
                    height: Some(480),
                    hash: Some("abc123".into()),
                    metadata: None,
                },
                AssetRow {
                    id: 2,
                    sample_id: 1,
                    asset_type: "depth_map".into(),
                    file_name: "depth.png".into(),
                    file_path: "/data/depth.png".into(),
                    width: None,
                    height: None,
                    hash: None,
                    metadata: Some(r#"{"source":"lidar"}"#.into()),
                },
            ];
            let batch = assets_to_record_batch(&rows).unwrap();
            assert_eq!(batch.num_rows(), 2);

            // width: row 0 = 640, row 1 = null
            let width = batch
                .column(5)
                .as_primitive::<arrow_array::types::Int64Type>();
            assert_eq!(width.value(0), 640);
            assert!(batch.column(5).is_null(1));

            // metadata: row 0 = null, row 1 = present
            assert!(batch.column(8).is_null(0));
            assert!(!batch.column(8).is_null(1));
        }

        // ── annotations ─────────────────────────────────────────────────

        #[test]
        fn annotations_empty_batch_has_correct_schema() {
            let batch = annotations_to_record_batch(&[]).unwrap();
            assert_eq!(batch.num_rows(), 0);
            assert_eq!(batch.num_columns(), 11);
            assert_field(&batch, 0, "id", &DataType::Int64, false);
            assert_field(&batch, 1, "sample_id", &DataType::Int64, false);
            assert_field(&batch, 2, "asset_id", &DataType::Int64, true);
            assert_field(&batch, 3, "category_id", &DataType::Int64, true);
            assert_field(&batch, 4, "bbox_x", &DataType::Float64, true);
            assert_field(&batch, 5, "bbox_y", &DataType::Float64, true);
            assert_field(&batch, 6, "bbox_w", &DataType::Float64, true);
            assert_field(&batch, 7, "bbox_h", &DataType::Float64, true);

            // segmentation: LargeList<LargeList<Float64>>
            let schema = batch.schema();
            let seg_field = schema.field(8);
            assert_eq!(seg_field.name(), "segmentation");
            assert!(seg_field.is_nullable());
            match seg_field.data_type() {
                DataType::LargeList(inner) => {
                    assert_eq!(inner.name(), "item");
                    match inner.data_type() {
                        DataType::LargeList(innermost) => {
                            assert_eq!(innermost.data_type(), &DataType::Float64);
                        }
                        other => panic!("expected LargeList inner, got {other:?}"),
                    }
                }
                other => panic!("expected LargeList for segmentation, got {other:?}"),
            }

            // keypoints: LargeList<Float64>
            let kp_field = schema.field(9);
            assert_eq!(kp_field.name(), "keypoints");
            assert!(kp_field.is_nullable());
            match kp_field.data_type() {
                DataType::LargeList(inner) => {
                    assert_eq!(inner.data_type(), &DataType::Float64);
                }
                other => panic!("expected LargeList for keypoints, got {other:?}"),
            }

            assert_field(&batch, 10, "metadata", &DataType::Utf8, true);
        }

        #[test]
        fn annotations_bbox_values_round_trip() {
            let rows = vec![AnnotationRow {
                id: 1,
                sample_id: 1,
                asset_id: Some(1),
                category_id: Some(0),
                bbox: Some(r#"{"x":10.0,"y":20.0,"width":30.0,"height":40.0}"#.into()),
                segmentation: Some("[[1.0,2.0,3.0,4.0],[5.0,6.0]]".into()),
                keypoints: Some("[1.0,2.0,1.0,3.0,4.0,1.0]".into()),
                metadata: None,
            }];
            let batch = annotations_to_record_batch(&rows).unwrap();
            assert_eq!(batch.num_rows(), 1);

            let bbox_x = batch
                .column(4)
                .as_primitive::<arrow_array::types::Float64Type>();
            let bbox_y = batch
                .column(5)
                .as_primitive::<arrow_array::types::Float64Type>();
            let bbox_w = batch
                .column(6)
                .as_primitive::<arrow_array::types::Float64Type>();
            let bbox_h = batch
                .column(7)
                .as_primitive::<arrow_array::types::Float64Type>();
            assert!((bbox_x.value(0) - 10.0).abs() < f64::EPSILON);
            assert!((bbox_y.value(0) - 20.0).abs() < f64::EPSILON);
            assert!((bbox_w.value(0) - 30.0).abs() < f64::EPSILON);
            assert!((bbox_h.value(0) - 40.0).abs() < f64::EPSILON);

            // asset_id and category_id non-null
            assert!(!batch.column(2).is_null(0));
            assert!(!batch.column(3).is_null(0));
        }

        #[test]
        fn annotations_all_null_optional_fields() {
            let rows = vec![AnnotationRow {
                id: 1,
                sample_id: 1,
                asset_id: None,
                category_id: None,
                bbox: None,
                segmentation: None,
                keypoints: None,
                metadata: None,
            }];
            let batch = annotations_to_record_batch(&rows).unwrap();
            assert_eq!(batch.num_rows(), 1);

            // All optional columns should be null
            for col_idx in [2, 3, 4, 5, 6, 7, 8, 9, 10] {
                assert!(
                    batch.column(col_idx).is_null(0),
                    "column {col_idx} ({}) should be null",
                    batch.schema().field(col_idx).name()
                );
            }
        }

        #[test]
        fn annotations_malformed_bbox_produces_nulls() {
            let rows = vec![AnnotationRow {
                id: 1,
                sample_id: 1,
                asset_id: None,
                category_id: None,
                bbox: Some("not valid json".into()),
                segmentation: None,
                keypoints: None,
                metadata: None,
            }];
            let batch = annotations_to_record_batch(&rows).unwrap();
            assert_eq!(batch.num_rows(), 1);
            // Malformed bbox should produce null for all 4 bbox columns
            assert!(batch.column(4).is_null(0), "bbox_x should be null");
            assert!(batch.column(5).is_null(0), "bbox_y should be null");
            assert!(batch.column(6).is_null(0), "bbox_w should be null");
            assert!(batch.column(7).is_null(0), "bbox_h should be null");
        }

        #[test]
        fn annotations_malformed_segmentation_produces_null() {
            let rows = vec![AnnotationRow {
                id: 1,
                sample_id: 1,
                asset_id: None,
                category_id: None,
                bbox: None,
                segmentation: Some("{bad}".into()),
                keypoints: None,
                metadata: None,
            }];
            let batch = annotations_to_record_batch(&rows).unwrap();
            assert_eq!(batch.num_rows(), 1);
            assert!(batch.column(8).is_null(0), "segmentation should be null");
        }

        #[test]
        fn annotations_malformed_keypoints_produces_null() {
            let rows = vec![AnnotationRow {
                id: 1,
                sample_id: 1,
                asset_id: None,
                category_id: None,
                bbox: None,
                segmentation: None,
                keypoints: Some("not an array".into()),
                metadata: None,
            }];
            let batch = annotations_to_record_batch(&rows).unwrap();
            assert_eq!(batch.num_rows(), 1);
            assert!(batch.column(9).is_null(0), "keypoints should be null");
        }

        #[test]
        fn annotations_bbox_missing_fields_produces_nulls() {
            // Valid JSON but missing required bbox keys
            let rows = vec![AnnotationRow {
                id: 1,
                sample_id: 1,
                asset_id: None,
                category_id: None,
                bbox: Some(r#"{"x":1.0,"y":2.0}"#.into()), // missing width+height
                segmentation: None,
                keypoints: None,
                metadata: None,
            }];
            let batch = annotations_to_record_batch(&rows).unwrap();
            assert!(batch.column(4).is_null(0), "bbox_x should be null");
            assert!(batch.column(5).is_null(0), "bbox_y should be null");
        }

        // ── categories ──────────────────────────────────────────────────

        #[test]
        fn categories_empty_batch_has_correct_schema() {
            let batch = categories_to_record_batch(&[]).unwrap();
            assert_eq!(batch.num_rows(), 0);
            assert_eq!(batch.num_columns(), 4);
            assert_field(&batch, 0, "id", &DataType::Int64, false);
            assert_field(&batch, 1, "dataset_id", &DataType::Int64, false);
            assert_field(&batch, 2, "name", &DataType::Utf8, false);
            assert_field(&batch, 3, "supercategory", &DataType::Utf8, true);
        }

        #[test]
        fn categories_round_trip_with_null_supercategory() {
            let rows = vec![
                CategoryRow {
                    id: 0,
                    dataset_id: 1,
                    name: "person".into(),
                    supercategory: Some("human".into()),
                },
                CategoryRow {
                    id: 1,
                    dataset_id: 1,
                    name: "car".into(),
                    supercategory: None,
                },
            ];
            let batch = categories_to_record_batch(&rows).unwrap();
            assert_eq!(batch.num_rows(), 2);

            let names = batch.column(2).as_string::<i32>();
            assert_eq!(names.value(0), "person");
            assert_eq!(names.value(1), "car");

            // supercategory: row 0 present, row 1 null
            assert!(!batch.column(3).is_null(0));
            assert!(batch.column(3).is_null(1));
            let supercat = batch.column(3).as_string::<i32>();
            assert_eq!(supercat.value(0), "human");
        }
    }
}
