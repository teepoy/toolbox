use crate::{
    error::{DmanError, Result},
    schema::{ColumnDef, SchemaDefinition},
    types::{Image, SchemaOp},
};

pub struct SchemaTransformer;

impl SchemaTransformer {
    pub fn apply(schema: &SchemaDefinition, ops: &[SchemaOp]) -> Result<SchemaDefinition> {
        let mut current = schema.clone();
        for op in ops {
            current = apply_single(&current, op)?;
        }
        Ok(current)
    }

    pub fn apply_to_images(images: &[Image], ops: &[SchemaOp]) -> Result<Vec<Image>> {
        let mut result: Vec<Image> = Vec::with_capacity(images.len());
        for img in images {
            let new_metadata = transform_metadata(img.metadata.clone(), ops);
            result.push(Image {
                id: img.id,
                dataset_id: img.dataset_id,
                file_name: img.file_name.clone(),
                file_path: img.file_path.clone(),
                width: img.width,
                height: img.height,
                hash: img.hash.clone(),
                metadata: new_metadata,
            });
        }
        Ok(result)
    }
}

fn apply_single(schema: &SchemaDefinition, op: &SchemaOp) -> Result<SchemaDefinition> {
    match op {
        SchemaOp::RenameColumn { from, to } => {
            let pos = find_column(&schema.columns, from).ok_or_else(|| {
                DmanError::SchemaValidation(format!(
                    "RenameColumn: column '{}' not found in schema '{}'",
                    from, schema.name
                ))
            })?;

            if from != to && find_column(&schema.columns, to).is_some() {
                return Err(DmanError::SchemaValidation(format!(
                    "RenameColumn: target column '{}' already exists in schema '{}'",
                    to, schema.name
                )));
            }

            let mut columns = schema.columns.clone();
            columns[pos].name = to.clone();

            Ok(SchemaDefinition {
                name: schema.name.clone(),
                version: schema.version.clone(),
                columns,
                annotation_format: schema.annotation_format.clone(),
            })
        }

        SchemaOp::AddColumn {
            name,
            dtype,
            default,
        } => {
            if find_column(&schema.columns, name).is_some() {
                return Err(DmanError::SchemaValidation(format!(
                    "AddColumn: column '{}' already exists in schema '{}'",
                    name, schema.name
                )));
            }

            let data_type = parse_dtype(dtype)?;
            let toml_default = json_value_to_toml(default);

            let new_col = ColumnDef {
                name: name.clone(),
                dtype: data_type,
                required: false,
                default: toml_default,
                description: None,
            };

            let mut columns = schema.columns.clone();
            columns.push(new_col);

            Ok(SchemaDefinition {
                name: schema.name.clone(),
                version: schema.version.clone(),
                columns,
                annotation_format: schema.annotation_format.clone(),
            })
        }

        SchemaOp::RemoveColumn(name) => {
            let pos = find_column(&schema.columns, name).ok_or_else(|| {
                DmanError::SchemaValidation(format!(
                    "RemoveColumn: column '{}' not found in schema '{}'",
                    name, schema.name
                ))
            })?;

            let mut columns = schema.columns.clone();
            columns.remove(pos);

            Ok(SchemaDefinition {
                name: schema.name.clone(),
                version: schema.version.clone(),
                columns,
                annotation_format: schema.annotation_format.clone(),
            })
        }

        SchemaOp::CastColumn { name, dtype } => {
            let pos = find_column(&schema.columns, name).ok_or_else(|| {
                DmanError::SchemaValidation(format!(
                    "CastColumn: column '{}' not found in schema '{}'",
                    name, schema.name
                ))
            })?;

            let new_dtype = parse_dtype(dtype)?;
            let mut columns = schema.columns.clone();
            columns[pos].dtype = new_dtype;

            Ok(SchemaDefinition {
                name: schema.name.clone(),
                version: schema.version.clone(),
                columns,
                annotation_format: schema.annotation_format.clone(),
            })
        }
    }
}

fn find_column(columns: &[ColumnDef], name: &str) -> Option<usize> {
    columns.iter().position(|c| c.name == name)
}

fn parse_dtype(dtype: &str) -> Result<crate::schema::DataType> {
    serde_json::from_value::<crate::schema::DataType>(serde_json::json!(dtype))
        .map_err(|_| DmanError::SchemaValidation(format!("unknown dtype: '{}'", dtype)))
}

fn json_value_to_toml(val: &serde_json::Value) -> Option<toml::Value> {
    match val {
        serde_json::Value::Null => None,
        serde_json::Value::Bool(b) => Some(toml::Value::Boolean(*b)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(toml::Value::Integer(i))
            } else if let Some(f) = n.as_f64() {
                Some(toml::Value::Float(f))
            } else {
                None
            }
        }
        serde_json::Value::String(s) => Some(toml::Value::String(s.clone())),
        serde_json::Value::Array(arr) => {
            let toml_arr: Vec<toml::Value> = arr.iter().filter_map(json_value_to_toml).collect();
            Some(toml::Value::Array(toml_arr))
        }
        serde_json::Value::Object(_) => Some(toml::Value::String(val.to_string())),
    }
}

fn transform_metadata(
    metadata: Option<serde_json::Value>,
    ops: &[SchemaOp],
) -> Option<serde_json::Value> {
    let has_add_with_default = ops
        .iter()
        .any(|op| matches!(op, SchemaOp::AddColumn { default, .. } if !default.is_null()));
    if metadata.is_none() && !has_add_with_default {
        return None;
    }

    let mut obj = match metadata {
        Some(serde_json::Value::Object(m)) => m,
        Some(other) => return Some(other),
        None => serde_json::Map::new(),
    };

    for op in ops {
        match op {
            SchemaOp::RenameColumn { from, to } => {
                if let Some(val) = obj.remove(from) {
                    obj.insert(to.clone(), val);
                }
            }
            SchemaOp::AddColumn { name, default, .. } => {
                if !obj.contains_key(name) && !default.is_null() {
                    obj.insert(name.clone(), default.clone());
                }
            }
            SchemaOp::RemoveColumn(name) => {
                obj.remove(name);
            }
            SchemaOp::CastColumn { .. } => {}
        }
    }

    if obj.is_empty() {
        None
    } else {
        Some(serde_json::Value::Object(obj))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Schema;

    fn make_schema() -> SchemaDefinition {
        Schema::from_str(
            r#"
name = "test-schema"
version = "1"

[[columns]]
name = "label"
dtype = "String"
required = true

[[columns]]
name = "confidence"
dtype = "Float"
required = false

[[columns]]
name = "count"
dtype = "Int"
required = false
"#,
        )
        .expect("parse schema")
    }

    fn make_image_with_metadata(id: i64, meta: serde_json::Value) -> Image {
        use std::path::PathBuf;
        Image {
            id,
            dataset_id: 1,
            file_name: format!("img{}.jpg", id),
            file_path: PathBuf::from(format!("/tmp/img{}.jpg", id)),
            width: None,
            height: None,
            hash: None,
            metadata: Some(meta),
        }
    }

    #[test]
    fn test_rename_column_schema() {
        let schema = make_schema();
        let ops = vec![SchemaOp::RenameColumn {
            from: "label".to_string(),
            to: "class".to_string(),
        }];
        let evolved = SchemaTransformer::apply(&schema, &ops).expect("apply rename");

        assert!(
            evolved.columns.iter().all(|c| c.name != "label"),
            "old column 'label' should be absent"
        );
        assert!(
            evolved.columns.iter().any(|c| c.name == "class"),
            "new column 'class' should be present"
        );
        assert_eq!(evolved.columns.len(), 3);
    }

    #[test]
    fn test_add_column_schema() {
        let schema = make_schema();
        let ops = vec![SchemaOp::AddColumn {
            name: "split".to_string(),
            dtype: "String".to_string(),
            default: serde_json::json!("train"),
        }];
        let evolved = SchemaTransformer::apply(&schema, &ops).expect("apply add");

        let col = evolved
            .columns
            .iter()
            .find(|c| c.name == "split")
            .expect("split column should exist");
        assert_eq!(col.dtype, crate::schema::DataType::String);
        assert!(col.default.is_some(), "default should be set");
        assert_eq!(evolved.columns.len(), 4);
    }

    #[test]
    fn test_remove_column_schema() {
        let schema = make_schema();
        let ops = vec![SchemaOp::RemoveColumn("confidence".to_string())];
        let evolved = SchemaTransformer::apply(&schema, &ops).expect("apply remove");

        assert!(
            evolved.columns.iter().all(|c| c.name != "confidence"),
            "removed column should be absent"
        );
        assert_eq!(evolved.columns.len(), 2);
    }

    #[test]
    fn test_cast_column_schema() {
        let schema = make_schema();
        let ops = vec![SchemaOp::CastColumn {
            name: "count".to_string(),
            dtype: "Float".to_string(),
        }];
        let evolved = SchemaTransformer::apply(&schema, &ops).expect("apply cast");

        let col = evolved
            .columns
            .iter()
            .find(|c| c.name == "count")
            .expect("count column");
        assert_eq!(col.dtype, crate::schema::DataType::Float);
        assert_eq!(evolved.columns.len(), 3);
    }

    #[test]
    fn test_chain_of_ops_schema() {
        let schema = make_schema();
        let ops = vec![
            SchemaOp::RenameColumn {
                from: "label".to_string(),
                to: "class_name".to_string(),
            },
            SchemaOp::AddColumn {
                name: "split".to_string(),
                dtype: "String".to_string(),
                default: serde_json::json!("train"),
            },
            SchemaOp::RemoveColumn("confidence".to_string()),
        ];
        let evolved = SchemaTransformer::apply(&schema, &ops).expect("apply chain");

        assert!(evolved.columns.iter().any(|c| c.name == "class_name"));
        assert!(evolved.columns.iter().all(|c| c.name != "label"));
        assert!(evolved.columns.iter().any(|c| c.name == "split"));
        assert!(evolved.columns.iter().all(|c| c.name != "confidence"));
        assert!(evolved.columns.iter().any(|c| c.name == "count"));
        assert_eq!(evolved.columns.len(), 3);
    }

    #[test]
    fn test_invalid_column_rename_returns_error() {
        let schema = make_schema();
        let ops = vec![SchemaOp::RenameColumn {
            from: "nonexistent".to_string(),
            to: "new_name".to_string(),
        }];
        let err = SchemaTransformer::apply(&schema, &ops).expect_err("should fail");
        assert!(
            matches!(err, DmanError::SchemaValidation(ref msg) if msg.contains("nonexistent")),
            "error should mention the missing column: {:?}",
            err
        );
    }

    #[test]
    fn test_invalid_column_remove_returns_error() {
        let schema = make_schema();
        let ops = vec![SchemaOp::RemoveColumn("ghost".to_string())];
        let err = SchemaTransformer::apply(&schema, &ops).expect_err("should fail");
        assert!(
            matches!(err, DmanError::SchemaValidation(ref msg) if msg.contains("ghost")),
            "error should mention the missing column: {:?}",
            err
        );
    }

    #[test]
    fn test_invalid_column_cast_returns_error() {
        let schema = make_schema();
        let ops = vec![SchemaOp::CastColumn {
            name: "missing_col".to_string(),
            dtype: "Int".to_string(),
        }];
        let err = SchemaTransformer::apply(&schema, &ops).expect_err("should fail");
        assert!(matches!(err, DmanError::SchemaValidation(_)));
    }

    #[test]
    fn test_add_duplicate_column_returns_error() {
        let schema = make_schema();
        let ops = vec![SchemaOp::AddColumn {
            name: "label".to_string(),
            dtype: "String".to_string(),
            default: serde_json::json!(null),
        }];
        let err = SchemaTransformer::apply(&schema, &ops).expect_err("should fail");
        assert!(
            matches!(err, DmanError::SchemaValidation(ref msg) if msg.contains("label")),
            "error should mention the duplicate column: {:?}",
            err
        );
    }

    #[test]
    fn test_rename_column_images() {
        let images = vec![make_image_with_metadata(
            1,
            serde_json::json!({"label": "cat", "confidence": 0.9}),
        )];
        let ops = vec![SchemaOp::RenameColumn {
            from: "label".to_string(),
            to: "class".to_string(),
        }];
        let result = SchemaTransformer::apply_to_images(&images, &ops).expect("apply to images");

        let meta = result[0].metadata.as_ref().expect("metadata");
        assert!(meta.get("label").is_none());
        assert_eq!(meta["class"], serde_json::json!("cat"));
    }

    #[test]
    fn test_add_column_images_with_non_null_default() {
        let images = vec![make_image_with_metadata(
            1,
            serde_json::json!({"label": "dog"}),
        )];
        let ops = vec![SchemaOp::AddColumn {
            name: "split".to_string(),
            dtype: "String".to_string(),
            default: serde_json::json!("train"),
        }];
        let result = SchemaTransformer::apply_to_images(&images, &ops).expect("apply");

        let meta = result[0].metadata.as_ref().expect("metadata");
        assert_eq!(meta["split"], serde_json::json!("train"));
    }

    #[test]
    fn test_add_column_images_with_null_default_does_not_insert() {
        let images = vec![make_image_with_metadata(
            1,
            serde_json::json!({"label": "dog"}),
        )];
        let ops = vec![SchemaOp::AddColumn {
            name: "optional_col".to_string(),
            dtype: "String".to_string(),
            default: serde_json::json!(null),
        }];
        let result = SchemaTransformer::apply_to_images(&images, &ops).expect("apply");

        let meta = result[0].metadata.as_ref().expect("metadata");
        assert!(meta.get("optional_col").is_none());
    }

    #[test]
    fn test_remove_column_images() {
        let images = vec![make_image_with_metadata(
            1,
            serde_json::json!({"label": "cat", "confidence": 0.9}),
        )];
        let ops = vec![SchemaOp::RemoveColumn("confidence".to_string())];
        let result = SchemaTransformer::apply_to_images(&images, &ops).expect("apply");

        let meta = result[0].metadata.as_ref().expect("metadata");
        assert!(meta.get("confidence").is_none());
        assert!(meta.get("label").is_some());
    }

    #[test]
    fn test_cast_column_images_no_change() {
        let original_meta = serde_json::json!({"count": 5});
        let images = vec![make_image_with_metadata(1, original_meta.clone())];
        let ops = vec![SchemaOp::CastColumn {
            name: "count".to_string(),
            dtype: "Float".to_string(),
        }];
        let result = SchemaTransformer::apply_to_images(&images, &ops).expect("apply");

        assert_eq!(result[0].metadata, Some(original_meta));
    }

    #[test]
    fn test_images_without_metadata_add_column_with_default() {
        use std::path::PathBuf;
        let images = vec![Image {
            id: 1,
            dataset_id: 1,
            file_name: "img1.jpg".to_string(),
            file_path: PathBuf::from("/tmp/img1.jpg"),
            width: None,
            height: None,
            hash: None,
            metadata: None,
        }];
        let ops = vec![SchemaOp::AddColumn {
            name: "split".to_string(),
            dtype: "String".to_string(),
            default: serde_json::json!("val"),
        }];
        let result = SchemaTransformer::apply_to_images(&images, &ops).expect("apply");

        let meta = result[0]
            .metadata
            .as_ref()
            .expect("metadata should be created");
        assert_eq!(meta["split"], serde_json::json!("val"));
    }

    #[test]
    fn test_schema_not_mutated_by_apply() {
        let schema = make_schema();
        let original_len = schema.columns.len();
        let ops = vec![SchemaOp::AddColumn {
            name: "new_col".to_string(),
            dtype: "Bool".to_string(),
            default: serde_json::json!(false),
        }];
        let _ = SchemaTransformer::apply(&schema, &ops).expect("apply");

        assert_eq!(schema.columns.len(), original_len);
    }
}
