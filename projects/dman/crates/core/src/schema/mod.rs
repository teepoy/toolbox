use std::fmt;
use std::path::Path;

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::Result;

// ---------------------------------------------------------------------------
// DataType
// ---------------------------------------------------------------------------

/// Column data types supported by dman schemas.
///
/// Serializes / deserializes from TOML strings such as `"String"`, `"Int"`,
/// `"EmbeddingVector(128)"`, etc.  The comparison is **case-insensitive** so
/// that `"string"` and `"String"` are both accepted.
#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    String,
    Int,
    Float,
    Bool,
    BBox,
    Polygon,
    Keypoints,
    /// Fixed-length embedding vector, e.g. `EmbeddingVector(128)`.
    EmbeddingVector(usize),
    ImagePath,
    Json,
    List(Box<DataType>),
}

impl DataType {
    /// Parse from a string representation.
    fn from_str_repr(s: &str) -> std::result::Result<Self, std::string::String> {
        let trimmed = s.trim();

        // EmbeddingVector(N)
        if let Some(inner) = trimmed
            .strip_prefix("EmbeddingVector(")
            .or_else(|| trimmed.strip_prefix("embeddingvector("))
        {
            let n_str = inner
                .strip_suffix(')')
                .ok_or_else(|| format!("malformed EmbeddingVector type: '{s}'"))?;
            let n: usize = n_str.parse().map_err(|_| {
                format!("EmbeddingVector dimension is not a valid integer: '{n_str}'")
            })?;
            return Ok(DataType::EmbeddingVector(n));
        }

        // List(InnerType)
        if let Some(inner) = trimmed
            .strip_prefix("List(")
            .or_else(|| trimmed.strip_prefix("list("))
        {
            let inner_str = inner
                .strip_suffix(')')
                .ok_or_else(|| format!("malformed List type: '{s}'"))?;
            let inner_dt = DataType::from_str_repr(inner_str)?;
            return Ok(DataType::List(Box::new(inner_dt)));
        }

        match trimmed.to_ascii_lowercase().as_str() {
            "string" | "str" => Ok(DataType::String),
            "int" | "integer" => Ok(DataType::Int),
            "float" | "f64" | "f32" => Ok(DataType::Float),
            "bool" | "boolean" => Ok(DataType::Bool),
            "bbox" | "boundingbox" => Ok(DataType::BBox),
            "polygon" => Ok(DataType::Polygon),
            "keypoints" => Ok(DataType::Keypoints),
            "imagepath" | "image_path" | "image" => Ok(DataType::ImagePath),
            "json" => Ok(DataType::Json),
            _ => Err(format!("unknown dtype: '{s}'")),
        }
    }

    /// Return a canonical string representation (used for serialization).
    fn to_str_repr(&self) -> std::string::String {
        match self {
            DataType::String => "String".to_owned(),
            DataType::Int => "Int".to_owned(),
            DataType::Float => "Float".to_owned(),
            DataType::Bool => "Bool".to_owned(),
            DataType::BBox => "BBox".to_owned(),
            DataType::Polygon => "Polygon".to_owned(),
            DataType::Keypoints => "Keypoints".to_owned(),
            DataType::EmbeddingVector(n) => format!("EmbeddingVector({n})"),
            DataType::ImagePath => "ImagePath".to_owned(),
            DataType::Json => "Json".to_owned(),
            DataType::List(inner) => format!("List({})", inner.to_str_repr()),
        }
    }
}

// --- Custom Serde for DataType ---

impl Serialize for DataType {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_str_repr())
    }
}

struct DataTypeVisitor;

impl<'de> Visitor<'de> for DataTypeVisitor {
    type Value = DataType;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a string like \"String\", \"Int\", \"BBox\", \"EmbeddingVector(128)\"")
    }

    fn visit_str<E: de::Error>(self, value: &str) -> std::result::Result<Self::Value, E> {
        DataType::from_str_repr(value).map_err(de::Error::custom)
    }
}

impl<'de> Deserialize<'de> for DataType {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        deserializer.deserialize_str(DataTypeVisitor)
    }
}

// ---------------------------------------------------------------------------
// AnnotationFormat
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnnotationFormat {
    BoundingBox,
    Polygon,
    Keypoints,
    Custom(std::string::String),
}

// ---------------------------------------------------------------------------
// ValidationError
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationError {
    pub column: std::string::String,
    pub message: std::string::String,
}

impl ValidationError {
    pub fn new(
        column: impl Into<std::string::String>,
        message: impl Into<std::string::String>,
    ) -> Self {
        ValidationError {
            column: column.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.column, self.message)
    }
}

// ---------------------------------------------------------------------------
// ColumnDef
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnDef {
    pub name: std::string::String,
    pub dtype: DataType,
    pub required: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<toml::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<std::string::String>,
}

// ---------------------------------------------------------------------------
// SchemaDefinition — top-level struct
//
// The fixture uses `[[fields]]` while the task spec uses `[[columns]]`.
// We support both by accepting whichever key is present (prefer `columns`,
// fall back to `fields`).
// ---------------------------------------------------------------------------

/// Internal raw TOML representation to allow either `columns` or `fields` key.
#[derive(Debug, Deserialize)]
struct RawSchema {
    name: std::string::String,
    version: std::string::String,
    /// `[[columns]]` form (matches task spec example).
    #[serde(default)]
    columns: Vec<ColumnDef>,
    /// `[[fields]]` form (matches the actual fixture file).
    #[serde(default)]
    fields: Vec<ColumnDef>,
    #[serde(rename = "annotation_format")]
    annotation_format: Option<AnnotationFormat>,
}

impl RawSchema {
    fn into_schema_definition(mut self) -> SchemaDefinition {
        // Prefer columns; fall back to fields if columns is empty.
        let columns = if !self.columns.is_empty() {
            self.columns
        } else {
            self.columns.append(&mut self.fields);
            self.columns
        };

        SchemaDefinition {
            name: self.name,
            version: self.version,
            columns,
            annotation_format: self.annotation_format,
        }
    }
}

/// Optional wrapper — some files may nest everything under `[schema]`.
#[derive(Debug, Deserialize)]
struct WrappedSchema {
    schema: RawSchema,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaDefinition {
    pub name: std::string::String,
    pub version: std::string::String,
    pub columns: Vec<ColumnDef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotation_format: Option<AnnotationFormat>,
}

// ---------------------------------------------------------------------------
// Schema — parse + validate
// ---------------------------------------------------------------------------

pub struct Schema;

impl Schema {
    /// Parse a TOML file at `path` into a `SchemaDefinition`.
    pub fn from_toml(path: &Path) -> Result<SchemaDefinition> {
        let content = std::fs::read_to_string(path)?;
        Schema::parse_toml(&content)
    }

    /// Parse a TOML string into a `SchemaDefinition`.
    pub fn parse_toml(s: &str) -> Result<SchemaDefinition> {
        // Try flat form first (name/version at top-level).
        if let Ok(raw) = toml::from_str::<RawSchema>(s) {
            return Ok(raw.into_schema_definition());
        }
        // Fall back to wrapped `[schema]` form.
        let wrapped: WrappedSchema = toml::from_str(s)?;
        Ok(wrapped.schema.into_schema_definition())
    }

    /// Validate a JSON row object against the schema.
    ///
    /// Returns a list of `ValidationError`s — an empty list means the row is valid.
    pub fn validate_row(
        schema: &SchemaDefinition,
        row: &serde_json::Value,
    ) -> Result<Vec<ValidationError>> {
        let mut errors = Vec::new();

        for col in &schema.columns {
            match row.get(&col.name) {
                None => {
                    // Missing value
                    if col.required && col.default.is_none() {
                        errors.push(ValidationError::new(&col.name, "required field is missing"));
                    }
                    // If not required or has a default, absence is fine — skip type check.
                }
                Some(val) => {
                    // Present — run type check unless value is null and not required.
                    if val.is_null() {
                        if col.required {
                            errors.push(ValidationError::new(&col.name, "required field is null"));
                        }
                    } else if let Some(err_msg) = check_type(val, &col.dtype) {
                        errors.push(ValidationError::new(&col.name, err_msg));
                    }
                }
            }
        }

        Ok(errors)
    }
}

/// Returns `Some(error_message)` if `val` does not match `dtype`, `None` if ok.
fn check_type(val: &serde_json::Value, dtype: &DataType) -> Option<std::string::String> {
    match dtype {
        DataType::String | DataType::ImagePath => {
            if !val.is_string() {
                return Some(format!("expected string, got {}", json_type_name(val)));
            }
        }
        DataType::Int => {
            if let Some(n) = val.as_f64() {
                if n.fract() != 0.0 {
                    return Some(format!("expected integer (no decimal), got float {n}"));
                }
            } else {
                return Some(format!("expected integer, got {}", json_type_name(val)));
            }
        }
        DataType::Float => {
            if !val.is_f64() && !val.is_i64() && !val.is_u64() {
                return Some(format!(
                    "expected float/number, got {}",
                    json_type_name(val)
                ));
            }
        }
        DataType::Bool => {
            if !val.is_boolean() {
                return Some(format!("expected bool, got {}", json_type_name(val)));
            }
        }
        DataType::BBox => {
            // Accept an object with x/y/width/height, or an array of 4 numbers.
            if !is_valid_bbox(val) {
                return Some(format!(
                    "expected bbox (object with x/y/width/height or array[4]), got {}",
                    json_type_name(val)
                ));
            }
        }
        DataType::Polygon => {
            // Array of [x, y] pairs
            if !is_valid_polygon(val) {
                return Some(format!(
                    "expected polygon (array of [x, y] pairs), got {}",
                    json_type_name(val)
                ));
            }
        }
        DataType::Keypoints => {
            // Array of numbers
            if !val.is_array() {
                return Some(format!(
                    "expected keypoints (array of numbers), got {}",
                    json_type_name(val)
                ));
            }
        }
        DataType::EmbeddingVector(n) => match val.as_array() {
            None => {
                return Some(format!(
                    "expected embedding vector (array of {n} numbers), got {}",
                    json_type_name(val)
                ));
            }
            Some(arr) => {
                if arr.len() != *n {
                    return Some(format!(
                        "expected embedding vector of length {n}, got length {}",
                        arr.len()
                    ));
                }
                if arr
                    .iter()
                    .any(|v| !v.is_f64() && !v.is_i64() && !v.is_u64())
                {
                    return Some("embedding vector must contain numbers only".to_string());
                }
            }
        },
        DataType::Json => {
            // Any JSON value is acceptable.
        }
        DataType::List(_inner) => {
            if !val.is_array() {
                return Some(format!(
                    "expected list (array), got {}",
                    json_type_name(val)
                ));
            }
            // Could recurse over elements, but basic check is sufficient here.
        }
    }
    None
}

fn json_type_name(val: &serde_json::Value) -> &'static str {
    match val {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

fn is_valid_bbox(val: &serde_json::Value) -> bool {
    match val {
        serde_json::Value::Object(map) => ["x", "y", "width", "height"]
            .iter()
            .all(|k| map.contains_key(*k)),
        serde_json::Value::Array(arr) => {
            arr.len() == 4 && arr.iter().all(|v| v.is_f64() || v.is_i64() || v.is_u64())
        }
        _ => false,
    }
}

fn is_valid_polygon(val: &serde_json::Value) -> bool {
    match val {
        serde_json::Value::Array(arr) => arr.iter().all(|v| match v {
            serde_json::Value::Array(pair) => {
                pair.len() == 2 && pair.iter().all(|n| n.is_f64() || n.is_i64() || n.is_u64())
            }
            _ => false,
        }),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::DmanError;

    // -----------------------------------------------------------------------
    // DataType parsing
    // -----------------------------------------------------------------------

    #[test]
    fn test_dtype_from_str_basic() {
        assert_eq!(DataType::from_str_repr("String").unwrap(), DataType::String);
        assert_eq!(DataType::from_str_repr("string").unwrap(), DataType::String);
        assert_eq!(DataType::from_str_repr("Int").unwrap(), DataType::Int);
        assert_eq!(DataType::from_str_repr("int").unwrap(), DataType::Int);
        assert_eq!(DataType::from_str_repr("Float").unwrap(), DataType::Float);
        assert_eq!(DataType::from_str_repr("float").unwrap(), DataType::Float);
        assert_eq!(DataType::from_str_repr("Bool").unwrap(), DataType::Bool);
        assert_eq!(DataType::from_str_repr("bool").unwrap(), DataType::Bool);
        assert_eq!(DataType::from_str_repr("BBox").unwrap(), DataType::BBox);
        assert_eq!(DataType::from_str_repr("bbox").unwrap(), DataType::BBox);
        assert_eq!(
            DataType::from_str_repr("Polygon").unwrap(),
            DataType::Polygon
        );
        assert_eq!(
            DataType::from_str_repr("Keypoints").unwrap(),
            DataType::Keypoints
        );
        assert_eq!(
            DataType::from_str_repr("ImagePath").unwrap(),
            DataType::ImagePath
        );
        assert_eq!(DataType::from_str_repr("Json").unwrap(), DataType::Json);
    }

    #[test]
    fn test_dtype_embedding_vector() {
        assert_eq!(
            DataType::from_str_repr("EmbeddingVector(128)").unwrap(),
            DataType::EmbeddingVector(128)
        );
        assert_eq!(
            DataType::from_str_repr("EmbeddingVector(512)").unwrap(),
            DataType::EmbeddingVector(512)
        );
    }

    #[test]
    fn test_dtype_list() {
        assert_eq!(
            DataType::from_str_repr("List(String)").unwrap(),
            DataType::List(Box::new(DataType::String))
        );
        assert_eq!(
            DataType::from_str_repr("List(Float)").unwrap(),
            DataType::List(Box::new(DataType::Float))
        );
    }

    #[test]
    fn test_dtype_unknown_returns_error() {
        assert!(DataType::from_str_repr("NotAType").is_err());
    }

    #[test]
    fn test_dtype_serde_roundtrip() {
        let dt = DataType::EmbeddingVector(256);
        let s = serde_json::to_string(&dt).unwrap();
        let back: DataType = serde_json::from_str(&s).unwrap();
        assert_eq!(dt, back);
    }

    // -----------------------------------------------------------------------
    // Schema::from_str — inline TOML
    // -----------------------------------------------------------------------

    #[test]
    fn test_schema_from_str_fields_form() {
        let toml = r#"
name = "basic-schema"
version = "1"

[[fields]]
name = "label"
dtype = "string"
required = true

[[fields]]
name = "confidence"
dtype = "float"
required = false
default = 1.0
"#;
        let schema = Schema::parse_toml(toml).expect("should parse");
        assert_eq!(schema.name, "basic-schema");
        assert_eq!(schema.version, "1");
        assert_eq!(schema.columns.len(), 2);
        assert_eq!(schema.columns[0].name, "label");
        assert_eq!(schema.columns[0].dtype, DataType::String);
        assert!(schema.columns[0].required);
        assert_eq!(schema.columns[1].name, "confidence");
        assert_eq!(schema.columns[1].dtype, DataType::Float);
        assert!(!schema.columns[1].required);
        // default is present
        assert!(schema.columns[1].default.is_some());
    }

    #[test]
    fn test_schema_from_str_columns_form() {
        let toml = r#"
name = "object-detection"
version = "1.0"

[[columns]]
name = "image_path"
dtype = "ImagePath"
required = true

[[columns]]
name = "bbox"
dtype = "BBox"
required = true

[[columns]]
name = "label"
dtype = "String"
required = true

[[columns]]
name = "confidence"
dtype = "Float"
required = false
default = 1.0
"#;
        let schema = Schema::parse_toml(toml).expect("should parse");
        assert_eq!(schema.name, "object-detection");
        assert_eq!(schema.columns.len(), 4);
        assert_eq!(schema.columns[0].dtype, DataType::ImagePath);
        assert_eq!(schema.columns[1].dtype, DataType::BBox);
    }

    #[test]
    fn test_schema_from_str_wrapped_form() {
        let toml = r#"
[schema]
name = "wrapped"
version = "2.0"

[[schema.columns]]
name = "embedding"
dtype = "EmbeddingVector(128)"
required = true
"#;
        let schema = Schema::parse_toml(toml).expect("should parse wrapped form");
        assert_eq!(schema.name, "wrapped");
        assert_eq!(schema.columns[0].dtype, DataType::EmbeddingVector(128));
    }

    #[test]
    fn test_schema_from_str_invalid_toml_returns_error() {
        let bad = "this is not toml :::";
        let result = Schema::parse_toml(bad);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), DmanError::Toml(_)));
    }

    // -----------------------------------------------------------------------
    // Schema::from_toml — fixture file
    // -----------------------------------------------------------------------

    #[test]
    fn test_schema_from_toml_fixture() {
        let fixture = std::path::Path::new("tests/fixtures/schema/basic.toml");
        let schema = Schema::from_toml(fixture).expect("should parse fixture");
        assert_eq!(schema.name, "basic-schema");
        assert_eq!(schema.version, "1");
        assert_eq!(schema.columns.len(), 4);

        // label
        assert_eq!(schema.columns[0].name, "label");
        assert_eq!(schema.columns[0].dtype, DataType::String);
        assert!(schema.columns[0].required);

        // bbox
        assert_eq!(schema.columns[1].name, "bbox");
        assert_eq!(schema.columns[1].dtype, DataType::BBox);
        assert!(!schema.columns[1].required);

        // confidence
        assert_eq!(schema.columns[2].name, "confidence");
        assert_eq!(schema.columns[2].dtype, DataType::Float);
        assert!(!schema.columns[2].required);
        assert!(schema.columns[2].default.is_some());

        // split
        assert_eq!(schema.columns[3].name, "split");
        assert_eq!(schema.columns[3].dtype, DataType::String);
    }

    // -----------------------------------------------------------------------
    // Schema::validate_row
    // -----------------------------------------------------------------------

    fn make_test_schema() -> SchemaDefinition {
        Schema::parse_toml(
            r#"
name = "test"
version = "1"

[[columns]]
name = "label"
dtype = "String"
required = true

[[columns]]
name = "score"
dtype = "Float"
required = false

[[columns]]
name = "count"
dtype = "Int"
required = false

[[columns]]
name = "active"
dtype = "Bool"
required = true

[[columns]]
name = "embedding"
dtype = "EmbeddingVector(3)"
required = false
"#,
        )
        .unwrap()
    }

    #[test]
    fn test_validate_row_valid() {
        let schema = make_test_schema();
        let row = serde_json::json!({
            "label": "cat",
            "score": 0.95,
            "count": 3,
            "active": true,
        });
        let errors = Schema::validate_row(&schema, &row).unwrap();
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
    }

    #[test]
    fn test_validate_row_missing_required_field() {
        let schema = make_test_schema();
        // missing "label" and "active"
        let row = serde_json::json!({ "score": 0.5 });
        let errors = Schema::validate_row(&schema, &row).unwrap();
        let names: Vec<&str> = errors.iter().map(|e| e.column.as_str()).collect();
        assert!(
            names.contains(&"label"),
            "expected 'label' error, got {names:?}"
        );
        assert!(
            names.contains(&"active"),
            "expected 'active' error, got {names:?}"
        );
    }

    #[test]
    fn test_validate_row_type_mismatch_string() {
        let schema = make_test_schema();
        let row = serde_json::json!({
            "label": 123,   // should be string
            "active": true,
        });
        let errors = Schema::validate_row(&schema, &row).unwrap();
        assert!(
            errors.iter().any(|e| e.column == "label"),
            "expected type error for 'label', got {errors:?}"
        );
    }

    #[test]
    fn test_validate_row_type_mismatch_bool() {
        let schema = make_test_schema();
        let row = serde_json::json!({
            "label": "x",
            "active": "yes",   // should be bool
        });
        let errors = Schema::validate_row(&schema, &row).unwrap();
        assert!(
            errors.iter().any(|e| e.column == "active"),
            "expected type error for 'active', got {errors:?}"
        );
    }

    #[test]
    fn test_validate_row_int_rejects_float() {
        let schema = make_test_schema();
        let row = serde_json::json!({
            "label": "x",
            "active": true,
            "count": 3.5,   // float for Int column
        });
        let errors = Schema::validate_row(&schema, &row).unwrap();
        assert!(
            errors.iter().any(|e| e.column == "count"),
            "expected type error for 'count', got {errors:?}"
        );
    }

    #[test]
    fn test_validate_row_embedding_vector_wrong_length() {
        let schema = make_test_schema();
        let row = serde_json::json!({
            "label": "x",
            "active": true,
            "embedding": [1.0, 2.0],  // needs 3 elements
        });
        let errors = Schema::validate_row(&schema, &row).unwrap();
        assert!(
            errors.iter().any(|e| e.column == "embedding"),
            "expected embedding error, got {errors:?}"
        );
    }

    #[test]
    fn test_validate_row_embedding_vector_correct() {
        let schema = make_test_schema();
        let row = serde_json::json!({
            "label": "x",
            "active": true,
            "embedding": [1.0, 2.0, 3.0],
        });
        let errors = Schema::validate_row(&schema, &row).unwrap();
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
    }

    #[test]
    fn test_validate_row_null_required_field() {
        let schema = make_test_schema();
        let row = serde_json::json!({
            "label": null,  // null but required
            "active": true,
        });
        let errors = Schema::validate_row(&schema, &row).unwrap();
        assert!(
            errors.iter().any(|e| e.column == "label"),
            "expected null error for 'label', got {errors:?}"
        );
    }

    #[test]
    fn test_validate_row_optional_field_absent_is_ok() {
        let schema = make_test_schema();
        // score and count are optional, absent is fine
        let row = serde_json::json!({
            "label": "dog",
            "active": false,
        });
        let errors = Schema::validate_row(&schema, &row).unwrap();
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
    }
}
