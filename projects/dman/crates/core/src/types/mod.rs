use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DatasetFormat {
    Yolo,
    Coco,
    HuggingFace,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub id: i64,
    pub name: String,
    pub path: PathBuf,
    pub format: DatasetFormat,
    pub schema_path: Option<PathBuf>,
    pub created_at: String,
    pub updated_at: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BBox {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Image {
    pub id: i64,
    pub dataset_id: i64,
    pub file_name: String,
    pub file_path: PathBuf,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub hash: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Annotation {
    pub id: i64,
    pub image_id: i64,
    pub category_id: Option<i64>,
    pub bbox: Option<BBox>,
    pub segmentation: Option<Vec<Vec<f64>>>,
    pub keypoints: Option<Vec<f64>>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Category {
    pub id: i64,
    pub dataset_id: i64,
    pub name: String,
    pub supercategory: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    pub id: i64,
    pub image_id: i64,
    pub model_name: String,
    pub vector: Vec<f32>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub id: i64,
    pub image_id: i64,
    pub model_version: String,
    pub result: serde_json::Value,
    pub score: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Patch {
    pub id: i64,
    pub image_id: i64,
    pub bbox: BBox,
    pub file_path: Option<PathBuf>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FilterOp {
    Eq,
    Ne,
    Gt,
    Lt,
    Gte,
    Lte,
    Contains,
    In,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchemaOp {
    RenameColumn {
        from: String,
        to: String,
    },
    AddColumn {
        name: String,
        dtype: String,
        default: serde_json::Value,
    },
    RemoveColumn(String),
    CastColumn {
        name: String,
        dtype: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VirtualDatasetDef {
    Filter {
        column: String,
        op: FilterOp,
        value: serde_json::Value,
    },
    Merge {
        datasets: Vec<i64>,
    },
    Sample {
        ratio: f64,
    },
    Split {
        ratios: HashMap<String, f64>,
    },
    SchemaTransform {
        transforms: Vec<SchemaOp>,
    },
    Chain(Vec<VirtualDatasetDef>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualDataset {
    pub id: i64,
    pub name: String,
    pub source_datasets: Vec<i64>,
    pub definition: VirtualDatasetDef,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bbox_serde_roundtrip() {
        let bbox = BBox {
            x: 10.0,
            y: 20.0,
            width: 100.0,
            height: 80.0,
        };
        let json = serde_json::to_string(&bbox).expect("serialize");
        let back: BBox = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(bbox, back);
    }

    #[test]
    fn test_dataset_format_serde() {
        let fmt = DatasetFormat::Custom("labelbox".to_string());
        let json = serde_json::to_string(&fmt).expect("serialize");
        let back: DatasetFormat = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(fmt, back);
    }

    #[test]
    fn test_virtual_dataset_chain() {
        let chain = VirtualDatasetDef::Chain(vec![
            VirtualDatasetDef::Filter {
                column: "label".to_string(),
                op: FilterOp::Eq,
                value: serde_json::json!("cat"),
            },
            VirtualDatasetDef::Sample { ratio: 0.5 },
            VirtualDatasetDef::SchemaTransform {
                transforms: vec![SchemaOp::AddColumn {
                    name: "split".to_string(),
                    dtype: "string".to_string(),
                    default: serde_json::json!("train"),
                }],
            },
        ]);
        let json = serde_json::to_string(&chain).expect("serialize chain");
        let back: VirtualDatasetDef = serde_json::from_str(&json).expect("deserialize chain");
        let json2 = serde_json::to_string(&back).expect("re-serialize");
        assert_eq!(json, json2);
    }

    #[test]
    fn test_serde_roundtrip_all_types() {
        let ds = Dataset {
            id: 1,
            name: "test".to_string(),
            path: std::path::PathBuf::from("/tmp/test"),
            format: DatasetFormat::Yolo,
            schema_path: None,
            created_at: "2026-01-01".to_string(),
            updated_at: None,
            metadata: None,
        };
        let json = serde_json::to_string(&ds).expect("serialize Dataset");
        let _: Dataset = serde_json::from_str(&json).expect("deserialize Dataset");

        let img = Image {
            id: 1,
            dataset_id: 1,
            file_name: "img001.jpg".to_string(),
            file_path: std::path::PathBuf::from("/tmp/img001.jpg"),
            width: Some(640),
            height: Some(480),
            hash: None,
            metadata: None,
        };
        let json = serde_json::to_string(&img).expect("serialize Image");
        let _: Image = serde_json::from_str(&json).expect("deserialize Image");

        let ann = Annotation {
            id: 1,
            image_id: 1,
            category_id: Some(1),
            bbox: Some(BBox {
                x: 10.0,
                y: 20.0,
                width: 100.0,
                height: 80.0,
            }),
            segmentation: None,
            keypoints: None,
            metadata: None,
        };
        let json = serde_json::to_string(&ann).expect("serialize Annotation");
        let _: Annotation = serde_json::from_str(&json).expect("deserialize Annotation");
    }
}
