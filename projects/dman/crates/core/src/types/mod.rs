use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(transparent)]
pub struct DatasetFormat(String);

impl DatasetFormat {
    pub const YOLO: &'static str = "yolo";
    pub const COCO: &'static str = "coco";
    pub const HUGGINGFACE: &'static str = "huggingface";
    pub const LABEL_STUDIO: &'static str = "label-studio";
    pub const BUILDER: &'static str = "builder";

    pub fn new(value: impl AsRef<str>) -> Self {
        Self(Self::normalize(value.as_ref()))
    }

    pub fn yolo() -> Self {
        Self::new(Self::YOLO)
    }

    pub fn coco() -> Self {
        Self::new(Self::COCO)
    }

    pub fn huggingface() -> Self {
        Self::new(Self::HUGGINGFACE)
    }

    pub fn label_studio() -> Self {
        Self::new(Self::LABEL_STUDIO)
    }

    pub fn builder() -> Self {
        Self::new(Self::BUILDER)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn is_builtin(&self) -> bool {
        matches!(self.as_str(), Self::YOLO | Self::COCO | Self::HUGGINGFACE)
    }

    pub fn display_name(&self) -> &str {
        self.as_str()
    }

    pub fn normalize(value: &str) -> String {
        let trimmed = value.trim();
        let lowered = trimmed.to_ascii_lowercase();
        match lowered.as_str() {
            "yolo" => Self::YOLO.to_string(),
            "coco" => Self::COCO.to_string(),
            "hf" | "huggingface" | "hugging-face" => Self::HUGGINGFACE.to_string(),
            "labelstudio" | "label-studio" => Self::LABEL_STUDIO.to_string(),
            other => other.to_string(),
        }
    }
}

impl fmt::Display for DatasetFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<&str> for DatasetFormat {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for DatasetFormat {
    fn from(value: String) -> Self {
        Self::new(value)
    }
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AssetType {
    Image,
    DepthMap,
    PointCloud,
    Text,
    Audio,
    Video,
    Mask,
    Other(String),
}

impl fmt::Display for AssetType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AssetType::Image => write!(f, "image"),
            AssetType::DepthMap => write!(f, "depth_map"),
            AssetType::PointCloud => write!(f, "point_cloud"),
            AssetType::Text => write!(f, "text"),
            AssetType::Audio => write!(f, "audio"),
            AssetType::Video => write!(f, "video"),
            AssetType::Mask => write!(f, "mask"),
            AssetType::Other(s) => write!(f, "{}", s),
        }
    }
}

impl std::str::FromStr for AssetType {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "image" => AssetType::Image,
            "depth_map" => AssetType::DepthMap,
            "point_cloud" => AssetType::PointCloud,
            "text" => AssetType::Text,
            "audio" => AssetType::Audio,
            "video" => AssetType::Video,
            "mask" => AssetType::Mask,
            other => AssetType::Other(other.to_string()),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sample {
    pub id: i64,
    pub dataset_id: i64,
    pub name: String,
    pub metadata: Option<serde_json::Value>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Asset {
    pub id: i64,
    pub sample_id: i64,
    pub asset_type: AssetType,
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
    pub sample_id: i64,
    pub asset_id: Option<i64>,
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
    pub asset_id: i64,
    pub model_name: String,
    pub vector: Vec<f32>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub id: i64,
    pub sample_id: i64,
    pub asset_id: Option<i64>,
    pub model_version: String,
    pub result: serde_json::Value,
    pub score: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Patch {
    pub id: i64,
    pub asset_id: i64,
    pub annotation_id: Option<i64>,
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
        let fmt = DatasetFormat::new("labelbox");
        let json = serde_json::to_string(&fmt).expect("serialize");
        let back: DatasetFormat = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(fmt, back);
    }

    #[test]
    fn test_dataset_format_normalizes_aliases() {
        assert_eq!(DatasetFormat::new("Yolo").as_str(), DatasetFormat::YOLO);
        assert_eq!(
            DatasetFormat::new("hf").as_str(),
            DatasetFormat::HUGGINGFACE
        );
        assert_eq!(
            DatasetFormat::new("LabelStudio").as_str(),
            DatasetFormat::LABEL_STUDIO
        );
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
            format: DatasetFormat::yolo(),
            schema_path: None,
            created_at: "2026-01-01".to_string(),
            updated_at: None,
            metadata: None,
        };
        let json = serde_json::to_string(&ds).expect("serialize Dataset");
        let _: Dataset = serde_json::from_str(&json).expect("deserialize Dataset");

        let ann = Annotation {
            id: 1,
            sample_id: 1,
            asset_id: Some(2),
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

    #[test]
    fn test_asset_type_display_and_fromstr() {
        use std::str::FromStr;
        assert_eq!(AssetType::Image.to_string(), "image");
        assert_eq!(AssetType::DepthMap.to_string(), "depth_map");
        assert_eq!(AssetType::Other("lidar".to_string()).to_string(), "lidar");
        assert_eq!(AssetType::from_str("image").unwrap(), AssetType::Image);
        assert_eq!(
            AssetType::from_str("depth_map").unwrap(),
            AssetType::DepthMap
        );
        assert_eq!(
            AssetType::from_str("lidar").unwrap(),
            AssetType::Other("lidar".to_string())
        );
    }

    #[test]
    fn test_sample_serde_roundtrip() {
        let s = Sample {
            id: 1,
            dataset_id: 42,
            name: "sample-001".to_string(),
            metadata: Some(serde_json::json!({"key": "value"})),
            created_at: "2026-01-01T00:00:00Z".to_string(),
        };
        let json = serde_json::to_string(&s).unwrap();
        let s2: Sample = serde_json::from_str(&json).unwrap();
        assert_eq!(s.id, s2.id);
        assert_eq!(s.name, s2.name);
    }

    #[test]
    fn test_asset_serde_roundtrip() {
        let a = Asset {
            id: 1,
            sample_id: 10,
            asset_type: AssetType::Image,
            file_name: "frame.jpg".to_string(),
            file_path: std::path::PathBuf::from("/tmp/frame.jpg"),
            width: Some(640),
            height: Some(480),
            hash: Some("abc123".to_string()),
            metadata: None,
        };
        let json = serde_json::to_string(&a).unwrap();
        let a2: Asset = serde_json::from_str(&json).unwrap();
        assert_eq!(a.id, a2.id);
        assert_eq!(a.asset_type, a2.asset_type);
    }
}
