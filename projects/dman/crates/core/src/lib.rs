pub mod catalog;
pub mod config;
pub mod dataset;
pub mod db;
pub mod embeddings;
pub mod error;
pub mod formats;
pub mod ops;
pub mod patches;
pub mod predictions;
pub mod schema;
pub mod storage;
pub mod types;
pub mod virtual_dataset;

pub use error::{DmanError, Result};
pub use types::{
    Annotation, Asset, AssetType, BBox, Category, Dataset, DatasetFormat, Embedding, FilterOp,
    Patch, Prediction, Sample, SchemaOp, VirtualDataset, VirtualDatasetDef,
};

pub fn placeholder() {}
