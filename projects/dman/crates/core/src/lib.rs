pub mod catalog;
pub mod config;
pub mod dataset;
pub mod db;
pub mod embeddings;
pub mod error;
pub mod formats;
pub mod ops;
pub mod predictions;
pub mod schema;
pub mod storage;
pub mod types;
pub mod virtual_dataset;

pub use error::{DmanError, Result};
pub use types::{
    Annotation, BBox, Category, Dataset, DatasetFormat, Embedding, FilterOp, Image, Patch,
    Prediction, SchemaOp, VirtualDataset, VirtualDatasetDef,
};

pub fn placeholder() {}
