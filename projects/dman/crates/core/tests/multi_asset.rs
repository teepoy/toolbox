use std::collections::HashMap;
use std::path::Path;

use dman_core::{
    dataset::DatasetService,
    db::Database,
    ops::DatasetOps,
    types::{AssetType, DatasetFormat},
};
use tempfile::TempDir;

fn in_memory_db() -> Database {
    Database::open_in_memory().expect("in-memory DB")
}

fn register_dataset(db: &Database, name: &str, tmp: &TempDir) -> i64 {
    DatasetService::register(db, name, tmp.path(), DatasetFormat::yolo())
        .expect("register dataset")
        .id
}

#[test]
fn test_multi_asset_sample_creation_and_retrieval() {
    let tmp = TempDir::new().expect("tempdir");
    let db = in_memory_db();
    let ds_id = register_dataset(&db, "stereo-dataset", &tmp);

    let sample_a_id =
        DatasetService::add_sample(&db, ds_id, "frame-001", None).expect("add sample A");

    let left_asset_id = DatasetService::add_asset(
        &db,
        sample_a_id,
        AssetType::Image,
        "frame-001-left.jpg",
        Path::new("/data/left/frame-001.jpg"),
        Some(1920),
        Some(1080),
        Some("hash-left-001"),
        None,
    )
    .expect("add left asset");

    let right_asset_id = DatasetService::add_asset(
        &db,
        sample_a_id,
        AssetType::Image,
        "frame-001-right.jpg",
        Path::new("/data/right/frame-001.jpg"),
        Some(1920),
        Some(1080),
        Some("hash-right-001"),
        None,
    )
    .expect("add right asset");

    let sample_b_id =
        DatasetService::add_sample(&db, ds_id, "frame-002", None).expect("add sample B");

    let depth_asset_id = DatasetService::add_asset(
        &db,
        sample_b_id,
        AssetType::DepthMap,
        "frame-002-depth.png",
        Path::new("/data/depth/frame-002.png"),
        Some(640),
        Some(480),
        None,
        None,
    )
    .expect("add depth asset");

    assert_eq!(
        DatasetService::get_sample_count(&db, ds_id).expect("sample count"),
        2,
        "expected 2 samples"
    );
    assert_eq!(
        DatasetService::get_asset_count(&db, ds_id).expect("asset count"),
        3,
        "expected 3 assets total"
    );

    let samples = DatasetService::get_samples(&db, ds_id).expect("get samples");
    assert_eq!(samples.len(), 2);
    assert_eq!(samples[0].id, sample_a_id);
    assert_eq!(samples[0].name, "frame-001");
    assert_eq!(samples[1].id, sample_b_id);
    assert_eq!(samples[1].name, "frame-002");

    let assets_a = DatasetService::get_assets(&db, sample_a_id).expect("get assets A");
    assert_eq!(assets_a.len(), 2, "sample A should have 2 assets");
    let asset_ids_a: Vec<i64> = assets_a.iter().map(|a| a.id).collect();
    assert!(asset_ids_a.contains(&left_asset_id), "left asset missing");
    assert!(asset_ids_a.contains(&right_asset_id), "right asset missing");

    let left = assets_a
        .iter()
        .find(|a| a.id == left_asset_id)
        .expect("left asset");
    assert_eq!(left.file_name, "frame-001-left.jpg");
    assert_eq!(left.asset_type, AssetType::Image);
    assert_eq!(left.width, Some(1920));
    assert_eq!(left.height, Some(1080));
    assert_eq!(left.hash.as_deref(), Some("hash-left-001"));

    let right = assets_a
        .iter()
        .find(|a| a.id == right_asset_id)
        .expect("right asset");
    assert_eq!(right.file_name, "frame-001-right.jpg");

    let assets_b = DatasetService::get_assets(&db, sample_b_id).expect("get assets B");
    assert_eq!(assets_b.len(), 1, "sample B should have 1 asset");
    assert_eq!(assets_b[0].id, depth_asset_id);
    assert_eq!(assets_b[0].asset_type, AssetType::DepthMap);
    assert_eq!(assets_b[0].file_name, "frame-002-depth.png");
    assert_eq!(assets_b[0].width, Some(640));
    assert_eq!(assets_b[0].height, Some(480));
}

#[test]
fn test_duplicate_preserves_hierarchy() {
    let tmp = TempDir::new().expect("tempdir");
    let db = in_memory_db();
    let ds_id = register_dataset(&db, "original-multi", &tmp);

    let s1 = DatasetService::add_sample(&db, ds_id, "sample-1", None).expect("add s1");
    let a1 = DatasetService::add_asset(
        &db,
        s1,
        AssetType::Image,
        "s1-rgb.jpg",
        Path::new("/data/s1-rgb.jpg"),
        None,
        None,
        None,
        None,
    )
    .expect("add asset a1");
    let a2 = DatasetService::add_asset(
        &db,
        s1,
        AssetType::DepthMap,
        "s1-depth.png",
        Path::new("/data/s1-depth.png"),
        None,
        None,
        None,
        None,
    )
    .expect("add asset a2");
    DatasetService::add_annotation(&db, s1, Some(a1), None, None, None, None, None)
        .expect("annotation on a1");
    DatasetService::add_annotation(&db, s1, Some(a2), None, None, None, None, None)
        .expect("annotation on a2");

    let s2 = DatasetService::add_sample(&db, ds_id, "sample-2", None).expect("add s2");
    let a3 = DatasetService::add_asset(
        &db,
        s2,
        AssetType::Image,
        "s2-rgb.jpg",
        Path::new("/data/s2-rgb.jpg"),
        None,
        None,
        None,
        None,
    )
    .expect("add asset a3");
    DatasetService::add_annotation(&db, s2, Some(a3), None, None, None, None, None)
        .expect("annotation on a3");

    assert_eq!(
        DatasetService::get_sample_count(&db, ds_id).expect("sample count"),
        2
    );
    assert_eq!(
        DatasetService::get_asset_count(&db, ds_id).expect("asset count"),
        3
    );
    assert_eq!(
        DatasetService::get_annotation_count(&db, ds_id).expect("annotation count"),
        3
    );

    let copy_ds =
        DatasetOps::duplicate(&db, "original-multi", "copy-multi").expect("duplicate dataset");

    assert_eq!(copy_ds.name, "copy-multi");
    assert_ne!(copy_ds.id, ds_id, "copy should have a different ID");

    assert_eq!(
        DatasetService::get_sample_count(&db, copy_ds.id).expect("copy sample count"),
        2,
        "copy should have same sample count"
    );
    assert_eq!(
        DatasetService::get_asset_count(&db, copy_ds.id).expect("copy asset count"),
        3,
        "copy should have same asset count"
    );
    assert_eq!(
        DatasetService::get_annotation_count(&db, copy_ds.id).expect("copy annotation count"),
        3,
        "copy should have same annotation count"
    );

    assert_eq!(
        DatasetService::get_sample_count(&db, ds_id).expect("orig sample count"),
        2,
        "original should still have 2 samples"
    );

    let copy_samples = DatasetService::get_samples(&db, copy_ds.id).expect("copy samples");
    assert_eq!(copy_samples.len(), 2);

    let copy_s1 = copy_samples
        .iter()
        .find(|s| s.name == "sample-1")
        .expect("copy should have sample-1");
    let copy_s1_assets = DatasetService::get_assets(&db, copy_s1.id).expect("copy s1 assets");
    assert_eq!(
        copy_s1_assets.len(),
        2,
        "copied sample-1 should have 2 assets"
    );

    let copy_s2 = copy_samples
        .iter()
        .find(|s| s.name == "sample-2")
        .expect("copy should have sample-2");
    let copy_s2_assets = DatasetService::get_assets(&db, copy_s2.id).expect("copy s2 assets");
    assert_eq!(
        copy_s2_assets.len(),
        1,
        "copied sample-2 should have 1 asset"
    );

    let copy_s1_anns =
        DatasetService::get_annotations_for_sample(&db, copy_s1.id).expect("copy s1 anns");
    assert_eq!(
        copy_s1_anns.len(),
        2,
        "copied sample-1 should have 2 annotations"
    );
    let copy_s1_asset_ids: Vec<i64> = copy_s1_assets.iter().map(|a| a.id).collect();
    for ann in &copy_s1_anns {
        let ann_asset_id = ann.asset_id.expect("annotation should reference an asset");
        assert!(
            copy_s1_asset_ids.contains(&ann_asset_id),
            "annotation asset_id {} must point to a copied asset, not the original",
            ann_asset_id
        );
    }
}

#[test]
fn test_split_preserves_sample_integrity() {
    let tmp = TempDir::new().expect("tempdir");
    let db = in_memory_db();
    let ds_id = register_dataset(&db, "split-source", &tmp);

    for i in 0..10_i32 {
        let s_id = DatasetService::add_sample(&db, ds_id, &format!("frame-{:03}", i), None)
            .expect("add sample");
        DatasetService::add_asset(
            &db,
            s_id,
            AssetType::Image,
            &format!("frame-{:03}-rgb.jpg", i),
            Path::new(&format!("/data/rgb/frame-{:03}.jpg", i)),
            Some(1920),
            Some(1080),
            None,
            None,
        )
        .expect("add rgb asset");
        DatasetService::add_asset(
            &db,
            s_id,
            AssetType::DepthMap,
            &format!("frame-{:03}-depth.png", i),
            Path::new(&format!("/data/depth/frame-{:03}.png", i)),
            Some(1920),
            Some(1080),
            None,
            None,
        )
        .expect("add depth asset");
    }

    assert_eq!(
        DatasetService::get_sample_count(&db, ds_id).expect("sample count"),
        10
    );
    assert_eq!(
        DatasetService::get_asset_count(&db, ds_id).expect("asset count"),
        20
    );

    let mut ratios = HashMap::new();
    ratios.insert("train".to_string(), 0.7);
    ratios.insert("val".to_string(), 0.3);

    let splits = DatasetOps::split(&db, "split-source", ratios, 42).expect("split");
    assert_eq!(splits.len(), 2, "should produce 2 split datasets");

    let total_samples: i64 = splits
        .iter()
        .map(|ds| DatasetService::get_sample_count(&db, ds.id).expect("split sample count"))
        .sum();
    assert_eq!(total_samples, 10, "all 10 samples must be distributed");

    for split_ds in &splits {
        let samples = DatasetService::get_samples(&db, split_ds.id).expect("get split samples");
        for sample in &samples {
            let assets = DatasetService::get_assets(&db, sample.id).expect("get sample assets");
            assert_eq!(
                assets.len(),
                2,
                "sample '{}' in '{}' should have 2 assets, not {}",
                sample.name,
                split_ds.name,
                assets.len()
            );
        }
        let split_sample_count =
            DatasetService::get_sample_count(&db, split_ds.id).expect("split sample count");
        let split_asset_count =
            DatasetService::get_asset_count(&db, split_ds.id).expect("split asset count");
        assert_eq!(
            split_asset_count,
            split_sample_count * 2,
            "partition '{}': asset count should equal 2x sample count",
            split_ds.name
        );
    }
}

#[test]
fn test_dual_level_annotation() {
    let tmp = TempDir::new().expect("tempdir");
    let db = in_memory_db();
    let ds_id = register_dataset(&db, "dual-annotation", &tmp);

    let sample_id = DatasetService::add_sample(&db, ds_id, "scene-001", None).expect("add sample");

    let asset_id = DatasetService::add_asset(
        &db,
        sample_id,
        AssetType::Image,
        "scene-001.jpg",
        Path::new("/data/scene-001.jpg"),
        Some(640),
        Some(480),
        None,
        None,
    )
    .expect("add image asset");

    let other_asset_id = DatasetService::add_asset(
        &db,
        sample_id,
        AssetType::DepthMap,
        "scene-001-depth.png",
        Path::new("/data/scene-001-depth.png"),
        Some(640),
        Some(480),
        None,
        None,
    )
    .expect("add depth asset");

    let sample_ann_id =
        DatasetService::add_annotation(&db, sample_id, None, None, None, None, None, None)
            .expect("add sample-level annotation");

    let asset_ann_id = DatasetService::add_annotation(
        &db,
        sample_id,
        Some(asset_id),
        None,
        None,
        None,
        None,
        None,
    )
    .expect("add asset-level annotation");

    let sample_anns =
        DatasetService::get_annotations_for_sample(&db, sample_id).expect("sample annotations");
    assert_eq!(
        sample_anns.len(),
        2,
        "get_annotations_for_sample must return both sample-level and asset-level annotations"
    );

    let ann_ids: Vec<i64> = sample_anns.iter().map(|a| a.id).collect();
    assert!(
        ann_ids.contains(&sample_ann_id),
        "sample-level annotation missing"
    );
    assert!(
        ann_ids.contains(&asset_ann_id),
        "asset-level annotation missing"
    );

    let sample_ann = sample_anns
        .iter()
        .find(|a| a.id == sample_ann_id)
        .expect("find sample-level annotation");
    assert_eq!(
        sample_ann.asset_id, None,
        "sample-level annotation must have asset_id=None"
    );
    assert_eq!(sample_ann.sample_id, sample_id);

    let asset_ann = sample_anns
        .iter()
        .find(|a| a.id == asset_ann_id)
        .expect("find asset-level annotation");
    assert_eq!(
        asset_ann.asset_id,
        Some(asset_id),
        "asset-level annotation must reference the image asset"
    );
    assert_eq!(asset_ann.sample_id, sample_id);

    let asset_anns =
        DatasetService::get_annotations_for_asset(&db, asset_id).expect("asset annotations");
    assert_eq!(
        asset_anns.len(),
        1,
        "get_annotations_for_asset must return only the asset-scoped annotation"
    );
    assert_eq!(asset_anns[0].id, asset_ann_id);
    assert_eq!(asset_anns[0].asset_id, Some(asset_id));

    let other_asset_anns =
        DatasetService::get_annotations_for_asset(&db, other_asset_id).expect("other asset anns");
    assert_eq!(other_asset_anns.len(), 0, "depth asset has no annotations");

    assert_eq!(
        DatasetService::get_annotation_count(&db, ds_id).expect("annotation count"),
        2,
        "dataset should have 2 total annotations"
    );
}
