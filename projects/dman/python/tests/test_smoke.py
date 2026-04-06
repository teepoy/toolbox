"""Smoke tests for the dman Python SDK.

These tests exercise the full create → load → update → query cycle against
a temporary, isolated catalog (DMAN_HOME set per test via tmp_path).
"""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point DMAN_HOME at a fresh temp directory and initialise the catalog."""
    import subprocess

    catalog_dir = tmp_path / "catalog"
    monkeypatch.setenv("DMAN_HOME", str(catalog_dir))

    result = subprocess.run(
        ["dman", "init"],
        capture_output=True,
        text=True,
        timeout=10,
        env={**os.environ, "DMAN_HOME": str(catalog_dir)},
    )
    assert result.returncode == 0, f"dman init failed: {result.stderr}"


@pytest.fixture()
def tiny_yolo(tmp_path: Path) -> Path:
    """Create a minimal 2-image YOLO dataset and return its root path."""
    root = tmp_path / "yolo"
    img_dir = root / "images" / "train"
    lbl_dir = root / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    # 1×1 PNG (smallest valid PNG) — we only need the file to exist
    from PIL import Image

    for i in range(2):
        Image.new("RGB", (4, 4), color=(100 + i * 50, 80, 80)).save(img_dir / f"img{i}.png")
        (lbl_dir / f"img{i}.txt").write_text(f"{i % 2} 0.5 0.5 0.4 0.4\n")

    (root / "data.yaml").write_text(
        textwrap.dedent("""\
            path: .
            train: images/train
            nc: 2
            names:
              0: cat
              1: dog
        """)
    )
    return root


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

def _import_yolo(yolo_path: Path, name: str = "smoke") -> None:
    """Import a YOLO dataset via the CLI."""
    import subprocess

    env = {**os.environ}
    result = subprocess.run(
        ["dman", "import", str(yolo_path), "--format", "yolo", "--name", name],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert result.returncode == 0, f"dman import failed: {result.stderr}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCreateAndLoad:
    """Builder → build → load roundtrip."""

    def test_create_dataset_returns_builder(self):
        import dman

        builder = dman.create_dataset("test_builder")
        assert isinstance(builder, dman.DmanDatasetBuilder)

    def test_build_empty_dataset(self):
        import dman

        builder = dman.create_dataset("empty_ds")
        ds = builder.build()
        assert isinstance(ds, dman.DmanDataset)
        assert ds.name == "empty_ds"
        assert len(ds) == 0
        assert ds.sample_count() == 0
        assert ds.asset_count() == 0
        assert ds.annotation_count() == 0
        assert ds.category_count() == 0

    def test_build_with_samples_and_assets(self, tmp_path: Path):
        import dman
        from PIL import Image

        img = tmp_path / "photo.png"
        Image.new("RGB", (4, 4), color=(255, 0, 0)).save(img)

        builder = dman.create_dataset("with_assets")
        builder.add_sample("s1", metadata={"tag": "red"})
        builder.add_asset("s1", "image", str(img), width=4, height=4)
        builder.add_annotation("s1", "cat", bbox=[1.0, 2.0, 3.0, 4.0])
        builder.set_category("cat", supercategory="animal")
        ds = builder.build()

        assert ds.name == "with_assets"
        assert len(ds) == 1
        assert ds.sample_count() == 1
        assert ds.asset_count() == 1
        assert ds.annotation_count() == 1
        assert ds.category_count() == 1

    def test_add_image_convenience(self, tmp_path: Path):
        import dman
        from PIL import Image

        img = tmp_path / "quick.png"
        Image.new("RGB", (4, 4), color=(0, 255, 0)).save(img)

        builder = dman.create_dataset("convenience")
        idx = builder.add_image(str(img))
        assert isinstance(idx, int)
        assert idx == 0

        ds = builder.build()
        assert ds.sample_count() == 1
        assert ds.asset_count() == 1

    def test_load_dataset_matches_build(self, tmp_path: Path):
        import dman
        from PIL import Image

        img = tmp_path / "img.png"
        Image.new("RGB", (4, 4), color=(0, 0, 255)).save(img)

        builder = dman.create_dataset("reload_test")
        builder.add_sample("s1")
        builder.add_asset("s1", "image", str(img))
        builder.build()

        ds = dman.load_dataset("reload_test")
        assert ds.name == "reload_test"
        assert len(ds) == 1

    def test_load_nonexistent_raises(self):
        import dman

        with pytest.raises(Exception):
            dman.load_dataset("does_not_exist")


class TestDatasetQuery:
    """Read methods on a loaded dataset."""

    @pytest.fixture(autouse=True)
    def _build_dataset(self, tmp_path: Path):
        import dman
        from PIL import Image

        img0 = tmp_path / "a.png"
        img1 = tmp_path / "b.png"
        Image.new("RGB", (4, 4), color=(255, 0, 0)).save(img0)
        Image.new("RGB", (4, 4), color=(0, 255, 0)).save(img1)

        builder = dman.create_dataset("query_ds")
        builder.add_sample("alpha")
        builder.add_asset("alpha", "image", str(img0), width=4, height=4)
        builder.add_annotation("alpha", "cat", bbox=[0.0, 0.0, 2.0, 2.0])

        builder.add_sample("beta")
        builder.add_asset("beta", "image", str(img1), width=4, height=4)

        builder.set_category("cat")
        self.ds = builder.build()

    def test_samples_returns_list_of_dicts(self):
        samples = self.ds.samples()
        assert isinstance(samples, list)
        assert len(samples) == 2
        assert all(isinstance(s, dict) for s in samples)
        assert all("name" in s for s in samples)
        assert all("assets" in s for s in samples)

    def test_getitem_returns_dict(self):
        item = self.ds[0]
        assert isinstance(item, dict)
        assert "id" in item
        assert "name" in item
        assert "assets" in item
        assert "annotations" in item

    def test_getitem_out_of_range_raises(self):
        with pytest.raises(IndexError):
            self.ds[999]

    def test_get_sample_by_name(self):
        s = self.ds.get_sample("alpha")
        assert s is not None
        assert s["name"] == "alpha"

    def test_get_sample_missing_returns_none(self):
        result = self.ds.get_sample("nonexistent")
        assert result is None

    def test_annotations_for_sample(self):
        anns = self.ds.annotations("alpha")
        assert isinstance(anns, list)
        assert len(anns) == 1
        assert "bbox" in anns[0]

    def test_annotations_empty_sample(self):
        anns = self.ds.annotations("beta")
        assert anns == []

    def test_annotations_missing_sample_raises(self):
        with pytest.raises(Exception):
            self.ds.annotations("no_such_sample")

    def test_images_returns_paths(self):
        paths = self.ds.images()
        assert isinstance(paths, list)
        assert len(paths) == 2
        assert all(isinstance(p, str) for p in paths)


class TestUpdate:
    """Updater workflow."""

    def test_update_adds_sample(self, tmp_path: Path):
        import dman
        from PIL import Image

        img = tmp_path / "orig.png"
        Image.new("RGB", (4, 4), color=(128, 128, 128)).save(img)

        builder = dman.create_dataset("upd_ds")
        builder.add_sample("first")
        builder.add_asset("first", "image", str(img))
        builder.build()

        # Update: add a second sample
        img2 = tmp_path / "new.png"
        Image.new("RGB", (4, 4), color=(64, 64, 64)).save(img2)

        updater = dman.update_dataset("upd_ds")
        updater.add_sample("second")
        updater.add_asset("second", "image", str(img2))
        updater.apply()

        ds = dman.load_dataset("upd_ds")
        assert ds.sample_count() == 2

    def test_update_remove_sample(self, tmp_path: Path):
        import dman
        from PIL import Image

        img = tmp_path / "r.png"
        Image.new("RGB", (4, 4), color=(200, 200, 200)).save(img)

        builder = dman.create_dataset("rm_ds")
        builder.add_sample("keep")
        builder.add_asset("keep", "image", str(img))
        builder.add_sample("remove_me")
        ds = builder.build()

        sample = ds.get_sample("remove_me")
        assert sample is not None

        updater = dman.update_dataset("rm_ds")
        updater.remove_sample(sample["id"])
        updater.apply()

        ds2 = dman.load_dataset("rm_ds")
        assert ds2.sample_count() == 1
        assert ds2.get_sample("remove_me") is None

    def test_update_add_annotation(self, tmp_path: Path):
        import dman
        from PIL import Image

        img = tmp_path / "ann.png"
        Image.new("RGB", (4, 4), color=(0, 0, 0)).save(img)

        builder = dman.create_dataset("ann_ds")
        builder.add_sample("s1")
        builder.add_asset("s1", "image", str(img))
        ds = builder.build()

        sample = ds.get_sample("s1")
        updater = dman.update_dataset("ann_ds")
        updater.add_annotation(sample["id"], "dog", bbox=[1.0, 1.0, 2.0, 2.0])
        updater.apply()

        ds2 = dman.load_dataset("ann_ds")
        assert ds2.annotation_count() == 1


class TestArrow:
    """Arrow zero-copy methods."""

    @pytest.fixture(autouse=True)
    def _build_dataset(self, tmp_path: Path):
        import dman
        from PIL import Image

        img = tmp_path / "arrow.png"
        Image.new("RGB", (4, 4), color=(10, 20, 30)).save(img)

        builder = dman.create_dataset("arrow_ds")
        builder.add_sample("s1")
        builder.add_asset("s1", "image", str(img))
        builder.add_annotation("s1", "cat")
        builder.set_category("cat")
        self.ds = builder.build()

    def test_samples_arrow(self):
        pyarrow = pytest.importorskip("pyarrow")
        batch = self.ds.samples_arrow()
        assert batch.num_rows == 1
        assert "name" in batch.schema.names

    def test_assets_arrow(self):
        pyarrow = pytest.importorskip("pyarrow")
        batch = self.ds.assets_arrow()
        assert batch.num_rows == 1
        assert "file_path" in batch.schema.names

    def test_annotations_arrow(self):
        pyarrow = pytest.importorskip("pyarrow")
        batch = self.ds.annotations_arrow()
        assert batch.num_rows == 1

    def test_categories_arrow(self):
        pyarrow = pytest.importorskip("pyarrow")
        batch = self.ds.categories_arrow()
        assert batch.num_rows == 1
        assert "name" in batch.schema.names

    def test_to_arrow_returns_dict(self):
        pyarrow = pytest.importorskip("pyarrow")
        result = self.ds.to_arrow()
        assert isinstance(result, dict)
        assert set(result.keys()) == {"samples", "assets", "annotations", "categories"}


class TestCLIImportRoundtrip:
    """Import via CLI, load via SDK."""

    def test_yolo_import_and_load(self, tiny_yolo: Path):
        import dman

        _import_yolo(tiny_yolo, name="cli_smoke")

        ds = dman.load_dataset("cli_smoke")
        assert ds.name == "cli_smoke"
        assert ds.sample_count() == 2
        assert ds.asset_count() == 2
        assert ds.annotation_count() == 2
        assert ds.category_count() == 2

    def test_yolo_import_samples_have_assets(self, tiny_yolo: Path):
        import dman

        _import_yolo(tiny_yolo, name="cli_assets")

        ds = dman.load_dataset("cli_assets")
        for s in ds.samples():
            assert len(s["assets"]) >= 1
            assert s["assets"][0]["asset_type"] == "image"

    def test_yolo_import_annotations_have_bbox(self, tiny_yolo: Path):
        import dman

        _import_yolo(tiny_yolo, name="cli_anns")

        ds = dman.load_dataset("cli_anns")
        for s in ds.samples():
            anns = ds.annotations(s["name"])
            assert len(anns) >= 1
            assert anns[0].get("bbox") is not None


class TestStubs:
    """Verify type stubs exist and are syntactically valid."""

    def test_pyi_file_exists(self):
        stub = Path(__file__).resolve().parent.parent / "dman" / "__init__.pyi"
        assert stub.exists(), f"Expected stub at {stub}"

    def test_py_typed_marker_exists(self):
        marker = Path(__file__).resolve().parent.parent / "dman" / "py.typed"
        assert marker.exists(), f"Expected py.typed at {marker}"

    def test_stub_is_valid_python(self):
        import ast

        stub = Path(__file__).resolve().parent.parent / "dman" / "__init__.pyi"
        source = stub.read_text()
        ast.parse(source)  # raises SyntaxError if invalid
