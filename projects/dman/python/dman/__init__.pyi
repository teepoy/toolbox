"""Type stubs for the dman Python SDK.

dman organizes data in a four-level hierarchy:
Dataset → Sample → Asset → Annotation.

Module-level functions create, load, or update datasets in the local
catalog.  Builder/Updater provide a transactional API for mutations,
and DmanDataset is an eagerly-loaded read-only snapshot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

if TYPE_CHECKING:
    import pyarrow

# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------

def create_dataset(
    name: str,
    schema_path: str | None = None,
) -> DmanDatasetBuilder:
    """Create a new dataset in the dman catalog.

    Args:
        name: Unique dataset name.
        schema_path: Optional path to a JSON schema file for validation.

    Returns:
        A builder for adding samples, assets, annotations, and categories.
    """
    ...

def update_dataset(name: str) -> DmanDatasetUpdater:
    """Open an existing dataset for mutation.

    Args:
        name: Dataset name registered in dman catalog.

    Returns:
        An updater for adding/removing samples, assets, and annotations.

    Raises:
        ValueError: If dataset not found.
        RuntimeError: On catalog/DB errors.
    """
    ...

def load_dataset(
    name: str,
    split: str | None = None,
) -> DmanDataset:
    """Load a dataset from the dman catalog by name.

    Args:
        name: Dataset name registered in dman catalog.
        split: Optional split name (reserved for future use; currently ignored).

    Returns:
        An eagerly-loaded read-only snapshot of the dataset.

    Raises:
        ValueError: If dataset not found.
        RuntimeError: On catalog/DB errors.
    """
    ...

# ---------------------------------------------------------------------------
# DmanDatasetBuilder
# ---------------------------------------------------------------------------

class DmanDatasetBuilder:
    """Transactional builder for creating a new dataset.

    Created via ``create_dataset()``.  Call ``build()`` to commit.
    """

    def add_sample(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Add a sample to the dataset.

        Args:
            name: Unique sample name within the dataset.
            metadata: Optional metadata dict (stored as JSON).

        Returns:
            Zero-based index of the added sample.
        """
        ...

    def add_asset(
        self,
        sample_name: str,
        asset_type: str,
        file_path: str,
        width: int | None = None,
        height: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Attach an asset to an existing sample.

        Args:
            sample_name: Name of the target sample.
            asset_type: Asset kind (``"image"``, ``"depth_map"``, etc.).
            file_path: Path to the asset file on disk.
            width: Optional pixel width.
            height: Optional pixel height.
            metadata: Optional metadata dict (stored as JSON).
        """
        ...

    def add_image(
        self,
        path: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Convenience: add a sample with a single image asset.

        Derives the sample name from the file name and creates both a
        sample and an ``"image"`` asset in one call.

        Args:
            path: Path to the image file.
            metadata: Optional metadata dict.

        Returns:
            Zero-based index of the added sample.
        """
        ...

    def add_annotation(
        self,
        sample_name: str,
        category: str,
        bbox: list[float] | None = None,
        segmentation: list[list[float]] | None = None,
        keypoints: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
        asset_name: str | None = None,
    ) -> None:
        """Add an annotation to a sample (or a specific asset).

        Args:
            sample_name: Name of the target sample.
            category: Category name for this annotation.
            bbox: Optional bounding box ``[x, y, width, height]``.
            segmentation: Optional polygon segmentation.
            keypoints: Optional keypoint coordinates.
            metadata: Optional metadata dict.
            asset_name: Optional asset file name to attach annotation to a
                specific asset rather than the sample.
        """
        ...

    def set_category(
        self,
        name: str,
        supercategory: str | None = None,
    ) -> None:
        """Register a category for the dataset.

        Args:
            name: Category name.
            supercategory: Optional parent category name.
        """
        ...

    def build(self) -> DmanDataset:
        """Commit the dataset to the catalog.

        Executes all buffered operations in a single transaction and
        returns the loaded dataset snapshot.

        Returns:
            The newly created dataset.

        Raises:
            RuntimeError: On transaction or DB errors.
        """
        ...

# ---------------------------------------------------------------------------
# DmanDatasetUpdater
# ---------------------------------------------------------------------------

class DmanDatasetUpdater:
    """Transactional updater for mutating an existing dataset.

    Created via ``update_dataset()``.  Call ``apply()`` to commit.
    """

    def add_sample(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a new sample to the dataset.

        Args:
            name: Unique sample name.
            metadata: Optional metadata dict.
        """
        ...

    def add_asset(
        self,
        sample_name: str,
        asset_type: str,
        file_path: str,
        width: int | None = None,
        height: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Attach an asset to an existing sample.

        Args:
            sample_name: Name of the target sample.
            asset_type: Asset kind (``"image"``, ``"depth_map"``, etc.).
            file_path: Path to the asset file on disk.
            width: Optional pixel width.
            height: Optional pixel height.
            metadata: Optional metadata dict.
        """
        ...

    def add_annotation(
        self,
        sample_id: int,
        category: str,
        bbox: list[float] | None = None,
        asset_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add an annotation to a sample by ID.

        Args:
            sample_id: Database ID of the target sample.
            category: Category name.
            bbox: Optional bounding box ``[x, y, width, height]``.
            asset_id: Optional asset ID to scope annotation to.
            metadata: Optional metadata dict.
        """
        ...

    def remove_sample(self, sample_id: int) -> None:
        """Remove a sample and its associated assets/annotations.

        Args:
            sample_id: Database ID of the sample to remove.
        """
        ...

    def apply(self) -> None:
        """Commit all buffered mutations in a single transaction.

        Raises:
            RuntimeError: On transaction or DB errors.
        """
        ...

# ---------------------------------------------------------------------------
# DmanDataset
# ---------------------------------------------------------------------------

class DmanDataset:
    """Eagerly-loaded, read-only snapshot of a dataset.

    Supports ``len()``, indexing with ``[]``, and iteration.
    """

    @property
    def name(self) -> str:
        """Dataset name."""
        ...

    @property
    def dataset_id(self) -> int:
        """Internal dataset ID."""
        ...

    def __len__(self) -> int:
        """Number of samples in the dataset."""
        ...

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a sample dict at the given index.

        The dict contains keys: ``id``, ``name``, ``metadata``,
        ``assets`` (list of asset dicts), ``annotations`` (list of
        annotation dicts).

        Raises:
            IndexError: If *idx* is out of range.
        """
        ...

    # ── Query methods ─────────────────────────────────────────────

    def samples(self) -> list[dict[str, Any]]:
        """All samples as dicts, each including its assets."""
        ...

    def get_sample(self, name: str) -> dict[str, Any] | None:
        """Lookup a sample by name.

        Returns:
            Sample dict with assets, or ``None`` if not found.
        """
        ...

    def annotations(
        self,
        sample_name: str,
        asset_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Annotations for a sample, optionally filtered by asset.

        Args:
            sample_name: Name of the sample.
            asset_name: Optional asset file name to filter by.

        Raises:
            ValueError: If sample (or asset) not found.
        """
        ...

    def images(self) -> list[str]:
        """File paths of all image-type assets."""
        ...

    # ── Count helpers ─────────────────────────────────────────────

    def sample_count(self) -> int:
        """Number of samples."""
        ...

    def asset_count(self) -> int:
        """Number of assets."""
        ...

    def annotation_count(self) -> int:
        """Number of annotations."""
        ...

    def category_count(self) -> int:
        """Number of categories."""
        ...

    # ── Interop ───────────────────────────────────────────────────

    def to_torch_dataset(self) -> Any:
        """Return a ``torch.utils.data.Dataset``-compatible wrapper.

        Raises:
            ImportError: If PyTorch is not installed.
        """
        ...

    def to_hf_dataset(self) -> Any:
        """Return a HuggingFace ``datasets.Dataset``.

        Raises:
            ImportError: If ``datasets`` is not installed.
        """
        ...

    # ── Arrow zero-copy ───────────────────────────────────────────

    def samples_arrow(self) -> pyarrow.RecordBatch:
        """Samples as an Arrow RecordBatch (zero-copy via FFI).

        Schema: ``id(int64), dataset_id(int64), name(utf8),
        metadata(utf8, nullable), created_at(utf8, nullable)``.
        """
        ...

    def assets_arrow(self) -> pyarrow.RecordBatch:
        """Assets as an Arrow RecordBatch (zero-copy via FFI).

        Schema: ``id(int64), sample_id(int64), asset_type(utf8),
        file_name(utf8), file_path(utf8), width(int64, nullable),
        height(int64, nullable), hash(utf8, nullable),
        metadata(utf8, nullable)``.
        """
        ...

    def annotations_arrow(self) -> pyarrow.RecordBatch:
        """Annotations as an Arrow RecordBatch (zero-copy via FFI).

        Schema: ``id(int64), sample_id(int64), asset_id(int64, nullable),
        category_id(int64, nullable), bbox(utf8, nullable),
        segmentation(utf8, nullable), keypoints(utf8, nullable),
        metadata(utf8, nullable)``.
        """
        ...

    def categories_arrow(self) -> pyarrow.RecordBatch:
        """Categories as an Arrow RecordBatch (zero-copy via FFI).

        Schema: ``id(int64), dataset_id(int64), name(utf8),
        supercategory(utf8, nullable)``.
        """
        ...

    def to_arrow(self) -> dict[str, pyarrow.RecordBatch]:
        """All tables as Arrow RecordBatches.

        Returns:
            Dict with keys ``"samples"``, ``"assets"``,
            ``"annotations"``, ``"categories"``.
        """
        ...

def main() -> int:
    """CLI entry point — resolves and runs ``dman-cli``."""
    ...
