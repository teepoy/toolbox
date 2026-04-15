from pathlib import Path

from klarf_adapter import (
    KlarfDefectList,
    KlarfDefectRecordSpec,
    KlarfDocument,
    KlarfStatement,
    KlarfTable,
)
from klarf_adapter.parser import dumps, load, loads


SAMPLE_KLARF = """FileVersion 1 8;
LotID LOT-42;
WaferID WAFER-7;
DeviceID DEVICE-X;
OrientationMarkLocation DOWN;
DiePitch 3.5717153320e+003 3.3786257324e+003;
DieOrigin 0.000000 0.000000;
SampleCenterLocation 1.4683145313e+005 1.3981967188e+005;
ClassLookup 2
0 \"Alignment Failure\"
17 \"Top Side Chipping\"
;
InspectionTest 1;
SampleTestPlan 3
29 0
30 0
31 0
;
DefectRecordSpec 17 DEFECTID XREL YREL XINDEX YINDEX XSIZE YSIZE DEFECTAREA DSIZE CLASSNUMBER TEST CLUSTERNUMBER ROUGHBINNUMBER FINEBINNUMBER REVIEWSAMPLE IMAGECOUNT IMAGELIST;
DefectList 327 3.4916440430e+003 3.2954443359e+003 57 12 16.998291 16.990662 288.812211 1.6990661621e+001 17 1 0 0 0 0 1 0;
TiffFileName TR7U5303.05_Ceres_256993787_1.jpg;
"""

SAMPLE_FILE = Path(__file__).resolve().parent.parent / "sample.000"


def test_parse_common_klarf_sections() -> None:
    document = loads(SAMPLE_KLARF)

    assert isinstance(document.records[0], KlarfStatement)
    assert document.records[0].name == "FileVersion"
    assert document.records[0].values == ["1", "8"]

    class_lookup = next(
        record
        for record in document.records
        if isinstance(record, KlarfTable) and record.name == "ClassLookup"
    )
    assert class_lookup.rows == [
        ["0", "Alignment Failure"],
        ["17", "Top Side Chipping"],
    ]

    defect_spec = next(
        record
        for record in document.records
        if isinstance(record, KlarfDefectRecordSpec)
    )
    defect_list = next(
        record for record in document.records if isinstance(record, KlarfDefectList)
    )
    assert defect_list.as_mapping(defect_spec.fields)["CLASSNUMBER"] == "17"
    assert defect_list.as_mapping(defect_spec.fields)["IMAGELIST"] == "0"


def test_dump_round_trip_preserves_record_model() -> None:
    original = loads(SAMPLE_KLARF)
    rendered = dumps(original)
    reparsed = loads(rendered)

    assert reparsed.to_dict() == original.to_dict()


def test_document_from_dict_round_trip() -> None:
    document = KlarfDocument(
        records=[
            KlarfStatement(name="LotID", values=["LOT-42"]),
            KlarfTable(
                name="SampleTestPlan", row_width=2, rows=[["29", "0"], ["30", "0"]]
            ),
            KlarfDefectRecordSpec(fields=["DEFECTID", "CLASSNUMBER"]),
            KlarfDefectList(rows=[["327", "17"]]),
        ]
    )

    restored = KlarfDocument.from_dict(document.to_dict())
    assert restored.to_dict() == document.to_dict()


def test_parse_real_sample_file() -> None:
    document = load(SAMPLE_FILE)

    class_lookup = next(
        record
        for record in document.records
        if isinstance(record, KlarfTable) and record.name == "ClassLookup"
    )
    sample_test_plan = next(
        record
        for record in document.records
        if isinstance(record, KlarfTable) and record.name == "SampleTestPlan"
    )
    defect_spec = next(
        record
        for record in document.records
        if isinstance(record, KlarfDefectRecordSpec)
    )
    defect_list = next(
        record for record in document.records if isinstance(record, KlarfDefectList)
    )

    assert len(document.records) == 26
    assert len(class_lookup.rows) == 276
    assert class_lookup.rows[0] == ["0", "Undefined"]
    assert class_lookup.rows[-1] == ["1020", "Group20"]
    assert len(sample_test_plan.rows) == 4988
    assert sample_test_plan.rows[0] == ["-37", "-8"]
    assert sample_test_plan.rows[-1] == ["38", "9"]
    assert defect_spec.fields == [
        "DEFECTID",
        "XREL",
        "YREL",
        "XINDEX",
        "YINDEX",
        "XSIZE",
        "YSIZE",
        "DEFECTAREA",
        "DSIZE",
        "CLASSNUMBER",
        "TEST",
        "ROUGHBINNUMBER",
        "IMAGECOUNT",
        "IMAGELIST",
    ]
    assert len(defect_list.rows) == 16
    assert defect_list.rows[0] == [
        "1",
        "5.5264000000e+02",
        "1.0430800000e+03",
        "-35",
        "-9",
        "2.680000",
        "3.640000",
        "9.755200",
        "4.5206407965e+00",
        "0",
        "1",
        "0",
        "0",
        "0",
    ]
    assert defect_list.as_mappings(defect_spec.fields)[-1]["DEFECTID"] == "16"
    assert defect_list.as_mappings(defect_spec.fields)[-1]["IMAGELIST"] == "0"
