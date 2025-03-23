import json
import pytest
import tempfile
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sane_coco import COCODataset
from sane_coco.metrics import MeanAveragePrecision


def default_pycocotools_params():
    iouThrs = np.linspace(
        0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
    )
    recThrs = np.linspace(
        0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
    )
    maxDets = [1, 10, 100]
    areaRng = [[0**2, 1e5**2], [0**2, 32**2], [32**2, 96**2], [96**2, 1e5**2]]
    areaRngLbl = ["all", "small", "medium", "large"]
    return iouThrs, recThrs, maxDets, areaRng, areaRngLbl


@pytest.fixture
def lvis_data_mini():
    lvis_path = Path("data/lvis/lvis_val_100.json")
    lvis_pred_path = Path("data/lvis/lvis_results_100.json")

    with open(lvis_path, "r") as f:
        lvis_data = json.load(f)

    lvis_data["images"] = lvis_data["images"][:1]
    lvis_data["annotations"] = [
        ann
        for ann in lvis_data["annotations"]
        if ann["image_id"] == lvis_data["images"][0]["id"]
    ][:2]

    with open(lvis_pred_path, "r") as f:
        lvis_pred_data = json.load(f)

    lvis_pred_data = [
        pred
        for pred in lvis_pred_data
        if pred["image_id"] == lvis_data["images"][0]["id"]
    ]

    dataset = COCODataset.from_dict(lvis_data)
    annotations_true = dataset.get_annotation_dicts()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as tmp:
        for ann in lvis_data["annotations"]:
            ann["iscrowd"] = 0
        json.dump(lvis_data, tmp)
        tmp.flush()

        old_coco = COCO(tmp.name)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as tmp:
        json.dump(lvis_pred_data, tmp)
        tmp.flush()

        old_pred_data = old_coco.loadRes(tmp.name)

    annotations_pred = []
    for image in dataset.images:
        annotations_image_pred = []
        for pred in lvis_pred_data:
            if pred["image_id"] == image.id:
                category = dataset.get_category_by_id(pred["category_id"])
                annotations_image_pred.append(
                    {
                        "category": category.name,
                        "bbox": pred["bbox"],
                        "score": pred["score"],
                    }
                )
        annotations_pred.append(annotations_image_pred)

    return annotations_true, annotations_pred, old_coco, old_pred_data


def test_lvis_map_comparison_mini(lvis_data_mini):
    annotations_true, annotations_pred, old_coco, old_pred_data = lvis_data_mini

    max_detections = 100
    iou_thresholds = [0.5]
    area_ranges = {
        "all": [0, float("inf")],
    }

    area_ranges_pycocotools = [area_range for area_range in area_ranges.values()]
    area_range_labels_pycocotools = list(area_ranges.keys())

    metric = MeanAveragePrecision(
        max_detections=max_detections,
        iou_thresholds=iou_thresholds,
        area_ranges=area_ranges,
    )

    metric.update(annotations_true, annotations_pred)
    results = metric.compute()

    old_eval = COCOeval(old_coco, old_pred_data, "bbox")
    old_eval.params.maxDets = [0, 10, max_detections]
    old_eval.params.iouThrs = np.array(iou_thresholds)
    old_eval.params.areaRng = area_ranges_pycocotools
    old_eval.params.areaRngLbl = area_range_labels_pycocotools
    old_eval.evaluate()
    old_eval.accumulate()
    old_eval.summarize()
    old_map = old_eval.stats[0]

    old_ap_05 = old_eval.stats[1]
    assert np.allclose(results["ap"][0.5], old_ap_05, atol=1e-6), (
        results["ap"][0.5],
        old_ap_05,
    )
    old_ar_05 = old_eval.stats[8]
    assert np.allclose(results["ar"][0.5], old_ar_05, atol=1e-6), (
        results["ar"][0.5],
        old_ar_05,
    )

    assert np.allclose(results["map"], old_map, atol=1e-6), (
        results["map"],
        old_map,
    )


# def test_lvis_map_comparison():
#     max_detections = 300
#     iou_thresholds = [0.5]  # ,, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
#     area_ranges = {
#         "all": [0, float("inf")],
#         # "small": [0, 32 * 32],
#         # "medium": [32 * 32, 96 * 96],
#         # "large": [96 * 96, float("inf")],
#     }

#     lvis_path = Path("data/lvis/lvis_val_100.json")
#     lvis_pred_path = Path("data/lvis/lvis_results_100.json")

#     with open(lvis_path, "r") as f:
#         lvis_data = json.load(f)

#     with open(lvis_pred_path, "r") as f:
#         lvis_pred_data = json.load(f)

#     dataset = COCODataset.from_dict(lvis_data)
#     annotations_true = dataset.get_annotation_dicts()

#     with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as tmp:
#         for ann in lvis_data["annotations"]:
#             ann["iscrowd"] = 0
#         json.dump(lvis_data, tmp)
#         tmp.flush()

#         old_coco = COCO(tmp.name)
#         old_pred_data = old_coco.loadRes(str(lvis_pred_path))

#     annotations_pred = []
#     for image in dataset.images:
#         annotations_image_pred = []
#         for pred in lvis_pred_data:
#             if pred["image_id"] == image.id:
#                 category = dataset.get_category_by_id(pred["category_id"])
#                 annotations_image_pred.append(
#                     {
#                         "category": category.name,
#                         "bbox": pred["bbox"],
#                         "score": pred["score"],
#                     }
#                 )
#         annotations_pred.append(annotations_image_pred)

#     metric = MeanAveragePrecision(
#         max_detections=max_detections,
#         iou_thresholds=iou_thresholds,
#         area_ranges=area_ranges,
#     )

#     metric.update(annotations_true, annotations_pred)
#     results = metric.compute()

#     old_eval = COCOeval(old_coco, old_pred_data, "bbox")
#     old_eval.params.maxDets = [max_detections]
#     old_eval.params.iouThrs = iou_thresholds
#     old_eval.params.areaRng = area_ranges

#     old_eval.evaluate()
#     old_eval.accumulate()
#     old_eval.summarize()
#     old_map = old_eval.stats[0]

#     assert np.allclose(results["map"], old_map, atol=1e-6), (
#         results["map"],
#         old_map,
#     )
