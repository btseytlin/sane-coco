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
def lvis_data_raw():
    lvis_path = Path("data/lvis/lvis_val_100.json")
    lvis_pred_path = Path("data/lvis/lvis_results_100.json")

    with open(lvis_path, "r") as f:
        lvis_data = json.load(f)

    with open(lvis_pred_path, "r") as f:
        lvis_pred_data = json.load(f)

    return lvis_data, lvis_pred_data


@pytest.fixture
def lvis_data_mini(lvis_data_raw):
    lvis_data, lvis_pred_data = lvis_data_raw

    # Pick one image
    lvis_data["images"] = lvis_data["images"][:1]

    # Two annotations
    lvis_data["annotations"] = [
        ann
        for ann in lvis_data["annotations"]
        if ann["image_id"] == lvis_data["images"][0]["id"]
    ][:2]

    true_categories = [ann["category_id"] for ann in lvis_data["annotations"]]

    # Predictions for the first image and the true categories
    lvis_pred_data = [
        pred
        for pred in lvis_pred_data
        if pred["image_id"] == lvis_data["images"][0]["id"]
        and pred["category_id"] in true_categories
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


@pytest.fixture
def lvis_data_small(lvis_data_raw):
    lvis_data, lvis_pred_data = lvis_data_raw

    # Pick 10 images
    lvis_data["images"] = lvis_data["images"][:2]
    image_ids = [image["id"] for image in lvis_data["images"]]

    # All annotations
    lvis_data["annotations"] = [
        ann for ann in lvis_data["annotations"] if ann["image_id"] in image_ids
    ]

    # Predictions for the included images and all categories
    lvis_pred_data = [pred for pred in lvis_pred_data if pred["image_id"] in image_ids]

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
    iou_thresholds = [0.5, 0.75, 0.95]
    area_ranges = {
        "all": [0, float("inf")],
        "small": [0, 32 * 32],
    }

    metric = MeanAveragePrecision(
        max_detections=max_detections,
        iou_thresholds=iou_thresholds,
        area_ranges=area_ranges,
    )

    metric.update(annotations_true, annotations_pred)
    results = metric.compute()

    area_ranges_pycocotools = [area_range for area_range in area_ranges.values()]
    area_range_labels_pycocotools = list(area_ranges.keys())

    old_eval = COCOeval(old_coco, old_pred_data, "bbox")
    old_eval.params.maxDets = [0, 10, max_detections]
    old_eval.params.iouThrs = np.array(iou_thresholds)
    old_eval.params.areaRng = area_ranges_pycocotools
    old_eval.params.areaRngLbl = area_range_labels_pycocotools
    old_eval.evaluate()
    old_eval.accumulate()
    old_eval.summarize()

    old_ap_05 = old_eval.stats[1]
    assert np.allclose(results["ap"][0.5], old_ap_05, atol=1e-6), (
        results["ap"][0.5],
        old_ap_05,
    )

    old_ap_75 = old_eval.stats[2]
    assert np.allclose(results["ap"][0.75], old_ap_75, atol=1e-6), (
        results["ap"][0.75],
        old_ap_75,
    )

    old_ar = old_eval.stats[8]
    assert np.allclose(results["mar"], old_ar, atol=1e-6), (
        results["mar"],
        old_ar,
    )

    old_map = old_eval.stats[0]
    assert np.allclose(results["map"], old_map, atol=1e-6), (
        results["map"],
        old_map,
    )


def test_lvis_data_loading(lvis_data_raw):
    lvis_data, lvis_pred_data = lvis_data_raw

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

    dataset = COCODataset.from_dict(lvis_data)
    annotations_true = dataset.get_annotation_dicts()
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

    assert len(annotations_true) == len(old_coco.imgs)
    assert len(annotations_pred) == len(old_coco.imgs)
    assert sum(len(x) for x in annotations_true) == len(old_coco.anns)
    assert sum(len(x) for x in annotations_pred) == len(old_pred_data.anns)

    pred_areas = np.mean(
        [ann["bbox"][2] * ann["bbox"][3] for preds in annotations_pred for ann in preds]
    )
    old_areas = np.mean([ann["area"] for ann in old_pred_data.anns.values()])
    assert np.allclose(pred_areas, old_areas, rtol=1e-6)

    pred_scores = np.mean([ann["score"] for preds in annotations_pred for ann in preds])
    old_scores = np.mean([ann["score"] for ann in old_pred_data.anns.values()])
    assert np.allclose(pred_scores, old_scores, rtol=1e-6)


def test_lvis_map_comparison_small(lvis_data_small):
    annotations_true, annotations_pred, old_coco, old_pred_data = lvis_data_small

    max_detections = 100
    iou_thresholds = [0.5, 0.75, 0.95]
    area_ranges = {
        "all": [0, float("inf")],
        "small": [0, 32 * 32],
    }

    metric = MeanAveragePrecision(
        max_detections=max_detections,
        iou_thresholds=iou_thresholds,
        area_ranges=area_ranges,
    )

    metric.update(annotations_true, annotations_pred)
    results = metric.compute()

    area_ranges_pycocotools = [area_range for area_range in area_ranges.values()]
    area_range_labels_pycocotools = list(area_ranges.keys())

    old_eval = COCOeval(old_coco, old_pred_data, "bbox")
    old_eval.params.maxDets = [0, 10, max_detections]
    old_eval.params.iouThrs = np.array(iou_thresholds)
    old_eval.params.areaRng = area_ranges_pycocotools
    old_eval.params.areaRngLbl = area_range_labels_pycocotools
    old_eval.evaluate()
    old_eval.accumulate()
    old_eval.summarize()

    print(results)
    # old_ap_05 = old_eval.stats[1]
    # assert np.allclose(results["ap"][0.5], old_ap_05, atol=1e-6), (
    #     results["ap"][0.5],
    #     old_ap_05,
    # )

    # old_ap_75 = old_eval.stats[2]
    # assert np.allclose(results["ap"][0.75], old_ap_75, atol=1e-6), (
    #     results["ap"][0.75],
    #     old_ap_75,
    # )
    # Add print statements after data loading
    print(f"Number of true annotations: {len(annotations_true)}")
    print(f"Number of predicted annotations: {sum(len(x) for x in annotations_pred)}")

    # After creating old_pred_data
    print(f"Pycocotools predictions: {len(old_pred_data.anns)}")
    print(f"Your predictions: {sum(len(x) for x in annotations_pred)}")

    # After metric.update()
    print(
        "Mean area in predictions:",
        np.mean(
            [
                ann["bbox"][2] * ann["bbox"][3]
                for preds in annotations_pred
                for ann in preds
            ]
        ),
    )
    print(
        "Mean area in pycocotools predictions:",
        np.mean([ann["area"] for ann in old_pred_data.anns.values()]),
    )

    print(
        "Score distribution:",
        np.mean([ann["score"] for preds in annotations_pred for ann in preds]),
    )
    print(
        "Score distribution in pycocotools:",
        np.mean([ann["score"] for ann in old_pred_data.anns.values()]),
    )

    old_map = old_eval.stats[0]
    assert np.allclose(results["map"], old_map, atol=1e-6), (
        results["map"],
        old_map,
    )

    old_ar = old_eval.stats[8]
    assert np.allclose(results["mar"], old_ar, atol=1e-6), (
        results["mar"],
        old_ar,
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
