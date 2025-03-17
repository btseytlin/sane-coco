import json
import os
import tempfile
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sane_coco import COCODataset
from sane_coco.metrics import MeanAveragePrecision


def test_lvis_map_comparison():
    max_detections = 100
    iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    area_ranges = {
        "all": [0, 10000000000.0],
        "small": [0, 32 * 32],
        "medium": [32 * 32, 96 * 96],
        "large": [96 * 96, 10000000000.0],
    }

    lvis_path = Path("data/lvis/lvis_val_100.json")
    lvis_pred_path = Path("data/lvis/lvis_results_100.json")

    with open(lvis_path, "r") as f:
        lvis_data = json.load(f)

    with open(lvis_pred_path, "r") as f:
        lvis_pred_data = json.load(f)

    dataset = COCODataset.from_dict(lvis_data)
    annotations_true = dataset.get_annotation_dicts()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as tmp:
        for ann in lvis_data["annotations"]:
            ann["iscrowd"] = 0
        json.dump(lvis_data, tmp)
        tmp.flush()

        old_coco = COCO(tmp.name)
        old_pred_data = old_coco.loadRes(str(lvis_pred_path))

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

    metric = MeanAveragePrecision(
        max_detections=max_detections,
        iou_thresholds=iou_thresholds,
        area_ranges=area_ranges,
    )

    metric.update(annotations_true, annotations_pred)
    results = metric.compute()

    print(results)

    old_eval = COCOeval(old_coco, old_pred_data, "bbox")
    old_eval.evaluate()
    old_eval.accumulate()
    old_eval.summarize()
    old_map = old_eval.stats[0]

    print(old_eval.stats)

    assert np.allclose(results["map"], old_map, atol=1e-6)
