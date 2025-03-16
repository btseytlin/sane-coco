import copy
import json
import numpy as np
from pathlib import Path
import pytest
from pycocotools.coco import COCO as OldCOCO
from pycocotools.cocoeval import COCOeval
from sane_coco import COCODataset, BBox
from sane_coco.metrics import MeanAveragePrecision


@pytest.fixture
def pred_data():
    return {
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [102, 98, 48, 102],
                "score": 0.9,
                "area": 48 * 102,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,
                "bbox": [198, 152, 82, 58],
                "score": 0.8,
                "area": 82 * 58,
                "iscrowd": 0,
            },
            {
                "id": 3,
                "image_id": 2,
                "category_id": 1,
                "bbox": [305, 195, 38, 92],
                "score": 0.95,
                "area": 38 * 92,
                "iscrowd": 0,
            },
        ]
    }


@pytest.fixture
def prepared_sample_data(sample_data):
    for ann in sample_data["annotations"]:
        ann["area"] = ann["bbox"][2] * ann["bbox"][3]
        ann["iscrowd"] = 0
    return sample_data


@pytest.fixture
def old_coco(prepared_sample_data):
    old_coco = OldCOCO()
    old_coco.dataset = prepared_sample_data
    old_coco.createIndex()
    return old_coco


@pytest.fixture
def old_eval(old_coco, pred_data):
    old_eval = COCOeval(old_coco, old_coco.loadRes(pred_data["annotations"]))
    old_eval.params.iouType = "bbox"
    old_eval.evaluate()
    old_eval.accumulate()
    old_eval.summarize()
    return old_eval


@pytest.fixture
def default_iou_thresholds(old_eval):
    return old_eval.params.iouThrs


@pytest.fixture
def default_recall_thresholds(old_eval):
    return old_eval.params.recThrs


@pytest.fixture
def default_max_dets(old_eval):
    return old_eval.params.maxDets[-1]


@pytest.fixture
def default_area_ranges():
    return {
        "all": (0, float("inf")),
        "small": (0, 32**2),
        "medium": (32**2, 96**2),
        "large": (96**2, float("inf")),
    }


@pytest.fixture
def dataset(prepared_sample_data):
    return COCODataset.from_dict(prepared_sample_data)


@pytest.fixture
def pred_annotations(dataset, pred_data):
    pred_annotations = []
    for img in dataset.images:
        img_pred = []
        for pred in pred_data["annotations"]:
            if pred["image_id"] == img.id:
                img_pred.append(
                    {
                        "category": dataset.get_category_by_id(
                            pred["category_id"]
                        ).name,
                        "bbox": pred["bbox"],
                        "score": pred["score"],
                    }
                )
        pred_annotations.append(img_pred)
    return pred_annotations


def test_basic_loading(sample_data):
    old_coco = OldCOCO()
    old_coco.dataset = sample_data
    old_coco.createIndex()

    dataset = COCODataset.from_dict(sample_data)

    assert len(dataset.categories) == 3
    person = dataset.categories["person"]
    assert person.id == 1
    assert person.name == "person"
    assert person.supercategory == "person"

    images = list(dataset.images)
    assert len(images) == 2
    first_image = images[0]
    assert first_image.width == 640
    assert first_image.height == 480
    assert first_image.file_name == "000000001.jpg"

    assert len(dataset.annotations) == 4
    assert dataset.annotations[0] is first_image.annotations[0]
    assert dataset.annotations[1] is first_image.annotations[1]

    all_annotations = [ann for img in images for ann in img.annotations]
    assert len(all_annotations) == len(old_coco.loadAnns(old_coco.getAnnIds()))

    first_ann = first_image.annotations[0]
    assert first_ann.bbox.x == 100
    assert first_ann.bbox.y == 100
    assert first_ann.bbox.width == 50
    assert first_ann.bbox.height == 100
    assert first_ann.image is first_image
    assert first_ann.category is person

    person_annotations = [ann for ann in all_annotations if ann.category == person]
    assert len(person_annotations) == len(
        old_coco.loadAnns(old_coco.getAnnIds(catIds=[1]))
    )


def test_from_pycocotools(old_coco):
    dataset = COCODataset.from_pycocotools(old_coco)

    assert len(dataset.categories) == len(old_coco.cats)
    assert len(dataset.images) == len(old_coco.imgs)
    assert len(dataset.annotations) == len(old_coco.anns)

    converted_dict = dataset.to_dict()

    assert len(converted_dict["categories"]) == len(old_coco.dataset["categories"])
    for cat in converted_dict["categories"]:
        old_cat = old_coco.cats[cat["id"]]
        assert cat["name"] == old_cat["name"]
        assert cat["supercategory"] == old_cat["supercategory"]

    assert len(converted_dict["images"]) == len(old_coco.dataset["images"])
    for img in converted_dict["images"]:
        old_img = old_coco.imgs[img["id"]]
        assert img["file_name"] == old_img["file_name"]
        assert img["height"] == old_img["height"]
        assert img["width"] == old_img["width"]

    assert len(converted_dict["annotations"]) == len(old_coco.dataset["annotations"])
    for ann in converted_dict["annotations"]:
        old_ann = old_coco.anns[ann["id"]]
        assert ann["image_id"] == old_ann["image_id"]
        assert ann["category_id"] == old_ann["category_id"]
        assert ann["bbox"] == old_ann["bbox"]


def test_category_queries(sample_data):
    old_coco = OldCOCO()
    old_coco.dataset = sample_data
    old_coco.createIndex()

    dataset = COCODataset.from_dict(sample_data)

    person_annotations = [
        ann
        for img in dataset.images
        for ann in img.annotations
        if ann.category.name == "person"
    ]
    assert len(person_annotations) == len(
        old_coco.loadAnns(
            old_coco.getAnnIds(catIds=old_coco.getCatIds(catNms=["person"]))
        )
    )

    cat_images = [
        img
        for img in dataset.images
        if any(ann.category.name == "cat" for ann in img.annotations)
    ]
    assert len(cat_images) == len(
        old_coco.loadImgs(old_coco.getImgIds(catIds=old_coco.getCatIds(catNms=["cat"])))
    )


def test_ap_metrics(
    dataset,
    pred_annotations,
    old_eval,
    default_iou_thresholds,
    default_max_dets,
    default_area_ranges,
):
    gt_annotations = dataset.get_annotation_dicts()
    metric = MeanAveragePrecision(
        iou_thresholds=default_iou_thresholds.tolist(),
        max_detections=default_max_dets,
        area_ranges=default_area_ranges,
    )
    metric.update(gt_annotations, pred_annotations)
    results = metric.compute()

    assert abs(results["map"] - old_eval.stats[0]) < 0.1
    assert abs(results["ap"][0.5] - old_eval.stats[1]) < 0.1
    assert abs(results["ap"][0.75] - old_eval.stats[2]) < 0.1


def test_ar_metrics(
    dataset,
    pred_annotations,
    old_eval,
    default_iou_thresholds,
    default_max_dets,
    default_area_ranges,
):
    gt_annotations = dataset.get_annotation_dicts()
    metric = MeanAveragePrecision(
        iou_thresholds=default_iou_thresholds.tolist(),
        max_detections=default_max_dets,
        area_ranges=default_area_ranges,
    )
    metric.update(gt_annotations, pred_annotations)
    results = metric.compute()

    assert abs(results["mar"] - old_eval.stats[8]) < 0.1
    assert 0 <= results["ar"][0.5] <= 1
    assert 0 <= results["ar"][0.75] <= 1

    for size in ["small", "medium", "large"]:
        for iou in default_iou_thresholds:
            assert 0 <= results["size"][size][iou] <= 1


def test_max_detections(dataset, pred_annotations, old_eval):
    gt_annotations = dataset.get_annotation_dicts()
    for max_dets in old_eval.params.maxDets:
        metric = MeanAveragePrecision(max_detections=max_dets)
        metric.update(gt_annotations, pred_annotations)
        results = metric.compute()
        assert "ap" in results
        assert "ar" in results


def test_per_category_evaluation(dataset, pred_annotations, old_coco, pred_data):
    old_eval_per_cat = COCOeval(old_coco, old_coco.loadRes(pred_data["annotations"]))
    old_eval_per_cat.params.iouType = "bbox"
    old_eval_per_cat.params.useCats = 1
    old_eval_per_cat.evaluate()
    old_eval_per_cat.accumulate()

    cat_ids = old_eval_per_cat.params.catIds

    gt_annotations = dataset.get_annotation_dicts()
    for cat_id in cat_ids:
        cat_name = dataset.get_category_by_id(cat_id).name

        cat_gt = []
        for img_gt in gt_annotations:
            cat_gt.append([ann for ann in img_gt if ann["category"] == cat_name])

        cat_pred = []
        for img_pred in pred_annotations:
            cat_pred.append([ann for ann in img_pred if ann["category"] == cat_name])

        metric = MeanAveragePrecision()
        metric.update(cat_gt, cat_pred)
        cat_results = metric.compute()

        assert "ap" in cat_results
        assert "ar" in cat_results


def test_area_based_evaluation(dataset, pred_annotations, default_area_ranges):
    gt_annotations = dataset.get_annotation_dicts()
    metric = MeanAveragePrecision(area_ranges=default_area_ranges)
    metric.update(gt_annotations, pred_annotations)
    results = metric.compute()

    assert "small" in results["size"]
    assert "medium" in results["size"]
    assert "large" in results["size"]

    for size in ["small", "medium", "large"]:
        for iou in metric.iou_thresholds:
            assert 0 <= results["size"][size][iou] <= 1
