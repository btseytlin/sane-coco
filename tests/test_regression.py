import copy
import json
import numpy as np
from pathlib import Path
import pytest
from pycocotools.coco import COCO as OldCOCO
from pycocotools.cocoeval import COCOeval
from sane_coco import COCODataset, BBox


@pytest.fixture
def sample_data():
    return {
        "images": [
            {"id": 1, "file_name": "000000001.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "000000002.jpg", "width": 640, "height": 480},
        ],
        "categories": [
            {"id": 1, "name": "person", "supercategory": "person"},
            {"id": 2, "name": "dog", "supercategory": "animal"},
            {"id": 3, "name": "cat", "supercategory": "animal"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 50, 100]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [200, 150, 80, 60]},
            {"id": 3, "image_id": 2, "category_id": 1, "bbox": [300, 200, 40, 90]},
            {"id": 4, "image_id": 2, "category_id": 3, "bbox": [350, 250, 70, 50]},
        ],
    }


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


def test_bbox_operations():
    bbox = BBox(100, 100, 50, 100)
    assert bbox.area == 5000
    assert bbox.corners == (100, 100, 150, 200)


# def test_mask_operations():
#     polygon = [[100, 100, 200, 100, 200, 200, 100, 200]]
#     height, width = 300, 400

#     binary_mask = mask.polygons_to_mask(polygon, height, width)
#     rle = mask.encode_rle(binary_mask)

#     old_rles = OldCOCO.frPyObjects(polygon, height, width)
#     old_mask = OldCOCO.decode(old_rles)
#     old_rle = OldCOCO.encode(np.asfortranarray(old_mask))

#     assert np.array_equal(old_mask, binary_mask)
#     assert old_rle == rle


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
                        "area": pred["area"],
                    }
                )
        pred_annotations.append(img_pred)
    return pred_annotations


@pytest.fixture
def gt_annotations(dataset):
    return dataset.get_annotation_dicts()


def test_ap_metrics(
    dataset,
    pred_annotations,
    old_eval,
    default_iou_thresholds,
    default_max_dets,
    default_area_ranges,
):
    from sane_coco.metrics import MeanAveragePrecision

    gt_annotations = dataset.get_annotation_dicts()
    # Run our evaluation with the same parameters as COCOeval
    metric = MeanAveragePrecision(
        iou_thresholds=default_iou_thresholds.tolist(),
        max_dets=default_max_dets,
        area_ranges=default_area_ranges,
    )
    metric.update(gt_annotations, pred_annotations)
    results = metric.compute()

    # Test AP metrics
    assert abs(results["ap"] - old_eval.stats[0]) < 0.1
    assert abs(results["ap_0.5"] - old_eval.stats[1]) < 0.1
    assert abs(results["ap_0.75"] - old_eval.stats[2]) < 0.1


def test_ar_metrics(
    gt_annotations,
    pred_annotations,
    old_eval,
    default_iou_thresholds,
    default_max_dets,
    default_area_ranges,
):
    from sane_coco.metrics import MeanAveragePrecision

    # Run our evaluation with the same parameters as COCOeval
    metric = MeanAveragePrecision(
        iou_thresholds=default_iou_thresholds.tolist(),
        max_dets=default_max_dets,
        area_ranges=default_area_ranges,
    )
    metric.update(gt_annotations, pred_annotations)
    results = metric.compute()

    # Test AR metrics
    assert abs(results["ar"] - old_eval.stats[8]) < 0.1
    # Note: COCOeval doesn't directly provide AR at specific IoU thresholds in stats
    # So we'll just check that our AR values are reasonable (between 0 and 1)
    assert 0 <= results["ar_0.5"] <= 1
    assert 0 <= results["ar_0.75"] <= 1

    # Test AR by object size
    assert "ar_small" in results
    assert "ar_medium" in results
    assert "ar_large" in results


def test_max_detections(gt_annotations, pred_annotations, old_eval):
    from sane_coco.metrics import MeanAveragePrecision

    # Test with different max detections
    for max_dets in old_eval.params.maxDets:
        metric = MeanAveragePrecision(max_dets=max_dets)
        metric.update(gt_annotations, pred_annotations)
        results = metric.compute()
        assert "ap" in results
        assert "ar" in results


def test_per_category_evaluation(
    dataset, gt_annotations, pred_annotations, old_coco, pred_data
):
    from sane_coco.metrics import MeanAveragePrecision

    # First, set up pycocotools per-category evaluation
    old_eval_per_cat = COCOeval(old_coco, old_coco.loadRes(pred_data["annotations"]))
    old_eval_per_cat.params.iouType = "bbox"
    old_eval_per_cat.params.useCats = 1  # Enable per-category evaluation
    old_eval_per_cat.evaluate()
    old_eval_per_cat.accumulate()

    # Get per-category results from pycocotools
    cat_ids = old_eval_per_cat.params.catIds

    # Test per-category evaluation in our implementation
    for cat_id in cat_ids:
        cat_name = dataset.get_category_by_id(cat_id).name

        # Filter ground truth and predictions for this category
        cat_gt = []
        for img_gt in gt_annotations:
            cat_gt.append([ann for ann in img_gt if ann["category"] == cat_name])

        cat_pred = []
        for img_pred in pred_annotations:
            cat_pred.append([ann for ann in img_pred if ann["category"] == cat_name])

        # Run evaluation for this category
        metric = MeanAveragePrecision()
        metric.update(cat_gt, cat_pred)
        cat_results = metric.compute()

        # Verify we get results for this category
        assert "ap" in cat_results
        assert "ar" in cat_results


def test_area_based_evaluation(gt_annotations, pred_annotations, default_area_ranges):
    from sane_coco.metrics import MeanAveragePrecision

    # Test area-based evaluation
    metric = MeanAveragePrecision(area_ranges=default_area_ranges)
    metric.update(gt_annotations, pred_annotations)
    results = metric.compute()

    # Verify we have results for each area range
    assert "ar_small" in results
    assert "ar_medium" in results
    assert "ar_large" in results

    # Check that values are in valid range
    assert 0 <= results["ar_small"] <= 1
    assert 0 <= results["ar_medium"] <= 1
    assert 0 <= results["ar_large"] <= 1


def test_average_precision(sample_data):
    from sane_coco.metrics import MeanAveragePrecision

    dataset = COCODataset.from_dict(sample_data)

    predictions = []
    for img in dataset.images:
        for ann in img.annotations:
            x, y, w, h = ann.bbox.xywh
            pred = {
                "image_id": img.id,
                "category_id": ann.category.id,
                "bbox": [x + 2, y - 3, w + 1, h - 2],
                "score": 0.85,
            }
            predictions.append(pred)

        if img.file_name == "000000001.jpg":
            dog_category = next(
                c for c in dataset.categories.values() if c.name == "dog"
            )
            predictions.append(
                {
                    "image_id": img.id,
                    "category_id": dog_category.id,
                    "bbox": [400, 300, 50, 40],
                    "score": 0.65,
                }
            )

    gt_annotations = dataset.get_annotation_dicts()

    pred_annotations = []
    for img in dataset.images:
        img_pred = []
        for pred in predictions:
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

    metric = MeanAveragePrecision()
    metric.update(gt_annotations, pred_annotations)
    results = metric.compute()

    assert "ap" in results
    assert "ap_0.5" in results
    assert "ap_0.75" in results


def test_complex_queries(sample_data):
    dataset = COCODataset.from_dict(sample_data)

    images_with_dog_and_cat = [
        img
        for img in dataset.images
        if "dog" in [ann.category.name for ann in img.annotations]
        and "cat" in [ann.category.name for ann in img.annotations]
    ]

    images_with_dog = [
        img
        for img in dataset.images
        if any(ann.category.name == "dog" for ann in img.annotations)
    ]

    images_with_dog_or_cat = [
        img
        for img in dataset.images
        if any(ann.category.name == "dog" for ann in img.annotations)
        or any(ann.category.name == "cat" for ann in img.annotations)
    ]

    crowded_images = [
        img
        for img in dataset.images
        if len([ann for ann in img.annotations if ann.category.name == "person"]) >= 2
    ]

    animal_annotations = [
        ann for ann in dataset.annotations if ann.category.name in ["dog", "cat"]
    ]

    assert len(images_with_dog_and_cat) == 0
    assert len(images_with_dog) == 1
    assert len(images_with_dog_or_cat) == 2
    assert len(crowded_images) == 0
    assert len(animal_annotations) == 2


def test_missing_image_fields():
    with pytest.raises(
        ValueError, match="Image missing required fields: id, width, height"
    ):
        COCODataset.from_dict(
            {"images": [{"file_name": "test.jpg"}], "categories": [], "annotations": []}
        )


def test_missing_category_fields():
    with pytest.raises(
        ValueError, match="Category missing required fields: supercategory"
    ):
        COCODataset.from_dict(
            {
                "images": [
                    {"id": 1, "file_name": "test.jpg", "width": 100, "height": 100}
                ],
                "categories": [{"id": 1, "name": "test"}],
                "annotations": [],
            }
        )


def test_invalid_bbox_format():
    with pytest.raises(ValueError, match="Invalid bbox format"):
        COCODataset.from_dict(
            {
                "images": [
                    {"id": 1, "file_name": "test.jpg", "width": 100, "height": 100}
                ],
                "categories": [{"id": 1, "name": "test", "supercategory": "test"}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 2, 3]}
                ],
            }
        )


def test_invalid_bbox_dimensions():
    with pytest.raises(ValueError, match="Invalid bbox dimensions"):
        COCODataset.from_dict(
            {
                "images": [
                    {"id": 1, "file_name": "test.jpg", "width": 100, "height": 100}
                ],
                "categories": [{"id": 1, "name": "test", "supercategory": "test"}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, -5, 20]}
                ],
            }
        )


def test_invalid_image_reference():
    with pytest.raises(
        ValueError, match="Annotation 1 references non-existent image 999"
    ):
        COCODataset.from_dict(
            {
                "images": [
                    {"id": 1, "file_name": "test.jpg", "width": 100, "height": 100}
                ],
                "categories": [{"id": 1, "name": "test", "supercategory": "test"}],
                "annotations": [
                    {"id": 1, "image_id": 999, "category_id": 1, "bbox": [1, 2, 3, 4]}
                ],
            }
        )


def test_invalid_category_reference():
    with pytest.raises(
        ValueError, match="Annotation 1 references non-existent category 999"
    ):
        COCODataset.from_dict(
            {
                "images": [
                    {"id": 1, "file_name": "test.jpg", "width": 100, "height": 100}
                ],
                "categories": [{"id": 1, "name": "test", "supercategory": "test"}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 999, "bbox": [1, 2, 3, 4]}
                ],
            }
        )


def test_duplicate_image_ids():
    with pytest.raises(ValueError, match="Duplicate image id: 1"):
        COCODataset.from_dict(
            {
                "images": [
                    {"id": 1, "file_name": "test1.jpg", "width": 100, "height": 100},
                    {"id": 1, "file_name": "test2.jpg", "width": 100, "height": 100},
                ],
                "categories": [],
                "annotations": [],
            }
        )


def test_duplicate_category_ids():
    with pytest.raises(ValueError, match="Duplicate category id: 1"):
        COCODataset.from_dict(
            {
                "images": [],
                "categories": [
                    {"id": 1, "name": "test1", "supercategory": "test"},
                    {"id": 1, "name": "test2", "supercategory": "test"},
                ],
                "annotations": [],
            }
        )


def test_duplicate_annotation_ids():
    with pytest.raises(ValueError, match="Duplicate annotation id: 1"):
        COCODataset.from_dict(
            {
                "images": [
                    {"id": 1, "file_name": "test.jpg", "width": 100, "height": 100},
                    {"id": 2, "file_name": "test2.jpg", "width": 100, "height": 100},
                ],
                "categories": [{"id": 1, "name": "test", "supercategory": "test"}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
                    {"id": 1, "image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10]},
                ],
            }
        )


def test_invalid_image_dimensions():
    with pytest.raises(ValueError, match="Invalid image dimensions: 0x100"):
        COCODataset.from_dict(
            {
                "images": [
                    {"id": 1, "file_name": "test.jpg", "width": 0, "height": 100}
                ],
                "categories": [],
                "annotations": [],
            }
        )


def test_missing_annotation_fields():
    with pytest.raises(ValueError, match="Annotation missing required fields: bbox"):
        COCODataset.from_dict(
            {
                "images": [
                    {"id": 1, "file_name": "test.jpg", "width": 100, "height": 100}
                ],
                "categories": [{"id": 1, "name": "test", "supercategory": "test"}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1}],
            }
        )
