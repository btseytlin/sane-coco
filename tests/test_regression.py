import json
import numpy as np
from pathlib import Path
import pytest
from pycocotools.coco import COCO as OldCOCO
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


# def test_evaluation(sample_data):
#     pred_data = {
#         "annotations": [
#             {
#                 "id": 1,
#                 "image_id": 1,
#                 "category_id": 1,
#                 "bbox": [102, 98, 48, 102],
#                 "score": 0.9,
#             },
#             {
#                 "id": 2,
#                 "image_id": 1,
#                 "category_id": 2,
#                 "bbox": [198, 152, 82, 58],
#                 "score": 0.8,
#             },
#             {
#                 "id": 3,
#                 "image_id": 2,
#                 "category_id": 1,
#                 "bbox": [305, 195, 38, 92],
#                 "score": 0.95,
#             },
#         ]
#     }

#     gt_dataset = COCODataset.from_dict(sample_data)
#     pred_dataset = COCODataset.from_dict(pred_data)
#     results = evaluate(gt_dataset, pred_dataset)

#     old_coco = OldCOCO()
#     old_coco.dataset = sample_data
#     old_eval = OldCOCO.COCOeval(old_coco, pred_data)
#     old_eval.evaluate()
#     old_eval.accumulate()
#     old_eval.summarize()

#     assert abs(results.ap - old_eval.stats[0]) < 1e-6
#     assert abs(results.ap50 - old_eval.stats[1]) < 1e-6
#     assert abs(results.ap75 - old_eval.stats[2]) < 1e-6


def test_complex_queries(sample_data):
    dataset = COCODataset.from_dict(sample_data)

    dog_cat_images = [
        img
        for img in dataset.images
        if any(ann.category.name == "dog" for ann in img.annotations)
        and any(ann.category.name == "cat" for ann in img.annotations)
    ]

    crowded_images = [
        img
        for img in dataset.images
        if sum(1 for ann in img.annotations if ann.category.name == "person") >= 2
    ]

    animal_annotations = [
        ann
        for img in dataset.images
        for ann in img.annotations
        if ann.category.name in ["dog", "cat"]
    ]
