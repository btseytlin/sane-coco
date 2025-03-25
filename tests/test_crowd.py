import numpy as np
import pytest
from sane_coco import COCODataset, Mask, RLE


@pytest.fixture
def crowd_dataset():
    return {
        "images": [{"id": 1, "file_name": "crowd.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 80, 80],
                "segmentation": {
                    # Create an RLE that covers the entire region from (10,10) to (90,90)
                    # First 1000 pixels (10 rows) are 0, then 6400 pixels are 1
                    "counts": [1000, 6400],
                    "size": [100, 100],
                },
                "area": 6400,
                "iscrowd": 1,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [20, 20, 30, 40],
                "segmentation": [[20, 20, 50, 20, 50, 60, 20, 60]],
                "area": 1200,
                "iscrowd": 0,
            },
            {
                "id": 3,
                "image_id": 1,
                "category_id": 1,
                "bbox": [60, 60, 30, 30],
                "segmentation": [[60, 60, 90, 60, 90, 90, 60, 90]],
                "area": 900,
                "iscrowd": 0,
            },
        ],
    }


def test_crowd_area_computation(crowd_dataset):
    dataset = COCODataset.from_dict(crowd_dataset)
    crowd_ann = dataset.annotations[0]

    # Area should match the RLE counts
    assert crowd_ann.area == 6400
    assert crowd_ann.segmentation.area == 6400


def test_crowd_iou_computation(crowd_dataset):
    dataset = COCODataset.from_dict(crowd_dataset)
    crowd_ann = dataset.annotations[0]
    normal_ann = dataset.annotations[1]

    # Debug: Print the RLE counts and size
    print(f"RLE counts: {crowd_ann.segmentation.counts}")
    print(f"RLE size: {crowd_ann.segmentation.size}")

    # Debug: Create masks and check overlap
    img_width = crowd_ann.image.width
    img_height = crowd_ann.image.height
    size = (img_height, img_width)

    crowd_mask = crowd_ann.segmentation.to_mask(size)
    normal_mask = normal_ann.segmentation.to_mask(size)

    print(f"Crowd mask area: {crowd_mask.area}")
    print(f"Normal mask area: {normal_mask.area}")

    intersection = np.logical_and(crowd_mask.array, normal_mask.array).sum()
    print(f"Intersection area: {intersection}")

    # IoU between crowd and normal should use special computation
    iou = crowd_ann.compute_iou(normal_ann)
    print(f"Computed IoU: {iou}")
    assert 0 <= iou <= 1

    # IoU should be proportional to overlap area
    assert iou > 0.15  # The normal annotation is partially inside crowd


def test_multiple_crowd_annotations(crowd_dataset):
    # Add another crowd annotation
    crowd_dataset["annotations"].append(
        {
            "id": 4,
            "image_id": 1,
            "category_id": 1,
            "bbox": [50, 50, 40, 40],
            "segmentation": {"counts": [0, 1600], "size": [100, 100]},
            "area": 1600,
            "iscrowd": 1,
        }
    )

    dataset = COCODataset.from_dict(crowd_dataset)
    crowd_anns = [ann for ann in dataset.annotations if ann.iscrowd]
    assert len(crowd_anns) == 2

    # Test IoU between crowd annotations
    iou = crowd_anns[0].compute_iou(crowd_anns[1])
    assert 0 <= iou <= 1


def test_crowd_mask_operations(crowd_dataset):
    dataset = COCODataset.from_dict(crowd_dataset)
    crowd_ann = dataset.annotations[0]

    # Convert RLE to mask and back
    mask = crowd_ann.segmentation.to_mask()
    rle = mask.to_rle()

    assert isinstance(mask, Mask)
    assert isinstance(rle, RLE)
    assert np.array_equal(mask.array, rle.to_mask().array)


def test_crowd_vs_multiple_annotations(crowd_dataset):
    dataset = COCODataset.from_dict(crowd_dataset)
    crowd_ann = dataset.annotations[0]
    normal_anns = [ann for ann in dataset.annotations if not ann.iscrowd]

    # Compute IoU with each normal annotation
    ious = [crowd_ann.compute_iou(ann) for ann in normal_anns]
    assert all(0 <= iou <= 1 for iou in ious)

    # Both normal annotations overlap with crowd
    assert all(iou > 0 for iou in ious)


def test_crowd_annotation_validation():
    with pytest.raises(ValueError, match="Crowd annotations must use RLE segmentation"):
        COCODataset.from_dict(
            {
                "images": [
                    {"id": 1, "file_name": "test.jpg", "width": 100, "height": 100}
                ],
                "categories": [{"id": 1, "name": "test", "supercategory": "test"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [0, 0, 10, 10],
                        "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]],
                        "iscrowd": 1,
                    }
                ],
            }
        )


def test_crowd_area_validation():
    with pytest.raises(ValueError, match="Crowd annotation area must match RLE area"):
        COCODataset.from_dict(
            {
                "images": [
                    {"id": 1, "file_name": "test.jpg", "width": 100, "height": 100}
                ],
                "categories": [{"id": 1, "name": "test", "supercategory": "test"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [0, 0, 10, 10],
                        "segmentation": {"counts": [0, 100], "size": [100, 100]},
                        "area": 50,
                        "iscrowd": 1,
                    }
                ],
            }
        )
