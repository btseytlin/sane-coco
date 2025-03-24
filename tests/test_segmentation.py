# import numpy as np
# import pytest
# from sane_coco import COCODataset, Mask, RLE, Polygon


# @pytest.fixture
# def rle_data():
#     return {"counts": [0, 10, 5, 5, 5, 5], "size": [10, 10]}


# @pytest.fixture
# def polygon_data():
#     return [10, 10, 20, 10, 20, 20, 10, 20]


# @pytest.fixture
# def binary_mask():
#     mask = np.zeros((10, 10), dtype=bool)
#     mask[2:7, 2:7] = True
#     return mask


# @pytest.fixture
# def segmentation_dataset():
#     return {
#         "images": [{"id": 1, "file_name": "test.jpg", "width": 100, "height": 100}],
#         "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
#         "annotations": [
#             {
#                 "id": 1,
#                 "image_id": 1,
#                 "category_id": 1,
#                 "bbox": [10, 10, 50, 50],
#                 "segmentation": [[10, 10, 60, 10, 60, 60, 10, 60]],
#                 "area": 2500,
#                 "iscrowd": 0,
#             },
#             {
#                 "id": 2,
#                 "image_id": 1,
#                 "category_id": 1,
#                 "bbox": [30, 30, 40, 40],
#                 "segmentation": {
#                     # Create an RLE that overlaps with the polygon
#                     # First 3000 pixels (30 rows) are 0, then 4800 pixels are 1
#                     "counts": [3000, 4800],
#                     "size": [100, 100],
#                 },
#                 "area": 4800,
#                 "iscrowd": 1,
#             },
#         ],
#     }


# def test_crowd_annotations(segmentation_dataset):
#     dataset = COCODataset.from_dict(segmentation_dataset)

#     crowd_anns = [ann for ann in dataset.annotations if ann.iscrowd]
#     non_crowd_anns = [ann for ann in dataset.annotations if not ann.iscrowd]

#     assert len(crowd_anns) == 1
#     assert len(non_crowd_anns) == 1

#     crowd_ann = crowd_anns[0]
#     assert isinstance(crowd_ann.segmentation, RLE)
#     assert crowd_ann.area == 4800  # Updated to match the fixture

#     non_crowd_ann = non_crowd_anns[0]
#     assert isinstance(non_crowd_ann.segmentation, Polygon)
#     assert non_crowd_ann.area == 2500


# def test_crowd_overlap_computation(segmentation_dataset):
#     dataset = COCODataset.from_dict(segmentation_dataset)

#     ann1, ann2 = dataset.annotations

#     # Test overlap between crowd and non-crowd
#     overlap = ann1.compute_iou(ann2)
#     assert 0 <= overlap <= 1

#     # Test that crowd vs crowd uses area-based IoU
#     ann2.iscrowd = 0
#     overlap_no_crowd = ann1.compute_iou(ann2)
#     assert overlap != overlap_no_crowd


# def test_segmentation_validation():
#     with pytest.raises(ValueError, match="Invalid segmentation format"):
#         COCODataset.from_dict(
#             {
#                 "images": [
#                     {"id": 1, "file_name": "test.jpg", "width": 100, "height": 100}
#                 ],
#                 "categories": [{"id": 1, "name": "test", "supercategory": "test"}],
#                 "annotations": [
#                     {
#                         "id": 1,
#                         "image_id": 1,
#                         "category_id": 1,
#                         "bbox": [0, 0, 10, 10],
#                         "segmentation": "invalid",
#                     }
#                 ],
#             }
#         )


# def test_empty_segmentation():
#     with pytest.raises(ValueError, match="Empty segmentation"):
#         COCODataset.from_dict(
#             {
#                 "images": [
#                     {"id": 1, "file_name": "test.jpg", "width": 100, "height": 100}
#                 ],
#                 "categories": [{"id": 1, "name": "test", "supercategory": "test"}],
#                 "annotations": [
#                     {
#                         "id": 1,
#                         "image_id": 1,
#                         "category_id": 1,
#                         "bbox": [0, 0, 10, 10],
#                         "segmentation": [],
#                     }
#                 ],
#             }
#         )


# def test_invalid_polygon_points():
#     with pytest.raises(ValueError, match="Invalid polygon points"):
#         COCODataset.from_dict(
#             {
#                 "images": [
#                     {"id": 1, "file_name": "test.jpg", "width": 100, "height": 100}
#                 ],
#                 "categories": [{"id": 1, "name": "test", "supercategory": "test"}],
#                 "annotations": [
#                     {
#                         "id": 1,
#                         "image_id": 1,
#                         "category_id": 1,
#                         "bbox": [0, 0, 10, 10],
#                         "segmentation": [[1, 2, 3]],  # Not enough points
#                     }
#                 ],
#             }
#         )


# def test_invalid_rle_format():
#     with pytest.raises(ValueError, match="Invalid RLE format"):
#         COCODataset.from_dict(
#             {
#                 "images": [
#                     {"id": 1, "file_name": "test.jpg", "width": 100, "height": 100}
#                 ],
#                 "categories": [{"id": 1, "name": "test", "supercategory": "test"}],
#                 "annotations": [
#                     {
#                         "id": 1,
#                         "image_id": 1,
#                         "category_id": 1,
#                         "bbox": [0, 0, 10, 10],
#                         "segmentation": {"counts": [1, 2, 3]},  # Missing size
#                     }
#                 ],
#             }
#         )
