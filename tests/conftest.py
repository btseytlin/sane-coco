import pytest


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
