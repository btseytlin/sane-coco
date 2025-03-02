import json
import numpy as np
import pytest
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional, Union, Iterable
from dataclasses import dataclass

from sane_coco.dataset import CocoDataset, Image, Annotation, Category, BBox, Mask, RLE




def load_sample_dataset():
    with open("tests/data/sample_coco.json", "r") as f:
        data = json.load(f)
    return CocoDataset.from_dict(data)


class TestDataclassDesign:
    def test_image_dataclass(self):
        dataset = load_sample_dataset()
        img = dataset.images[0]
        
        assert isinstance(img.id, int)
        assert isinstance(img.width, int)
        assert isinstance(img.height, int)
        assert isinstance(img.file_name, str)
        
        assert hasattr(img, "annotations")
        assert isinstance(img.annotations, Iterable)
        assert hasattr(img, "path")
    
    def test_category_dataclass(self):
        dataset = load_sample_dataset()
        cat = dataset.categories[0]
        
        assert isinstance(cat.id, int)
        assert isinstance(cat.name, str)
        
        if hasattr(cat, "supercategory"):
            assert isinstance(cat.supercategory, (str, type(None)))
        
        assert hasattr(cat, "annotations")
        assert isinstance(cat.annotations, Iterable)
    
    def test_annotation_dataclass(self):
        dataset = load_sample_dataset()
        ann = dataset.annotations[0]
        
        assert isinstance(ann.id, int)
        assert isinstance(ann.bbox, BBox)
        assert isinstance(ann.area, (int, float))
        assert isinstance(ann.is_crowd, bool)
        
        assert isinstance(ann.image, Image)
        assert isinstance(ann.category, Category)
    
    def test_bbox_dataclass(self):
        dataset = load_sample_dataset()
        ann = dataset.annotations[0]
        bbox = ann.bbox
        
        assert isinstance(bbox, BBox)
        assert isinstance(bbox.x, (int, float))
        assert isinstance(bbox.y, (int, float))
        assert isinstance(bbox.width, (int, float))
        assert isinstance(bbox.height, (int, float))
        
        assert isinstance(bbox.area, (int, float))
        assert bbox.area == bbox.width * bbox.height
        
        xyxy = bbox.xyxy
        assert isinstance(xyxy, tuple)
        assert len(xyxy) == 4
        assert xyxy == (bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height)
    
    def test_bbox_contains(self):
        bbox = BBox(x=100, y=100, width=200, height=300)
        
        assert (200, 250) in bbox
        assert (99, 100) not in bbox
        assert (100, 99) not in bbox
        assert (301, 100) not in bbox
        assert (100, 401) not in bbox
    
    def test_bbox_iou(self):
        bbox1 = BBox(x=100, y=100, width=100, height=100)
        bbox2 = BBox(x=150, y=150, width=100, height=100)
        bbox3 = BBox(x=300, y=300, width=100, height=100)
        
        assert bbox1.iou(bbox1) == 1.0
        assert 0.0 < bbox1.iou(bbox2) < 1.0
        assert bbox1.iou(bbox3) == 0.0
    
    def test_mask_dataclass(self):
        dataset = load_sample_dataset()
        
        for ann in dataset.annotations:
            if ann.segmentation:
                mask = ann.mask
                assert isinstance(mask, Mask)
                
                assert isinstance(mask.array, np.ndarray)
                assert mask.shape == (ann.image.height, ann.image.width)
                assert mask.array.dtype == np.uint8
                assert np.unique(mask.array).tolist() == [0, 1]
                
                assert mask.area > 0
                assert mask.area == np.sum(mask.array)
                
                rle = mask.rle
                assert isinstance(rle, RLE)
                break
    
    def test_mask_zeros(self):
        mask = Mask.zeros(480, 640)
        assert mask.shape == (480, 640)
        assert mask.array.dtype == np.uint8
        assert np.all(mask.array == 0)
        assert mask.area == 0


class TestDirectReferences:
    def test_annotation_references(self):
        dataset = load_sample_dataset()
        ann = dataset.annotations[0]
        
        assert isinstance(ann.image, Image)
        assert isinstance(ann.category, Category)
        
        assert ann in ann.image.annotations
        assert ann in ann.category.annotations
    
    def test_image_path(self):
        dataset = load_sample_dataset()
        img = dataset.images[0]
        
        assert img.path is None
        
        dataset.image_dir = "/path/to/images"
        assert img.path == Path("/path/to/images") / img.file_name


class TestAccessByID:
    def test_access_by_id(self):
        dataset = load_sample_dataset()
        
        first_image = dataset.images[0]
        first_category = dataset.categories[0]
        first_annotation = dataset.annotations[0]
        
        assert dataset.get_image_by_id(first_image.id) == first_image
        assert dataset.get_category_by_id(first_category.id) == first_category
        assert dataset.get_annotation_by_id(first_annotation.id) == first_annotation
    
    def test_nonexistent_id(self):
        dataset = load_sample_dataset()
        nonexistent_id = max(img.id for img in dataset.images) + 999
        
        with pytest.raises(KeyError):
            dataset.get_image_by_id(nonexistent_id)
        
        with pytest.raises(KeyError):
            dataset.get_category_by_id(nonexistent_id)
        
        with pytest.raises(KeyError):
            dataset.get_annotation_by_id(nonexistent_id)
            
    def test_get_category_by_name(self):
        dataset = load_sample_dataset()
        first_category = dataset.categories[0]
        
        assert dataset.get_category_by_name(first_category.name) == first_category
        
        with pytest.raises(KeyError):
            dataset.get_category_by_name("nonexistent_category")


class TestSequenceOperations:
    def test_sequence_properties(self):
        dataset = load_sample_dataset()
        
        assert len(dataset.images) > 0
        assert len(dataset.categories) > 0
        assert len(dataset.annotations) > 0
    
    def test_slicing(self):
        dataset = load_sample_dataset()
        
        first_two_images = dataset.images[:2]
        assert len(first_two_images) == min(2, len(dataset.images))
        
        single_image = dataset.images[0]
        assert isinstance(single_image, Image)
        
        last_annotation = dataset.annotations[-1]
        assert last_annotation == dataset.annotations[len(dataset.annotations) - 1]
    
    def test_iteration(self):
        dataset = load_sample_dataset()
        
        images_list = list(dataset.images)
        assert len(images_list) == len(dataset.images)
        
        categories_list = list(dataset.categories)
        assert len(categories_list) == len(dataset.categories)
        
        annotations_list = list(dataset.annotations)
        assert len(annotations_list) == len(dataset.annotations)
    
    def test_empty_collections(self):
        dataset = CocoDataset()
        
        assert len(dataset.images) == 0
        assert len(dataset.categories) == 0
        assert len(dataset.annotations) == 0
        
        with pytest.raises(IndexError):
            dataset.images[0]


class TestPythonIdioms:
    def test_list_comprehensions(self):
        dataset = load_sample_dataset()
        
        person_anns = [ann for ann in dataset.annotations 
                      if ann.category.name == "person"]
        assert all(ann.category.name == "person" for ann in person_anns)
        
        large_boxes = [ann for ann in dataset.annotations 
                      if ann.bbox.area > 50000]
        assert all(ann.bbox.area > 50000 for ann in large_boxes)
    
    def test_filtering_with_built_ins(self):
        dataset = load_sample_dataset()
        
        car_category = next(cat for cat in dataset.categories if cat.name == "car")
        car_anns = list(filter(lambda ann: ann.category == car_category, dataset.annotations))
        assert all(ann.category == car_category for ann in car_anns)
    
    def test_dictionary_comprehensions(self):
        dataset = load_sample_dataset()
        
        id_to_image = {img.id: img for img in dataset.images}
        assert len(id_to_image) == len(dataset.images)
        assert all(id_to_image[img.id] == img for img in dataset.images)
        
        name_to_category = {cat.name: cat for cat in dataset.categories}
        assert "person" in name_to_category
        assert name_to_category["person"].name == "person"


class TestQueryOperations:
    def test_query_by_attributes(self):
        dataset = load_sample_dataset()
        
        person_category = next(cat for cat in dataset.categories if cat.name == "person")
        
        images_with_people = {ann.image for ann in person_category.annotations}
        assert len(images_with_people) > 0
        
        large_annotations = [ann for ann in dataset.annotations if ann.area > 50000]
        assert all(ann.area > 50000 for ann in large_annotations)
    
    def test_query_with_generators(self):
        dataset = load_sample_dataset()
        
        large_boxes = (ann for ann in dataset.annotations if ann.bbox.area > 40000)
        assert all(ann.bbox.area > 40000 for ann in large_boxes)
        
        img_640x480 = (img for img in dataset.images 
                      if img.width == 640 and img.height == 480)
        assert all(img.width == 640 and img.height == 480 for img in img_640x480)


class TestCollectionOperations:
    def test_set_operations(self):
        dataset = load_sample_dataset()
        
        person_category = next(cat for cat in dataset.categories if cat.name == "person")
        car_category = next(cat for cat in dataset.categories if cat.name == "car")
        
        images_with_people = {ann.image for ann in person_category.annotations}
        images_with_cars = {ann.image for ann in car_category.annotations}
        
        images_with_both = images_with_people.intersection(images_with_cars)
        assert all(img in images_with_people and img in images_with_cars 
                 for img in images_with_both)
        
        images_with_people_only = images_with_people - images_with_cars
        assert all(img in images_with_people and img not in images_with_cars 
                 for img in images_with_people_only)


class TestMaskOperations:
    def test_mask_boolean_operations(self):
        mask1 = Mask.zeros(100, 100)
        mask1.array[10:50, 10:50] = 1  # Top-left square
        
        mask2 = Mask.zeros(100, 100)
        mask2.array[30:70, 30:70] = 1  # Bottom-right square, overlapping
        
        intersection = mask1 & mask2
        assert isinstance(intersection, Mask)
        assert intersection.area < mask1.area
        assert intersection.area < mask2.area
        
        union = mask1 | mask2
        assert isinstance(union, Mask)
        assert union.area > mask1.area
        assert union.area > mask2.area
        
        iou = mask1.iou(mask2)
        assert 0.0 < iou < 1.0
        assert iou == intersection.area / union.area
    
    def test_mask_shape_mismatch(self):
        mask1 = Mask.zeros(100, 100)
        mask2 = Mask.zeros(200, 200)
        
        with pytest.raises(ValueError):
            mask1 & mask2
        
        with pytest.raises(ValueError):
            mask1 | mask2
    
    def test_rle_conversion(self):
        pytest.skip("RLE implementation is a placeholder, skipping full validation")
        
        dataset = load_sample_dataset()
        
        for ann in dataset.annotations:
            if ann.segmentation:
                mask = ann.mask
                rle = mask.rle
                
                assert isinstance(rle, RLE)
                assert hasattr(rle, "counts")
                assert rle.shape == mask.shape
                
                mask_from_rle = rle.mask
                assert isinstance(mask_from_rle, Mask)
                break


class TestSerialization:
    def test_dataset_to_dict(self):
        dataset = load_sample_dataset()
        data_dict = dataset.to_dict()
        
        assert "images" in data_dict
        assert "categories" in data_dict
        assert "annotations" in data_dict
        
        assert len(data_dict["images"]) == len(dataset.images)
        assert len(data_dict["categories"]) == len(dataset.categories)
        assert len(data_dict["annotations"]) == len(dataset.annotations)
        
        first_img = dataset.images[0]
        first_img_dict = next(img for img in data_dict["images"] if img["id"] == first_img.id)
        assert first_img_dict["width"] == first_img.width
        assert first_img_dict["height"] == first_img.height
        assert first_img_dict["file_name"] == first_img.file_name
        
        first_cat = dataset.categories[0]
        first_cat_dict = next(cat for cat in data_dict["categories"] if cat["id"] == first_cat.id)
        assert first_cat_dict["name"] == first_cat.name
        if first_cat.supercategory:
            assert first_cat_dict["supercategory"] == first_cat.supercategory
        
        first_ann = dataset.annotations[0]
        first_ann_dict = next(ann for ann in data_dict["annotations"] if ann["id"] == first_ann.id)
        assert first_ann_dict["image_id"] == first_ann.image_id
        assert first_ann_dict["category_id"] == first_ann.category_id
        assert first_ann_dict["bbox"] == [first_ann.bbox.x, first_ann.bbox.y, first_ann.bbox.width, first_ann.bbox.height]
        assert first_ann_dict["area"] == first_ann.area
        assert first_ann_dict["iscrowd"] == int(first_ann.is_crowd)
        if first_ann.segmentation:
            assert first_ann_dict["segmentation"] == first_ann.segmentation
    
    def test_round_trip_serialization(self):
        original_dataset = load_sample_dataset()
        data_dict = original_dataset.to_dict()
        new_dataset = CocoDataset.from_dict(data_dict)
        
        assert len(new_dataset.images) == len(original_dataset.images)
        assert len(new_dataset.categories) == len(original_dataset.categories)
        assert len(new_dataset.annotations) == len(original_dataset.annotations)
        
        first_img_id = original_dataset.images[0].id
        first_cat_id = original_dataset.categories[0].id
        first_ann_id = original_dataset.annotations[0].id
        
        assert new_dataset.get_image_by_id(first_img_id).file_name == original_dataset.get_image_by_id(first_img_id).file_name
        assert new_dataset.get_category_by_id(first_cat_id).name == original_dataset.get_category_by_id(first_cat_id).name
        assert new_dataset.get_annotation_by_id(first_ann_id).area == original_dataset.get_annotation_by_id(first_ann_id).area


class TestEvaluation:
    def test_simple_detection_evaluation(self):
        from sane_coco.eval import evaluate
        
        dataset = load_sample_dataset()
        
        # Create ground truth objects
        gt_bboxes = [ann.bbox for ann in dataset.annotations[:2]]
        gt_categories = [ann.category for ann in dataset.annotations[:2]]
        
        # Create predictions
        pred_bboxes = [
            BBox(x=100, y=100, width=200, height=200),
            BBox(x=300, y=300, width=100, height=100)
        ]
        pred_categories = [dataset.categories[0], dataset.categories[0]]
        pred_scores = [0.9, 0.8]
        
        # Test the simplified API
        metrics = evaluate(
            gt_bboxes=gt_bboxes,
            gt_categories=gt_categories,
            pred_bboxes=pred_bboxes,
            pred_categories=pred_categories,
            pred_scores=pred_scores,
            iou_thresholds=[0.5, 0.75]
        )
        
        assert "precision" in metrics
        assert "recall" in metrics
        assert "map" in metrics
        assert "map_per_category" in metrics
        
        assert isinstance(metrics["precision"], Dict)
        assert isinstance(metrics["recall"], Dict)
        assert isinstance(metrics["map"], float)
        assert isinstance(metrics["map_per_category"], Dict)
        
        for iou_threshold in [0.5, 0.75]:
            assert iou_threshold in metrics["precision"]
            assert iou_threshold in metrics["recall"]
            assert iou_threshold in metrics["map_per_category"]
            
            assert isinstance(metrics["precision"][iou_threshold], np.ndarray)
            assert isinstance(metrics["recall"][iou_threshold], np.ndarray)
    
    def test_segmentation_evaluation(self):
        from sane_coco.eval import evaluate
        
        dataset = load_sample_dataset()
        
        # Find annotations with segmentation
        anns_with_masks = [ann for ann in dataset.annotations if ann.segmentation]
        if not anns_with_masks:
            pytest.skip("No annotations with segmentation found")
        
        # Create ground truth objects
        gt_masks = [ann.mask for ann in anns_with_masks]
        gt_categories = [ann.category for ann in anns_with_masks]
        
        # Create predictions
        pred_mask = Mask.zeros(480, 640)  # Assuming standard dimensions
        pred_mask.array[100:200, 100:200] = 1
        
        pred_masks = [pred_mask]
        pred_categories = [anns_with_masks[0].category]
        pred_scores = [0.9]
        
        # Test the simplified API for segmentation
        metrics = evaluate(
            gt_masks=gt_masks,
            gt_categories=gt_categories,
            pred_masks=pred_masks,
            pred_categories=pred_categories,
            pred_scores=pred_scores,
            iou_thresholds=[0.5],
            eval_bbox=False,
            eval_segm=True
        )
        
        assert "precision" in metrics
        assert "recall" in metrics
        assert "map" in metrics
        assert "map_per_category" in metrics
    
    def test_combined_evaluation(self):
        from sane_coco.eval import evaluate
        
        dataset = load_sample_dataset()
        
        # Create ground truth objects
        gt_bboxes = [ann.bbox for ann in dataset.annotations[:2]]
        gt_categories = [ann.category for ann in dataset.annotations[:2]]
        
        # Find annotations with segmentation
        anns_with_masks = [ann for ann in dataset.annotations if ann.segmentation]
        gt_masks = [ann.mask for ann in anns_with_masks] if anns_with_masks else []
        
        # Create predictions
        pred_bboxes = [
            BBox(x=100, y=100, width=200, height=200),
            BBox(x=300, y=300, width=100, height=100)
        ]
        
        pred_mask = Mask.zeros(480, 640)
        pred_mask.array[100:200, 100:200] = 1
        
        pred_masks = [pred_mask] if gt_masks else []
        pred_categories = [dataset.categories[0], dataset.categories[0]]
        pred_scores = [0.9, 0.8]
        
        # Test using annotations directly
        gt_annotations = dataset.annotations[:2]
        predictions = [
            (BBox(x=100, y=100, width=200, height=200), dataset.categories[0], 0.9, None),
            (BBox(x=300, y=300, width=100, height=100), dataset.categories[0], 0.8, None)
        ]
        
        metrics_direct = evaluate(
            gt_annotations=gt_annotations,
            predictions=predictions,
            iou_thresholds=[0.5],
            return_per_class=True
        )
        
        assert "precision" in metrics_direct
        assert "recall" in metrics_direct
        assert "map" in metrics_direct
        
        # Test with separate lists
        metrics = evaluate(
            gt_bboxes=gt_bboxes,
            gt_categories=gt_categories,
            gt_masks=gt_masks[:1] if gt_masks else None,
            pred_bboxes=pred_bboxes,
            pred_categories=pred_categories,
            pred_scores=pred_scores,
            pred_masks=pred_masks[:1] if pred_masks else None,
            iou_thresholds=[0.5],
            return_per_class=True
        )
        
        assert "precision" in metrics
        assert "recall" in metrics
        assert "map" in metrics
        
        # Check per-class results are returned
        assert isinstance(metrics["per_class"], Dict)
        
    def test_traditional_detection_evaluation(self):
        from sane_coco.eval import evaluate_detections
        
        dataset = load_sample_dataset()
        sample_img = dataset.images[0]
        sample_cat = dataset.categories[0]
        
        # Legacy format for backward compatibility
        predictions = [
            {
                "bbox": BBox(x=100, y=100, width=200, height=200),
                "category_id": sample_cat.id,
                "score": 0.9,
                "image_id": sample_img.id
            },
            {
                "bbox": BBox(x=300, y=300, width=100, height=100),
                "category_id": sample_cat.id,
                "score": 0.8,
                "image_id": sample_img.id
            }
        ]
        
        metrics = evaluate_detections(
            predictions=predictions,
            ground_truth=dataset,
            iou_thresholds=[0.5, 0.75],
            max_detections=100
        )
        
        assert "precision" in metrics
        assert "recall" in metrics
        assert "map" in metrics
        assert "map_per_category" in metrics
    


class TestConvenienceProperties:
    def test_image_annotations(self):
        dataset = load_sample_dataset()
        
        for img in dataset.images:
            anns = list(img.annotations)
            assert all(ann.image == img for ann in anns)
    
    def test_category_annotations(self):
        dataset = load_sample_dataset()
        
        for cat in dataset.categories:
            anns = list(cat.annotations)
            assert all(ann.category == cat for ann in anns)
    
    def test_annotation_properties(self):
        dataset = load_sample_dataset()
        
        for ann in dataset.annotations:
            assert ann.image in dataset.images
            assert ann.category in dataset.categories
            assert ann.bbox.area > 0
            if ann.segmentation:
                assert ann.mask.area > 0


class TestStandardProtocols:
    def test_container_protocols(self):
        dataset = load_sample_dataset()
        
        assert dataset.images[0] in dataset.images
        assert dataset.categories[0] in dataset.categories
        assert dataset.annotations[0] in dataset.annotations
    
    def test_equality_protocols(self):
        dataset = load_sample_dataset()
        
        assert dataset.images[0] == dataset.images[0]
        assert dataset.categories[0] == dataset.categories[0]
        assert dataset.annotations[0] == dataset.annotations[0]
        
        assert dataset.images[0] != dataset.images[1] if len(dataset.images) > 1 else True
    
    def test_string_representation(self):
        dataset = load_sample_dataset()
        
        assert str(dataset.images[0]) != ""
        assert str(dataset.categories[0]) != ""
        assert str(dataset.annotations[0]) != ""
        
        assert repr(dataset.images[0]) != ""
        assert repr(dataset.categories[0]) != ""
        assert repr(dataset.annotations[0]) != ""


class TestInitializationAndLoading:
    def test_empty_initialization(self):
        dataset = CocoDataset()
        assert len(dataset.images) == 0
        assert len(dataset.categories) == 0
        assert len(dataset.annotations) == 0
    
    def test_from_dict_initialization(self):
        with open("tests/data/sample_coco.json", "r") as f:
            data = json.load(f)
        
        dataset = CocoDataset.from_dict(data)
        assert len(dataset.images) > 0
        assert len(dataset.categories) > 0
        assert len(dataset.annotations) > 0
    
    def test_image_dir_setting(self):
        dataset = load_sample_dataset()
        assert dataset.image_dir is None
        
        dataset.image_dir = "/path/to/images"
        assert str(dataset.image_dir) == "/path/to/images"
        
        # Test path for a sample image
        img = dataset.images[0]
        assert str(img.path) == f"/path/to/images/{img.file_name}"