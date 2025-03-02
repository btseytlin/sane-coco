import json
import numpy as np
import pytest
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional, Union
from dataclasses import dataclass

from sane_coco.dataset import CocoDataset, Image, Annotation, Category, BBox, Mask, RLE


@dataclass
class Prediction:
    bbox: BBox
    category_id: int
    score: float
    segmentation: Optional[Union[List, Dict]] = None


@pytest.fixture
def sample_data():
    with open("tests/data/sample_coco.json", "r") as f:
        return json.load(f)


@pytest.fixture
def sample_results():
    with open("tests/data/sample_results.json", "r") as f:
        return json.load(f)


@pytest.fixture
def dataset(sample_data):
    return CocoDataset.from_dict(sample_data)


class TestDataclassDesign:
    def test_image_dataclass(self, dataset):
        img = dataset.images[0]
        
        assert isinstance(img.id, int)
        assert isinstance(img.width, int)
        assert isinstance(img.height, int)
        assert isinstance(img.file_name, str)
        
        assert hasattr(img, "annotations")
        assert hasattr(img, "id")
        assert hasattr(img, "width")
        assert hasattr(img, "height")
        assert hasattr(img, "file_name")
    
    def test_category_dataclass(self, dataset):
        cat = dataset.categories[0]
        
        assert isinstance(cat.id, int)
        assert isinstance(cat.name, str)
        
        if hasattr(cat, "supercategory"):
            assert isinstance(cat.supercategory, (str, type(None)))
        
        assert hasattr(cat, "annotations")
    
    def test_annotation_dataclass(self, dataset):
        ann = dataset.annotations[0]
        
        assert isinstance(ann.id, int)
        assert isinstance(ann.bbox, BBox)
        assert isinstance(ann.area, (int, float))
        assert isinstance(ann.is_crowd, bool)
        
        assert hasattr(ann, "image")
        assert hasattr(ann, "category")
        assert hasattr(ann, "segmentation")
        
        assert isinstance(ann.image, Image)
        assert isinstance(ann.category, Category)
    
    def test_bbox_dataclass(self, dataset):
        ann = dataset.annotations[0]
        bbox = ann.bbox
        
        assert isinstance(bbox, BBox)
        assert hasattr(bbox, "x")
        assert hasattr(bbox, "y")
        assert hasattr(bbox, "width")
        assert hasattr(bbox, "height")
        
        assert isinstance(bbox.x, (int, float))
        assert isinstance(bbox.y, (int, float))
        assert isinstance(bbox.width, (int, float))
        assert isinstance(bbox.height, (int, float))
        
        xyxy = bbox.to_xyxy()
        assert isinstance(xyxy, tuple)
        assert len(xyxy) == 4
        assert xyxy == (bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height)
        
        assert bbox.area == bbox.width * bbox.height
    
    def test_mask_dataclass(self, dataset):
        for ann in dataset.annotations:
            if ann.segmentation:
                mask = ann.generate_mask()
                assert isinstance(mask, Mask)
                
                assert hasattr(mask, "array")
                assert hasattr(mask, "height")
                assert hasattr(mask, "width")
                
                assert isinstance(mask.array, np.ndarray)
                assert mask.height == ann.image.height
                assert mask.width == ann.image.width
                assert mask.array.shape == (mask.height, mask.width)
                
                assert mask.array.dtype == np.uint8
                assert np.unique(mask.array).tolist() == [0, 1]
                
                rle = mask.to_rle()
                assert isinstance(rle, RLE)
                break


class TestDirectReferences:
    def test_annotation_references(self, dataset):
        ann = dataset.annotations[0]
        
        assert isinstance(ann.image, Image)
        assert isinstance(ann.category, Category)
        
        assert ann in ann.image.annotations
        assert ann in ann.category.annotations


class TestPythonDataStructures:
    def test_dataset_creation(self, sample_data):
        images = [
            Image(id=img["id"], width=img["width"], height=img["height"], 
                  file_name=img["file_name"])
            for img in sample_data["images"]
        ]
        
        categories = [
            Category(id=cat["id"], name=cat["name"], 
                    supercategory=cat.get("supercategory"))
            for cat in sample_data["categories"]
        ]
        
        annotations = []
        for ann_data in sample_data["annotations"]:
            image = next(img for img in images if img.id == ann_data["image_id"])
            category = next(cat for cat in categories if cat.id == ann_data["category_id"])
            
            bbox_data = ann_data["bbox"]
            bbox = BBox(x=bbox_data[0], y=bbox_data[1], width=bbox_data[2], height=bbox_data[3])
            
            annotations.append(
                Annotation(
                    id=ann_data["id"],
                    image=image,
                    category=category,
                    bbox=bbox,
                    area=ann_data["area"],
                    segmentation=ann_data.get("segmentation"),
                    is_crowd=bool(ann_data.get("iscrowd", 0))
                )
            )
        
        dataset = CocoDataset(images=images, annotations=annotations, categories=categories)
        
        assert len(dataset.images) == len(sample_data["images"])
        assert len(dataset.annotations) == len(sample_data["annotations"])
        assert len(dataset.categories) == len(sample_data["categories"])
    
    def test_to_dict_conversion(self, dataset, sample_data):
        data_dict = dataset.to_dict()
        
        assert "images" in data_dict
        assert "annotations" in data_dict
        assert "categories" in data_dict
        
        assert len(data_dict["images"]) == len(sample_data["images"])
        assert len(data_dict["annotations"]) == len(sample_data["annotations"])
        assert len(data_dict["categories"]) == len(sample_data["categories"])


class TestFiltering:
    def test_filter_images(self, dataset):
        img = dataset.images[0]
        cat = dataset.categories[0]
        cat_name = cat.name
        
        filtered_by_id = dataset.filter_images(ids=[img.id])
        assert len(filtered_by_id) == 1
        assert filtered_by_id[0] == img
        
        filtered_by_cat = dataset.filter_images(categories=[cat])
        assert all(cat in [a.category for a in img.annotations] 
                 for img in filtered_by_cat)
        
        filtered_by_cat_name = dataset.filter_images(category_names=[cat_name])
        assert set(img.id for img in filtered_by_cat) == set(img.id for img in filtered_by_cat_name)
    
    def test_filter_annotations(self, dataset):
        img = dataset.images[0]
        cat = dataset.categories[0]
        
        filtered_by_img = dataset.filter_annotations(images=[img])
        assert all(ann.image == img for ann in filtered_by_img)
        
        filtered_by_cat = dataset.filter_annotations(categories=[cat])
        assert all(ann.category == cat for ann in filtered_by_cat)
        
        area_range = [1000, 70000]
        filtered_by_area = dataset.filter_annotations(area_range=area_range)
        assert all(area_range[0] <= ann.area <= area_range[1] for ann in filtered_by_area)
        
        combined = dataset.filter_annotations(
            images=[img], 
            categories=[cat], 
            area_range=area_range
        )
        assert all(ann.image == img for ann in combined)
        assert all(ann.category == cat for ann in combined)
        assert all(area_range[0] <= ann.area <= area_range[1] for ann in combined)
    
    def test_filter_categories(self, dataset):
        cat_id = dataset.categories[0].id
        cat_name = dataset.categories[0].name
        
        filtered_by_id = dataset.filter_categories(ids=[cat_id])
        assert len(filtered_by_id) == 1
        assert filtered_by_id[0].id == cat_id
        
        filtered_by_name = dataset.filter_categories(names=[cat_name])
        assert len(filtered_by_name) == 1
        assert filtered_by_name[0].name == cat_name
        
        if hasattr(dataset.categories[0], "supercategory") and dataset.categories[0].supercategory:
            supercategory = dataset.categories[0].supercategory
            filtered_by_super = dataset.filter_categories(supercategory=supercategory)
            assert all(cat.supercategory == supercategory for cat in filtered_by_super)


class TestProperties:
    def test_bbox_property(self, dataset):
        ann = dataset.annotations[0]
        bbox = ann.bbox
        
        assert isinstance(bbox, BBox)
        assert hasattr(bbox, "x")
        assert hasattr(bbox, "y")
        assert hasattr(bbox, "width")
        assert hasattr(bbox, "height")
        
        xyxy = bbox.to_xyxy()
        assert xyxy == (bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height)
        
        area = bbox.area
        assert area == bbox.width * bbox.height
    
    def test_collection_properties(self, dataset):
        img = dataset.images[0]
        cat = dataset.categories[0]
        
        assert len(img.annotations) > 0
        assert all(isinstance(ann, Annotation) for ann in img.annotations)
        assert all(ann.image == img for ann in img.annotations)
        
        assert len(cat.annotations) > 0
        assert all(isinstance(ann, Annotation) for ann in cat.annotations)
        assert all(ann.category == cat for ann in cat.annotations)


class TestMaskOperations:
    def test_generate_mask(self, dataset):
        for ann in dataset.annotations:
            if ann.segmentation:
                mask = ann.generate_mask()
                assert isinstance(mask, Mask)
                assert mask.height == ann.image.height
                assert mask.width == ann.image.width
                
                assert isinstance(mask.array, np.ndarray)
                assert mask.array.shape == (mask.height, mask.width)
                assert mask.array.dtype == np.uint8
                
                assert mask.intersection(mask).sum() == mask.sum()
                assert mask.union(mask).sum() == mask.sum()
                
                assert mask.iou(mask) == 1.0
                break
    
    def test_generate_rle(self, dataset):
        for ann in dataset.annotations:
            if ann.segmentation:
                rle = ann.generate_rle()
                assert isinstance(rle, RLE)
                assert hasattr(rle, "counts")
                assert hasattr(rle, "size")
                
                assert rle.size == (ann.image.height, ann.image.width)
                
                mask_from_rle = rle.to_mask()
                assert isinstance(mask_from_rle, Mask)
                break


class TestIteration:
    def test_image_iteration(self, dataset):
        image_list = list(dataset.images)
        assert len(image_list) == len(dataset.images)
        
        for i, img in enumerate(dataset.images):
            assert img == dataset.images[i]
    
    def test_annotation_iteration(self, dataset):
        annotation_list = list(dataset.annotations)
        assert len(annotation_list) == len(dataset.annotations)
        
        for i, ann in enumerate(dataset.annotations):
            assert ann == dataset.annotations[i]
    
    def test_category_iteration(self, dataset):
        category_list = list(dataset.categories)
        assert len(category_list) == len(dataset.categories)
        
        for i, cat in enumerate(dataset.categories):
            assert cat == dataset.categories[i]


class TestImmutabilityWithBuilders:
    def test_dataset_copy_with(self, dataset):
        new_images = dataset.images[:1]
        new_dataset = dataset.copy_with(images=new_images)
        
        assert len(new_dataset.images) == 1
        assert len(dataset.images) > len(new_dataset.images)
        
        assert new_dataset.images[0] == dataset.images[0]
        assert new_dataset != dataset
    
    def test_annotation_copy_with(self, dataset):
        ann = dataset.annotations[0]
        old_bbox = ann.bbox
        new_bbox = BBox(x=old_bbox.x + 10, y=old_bbox.y + 10, 
                        width=old_bbox.width, height=old_bbox.height)
        
        new_ann = ann.copy_with(bbox=new_bbox)
        
        assert new_ann.bbox != ann.bbox
        assert new_ann.bbox.x == old_bbox.x + 10
        assert new_ann.bbox.y == old_bbox.y + 10
        assert new_ann.id == ann.id
        assert new_ann.image == ann.image
        assert new_ann.category == ann.category


class TestSklearnStyleEvaluation:
    def test_raw_prediction_evaluation(self, dataset):
        from sane_coco.eval import CocoEvaluator
        
        sample_img = dataset.images[0]
        sample_cat = dataset.categories[0]
        
        raw_predictions = [
            Prediction(
                bbox=BBox(x=100, y=100, width=200, height=200),
                category_id=sample_cat.id,
                score=0.9
            ),
            Prediction(
                bbox=BBox(x=300, y=300, width=100, height=100),
                category_id=sample_cat.id,
                score=0.8
            )
        ]
        
        image_ids = [sample_img.id]
        ground_truth_boxes = [ann.bbox for ann in sample_img.annotations]
        ground_truth_classes = [ann.category.id for ann in sample_img.annotations]
        
        metrics = CocoEvaluator.evaluate_detections(
            ground_truth_boxes=ground_truth_boxes,
            ground_truth_classes=ground_truth_classes,
            predicted_boxes=[p.bbox for p in raw_predictions],
            predicted_classes=[p.category_id for p in raw_predictions],
            predicted_scores=[p.score for p in raw_predictions],
            image_ids=[sample_img.id] * len(raw_predictions),
            category_ids=[c.id for c in dataset.categories]
        )
        
        assert "precision" in metrics
        assert "recall" in metrics
        assert "ap" in metrics
        
        assert isinstance(metrics["precision"], np.ndarray)
        assert isinstance(metrics["recall"], np.ndarray)
        assert isinstance(metrics["ap"], (list, np.ndarray))
    
    def test_raw_segmentation_evaluation(self, dataset):
        from sane_coco.eval import CocoEvaluator
        
        for ann in dataset.annotations:
            if ann.segmentation:
                ground_truth_mask = ann.generate_mask()
                
                predicted_mask = Mask(
                    array=np.zeros((ann.image.height, ann.image.width), dtype=np.uint8),
                    height=ann.image.height,
                    width=ann.image.width
                )
                predicted_mask.array[100:200, 100:200] = 1
                
                metrics = CocoEvaluator.evaluate_segmentations(
                    ground_truth_masks=[ground_truth_mask],
                    ground_truth_classes=[ann.category.id],
                    predicted_masks=[predicted_mask],
                    predicted_classes=[ann.category.id],
                    predicted_scores=[0.9],
                    image_ids=[ann.image.id],
                    category_ids=[c.id for c in dataset.categories]
                )
                
                assert "precision" in metrics
                assert "recall" in metrics
                assert "ap" in metrics
                break


class TestCommonOperations:
    def test_annotations_for_image(self, dataset):
        img = dataset.images[0]
        
        anns_for_img = dataset.filter_annotations(images=[img])
        direct_anns = img.annotations
        
        assert len(anns_for_img) == len(direct_anns)
        assert set(ann.id for ann in anns_for_img) == set(ann.id for ann in direct_anns)
    
    def test_annotations_for_category(self, dataset):
        cat = dataset.categories[0]
        
        anns_for_cat = dataset.filter_annotations(categories=[cat])
        direct_anns = cat.annotations
        
        assert len(anns_for_cat) == len(direct_anns)
        assert set(ann.id for ann in anns_for_cat) == set(ann.id for ann in direct_anns)
    
    def test_images_for_category(self, dataset):
        cat = dataset.categories[0]
        
        imgs_for_cat = dataset.filter_images(categories=[cat])
        distinct_images = {ann.image for ann in cat.annotations}
        
        assert len(imgs_for_cat) == len(distinct_images)
        assert set(img.id for img in imgs_for_cat) == set(img.id for img in distinct_images)