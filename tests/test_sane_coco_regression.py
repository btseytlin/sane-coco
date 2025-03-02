import os
import numpy as np
import pytest
from pathlib import Path
import json
from typing import List, Dict, Any

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

from sane_coco.dataset import CocoDataset, Image, Annotation, Category
from sane_coco.eval import CocoEvaluator


@pytest.fixture
def coco_data():
    with open("tests/data/sample_coco.json", "r") as f:
        return json.load(f)


@pytest.fixture
def results_data():
    with open("tests/data/sample_results.json", "r") as f:
        return json.load(f)


@pytest.fixture
def coco_original(coco_data):
    tmp_path = Path("tests/data/tmp_coco.json")
    with open(tmp_path, "w") as f:
        json.dump(coco_data, f)
    coco = COCO(str(tmp_path))
    tmp_path.unlink()
    return coco


@pytest.fixture
def coco_sane(coco_data):
    return CocoDataset.from_dict(coco_data)


class TestDatasetInterface:
    def test_dataset_loading(self, coco_original, coco_sane):
        assert len(coco_original.imgs) == len(coco_sane.images)
        assert len(coco_original.anns) == len(coco_sane.annotations)
        assert len(coco_original.cats) == len(coco_sane.categories)
        
        sample_img_id = list(coco_original.imgs.keys())[0]
        assert sample_img_id in [img.id for img in coco_sane.images]
        
        orig_img = coco_original.imgs[sample_img_id]
        sane_img = next(img for img in coco_sane.images if img.id == sample_img_id)
        assert orig_img['file_name'] == sane_img.file_name
        assert orig_img['width'] == sane_img.width
        assert orig_img['height'] == sane_img.height
    
    def test_image_filtering(self, coco_original, coco_sane):
        cat_id = list(coco_original.cats.keys())[0]
        cat_name = coco_original.cats[cat_id]['name']
        
        orig_img_ids = coco_original.getImgIds(catIds=[cat_id])
        sane_imgs = coco_sane.filter_images(category_ids=[cat_id])
        assert set(orig_img_ids) == set(img.id for img in sane_imgs)
        
        sane_imgs_by_name = coco_sane.filter_images(category_names=[cat_name])
        assert set(img.id for img in sane_imgs) == set(img.id for img in sane_imgs_by_name)
        
        test_img_ids = orig_img_ids[:1]
        orig_filtered = coco_original.loadImgs(test_img_ids)
        sane_filtered = coco_sane.filter_images(ids=test_img_ids)
        assert len(orig_filtered) == len(sane_filtered)
        assert (set(img['id'] for img in orig_filtered) == 
                set(img.id for img in sane_filtered))

    def test_annotation_filtering(self, coco_original, coco_sane):
        img_id = list(coco_original.imgs.keys())[0]
        cat_id = list(coco_original.cats.keys())[0]
        
        orig_ann_ids = coco_original.getAnnIds(imgIds=[img_id])
        sane_anns = coco_sane.filter_annotations(image_ids=[img_id])
        assert len(orig_ann_ids) == len(sane_anns)
        
        orig_cat_ann_ids = coco_original.getAnnIds(catIds=[cat_id])
        sane_cat_anns = coco_sane.filter_annotations(category_ids=[cat_id])
        assert len(orig_cat_ann_ids) == len(sane_cat_anns)
        
        area_range = [100, 70000]
        orig_area_ann_ids = coco_original.getAnnIds(areaRng=area_range)
        sane_area_anns = coco_sane.filter_annotations(area_range=area_range)
        assert len(orig_area_ann_ids) == len(sane_area_anns)
        
        orig_combined = coco_original.getAnnIds(
            imgIds=[img_id], catIds=[cat_id], areaRng=area_range
        )
        sane_combined = coco_sane.filter_annotations(
            image_ids=[img_id], category_ids=[cat_id], area_range=area_range
        )
        assert len(orig_combined) == len(sane_combined)

    def test_category_operations(self, coco_original, coco_sane):
        orig_cats = coco_original.loadCats(coco_original.getCatIds())
        assert len(orig_cats) == len(coco_sane.categories)
        
        if 'supercategory' in orig_cats[0]:
            supercategory = orig_cats[0]['supercategory']
            orig_super_cats = [cat for cat in orig_cats 
                              if cat.get('supercategory') == supercategory]
            sane_super_cats = coco_sane.filter_categories(supercategory=supercategory)
            assert len(orig_super_cats) == len(sane_super_cats)
        
        cat_id = orig_cats[0]['id']
        cat_name = orig_cats[0]['name']
        
        orig_cat_anns = coco_original.loadAnns(
            coco_original.getAnnIds(catIds=[cat_id])
        )
        
        sane_category = coco_sane.filter_categories(names=[cat_name])[0]
        assert cat_name == sane_category.name
        
        sane_cat_anns = coco_sane.filter_annotations(category_ids=[cat_id])
        assert len(orig_cat_anns) == len(sane_cat_anns)


@pytest.fixture
def annotation_with_segmentation(coco_data):
    for ann in coco_data['annotations']:
        if 'segmentation' in ann:
            return {'ann_id': ann['id'], 'img_id': ann['image_id']}
    pytest.skip("No annotation with segmentation found")


class TestVisualizationAndMasks:
    def test_mask_conversion(self, coco_original, coco_sane, annotation_with_segmentation):
        ann_id = annotation_with_segmentation['ann_id']
        
        orig_ann = coco_original.loadAnns([ann_id])[0]
        sane_ann = next(ann for ann in coco_sane.annotations if ann.id == ann_id)
        
        if isinstance(orig_ann['segmentation'], list):
            mask = sane_ann.generate_mask()
            assert mask.shape == (sane_ann.image.height, sane_ann.image.width)
            
            rle = sane_ann.generate_rle()
            assert "counts" in rle
            assert "size" in rle
    
    def test_collection_operations(self, coco_original, coco_sane, annotation_with_segmentation):
        img_id = annotation_with_segmentation['img_id']
        
        orig_anns = coco_original.loadAnns(
            coco_original.getAnnIds(imgIds=[img_id])
        )
        
        sane_img = next(img for img in coco_sane.images if img.id == img_id)
        sane_anns = coco_sane.filter_annotations(image_ids=[img_id])
        
        assert len(orig_anns) == len(sane_anns)
        
        if len(sane_anns) >= 2:
            half_count = len(sane_anns) // 2
            half_ids = [ann.id for ann in sane_anns[:half_count]]
            
            filtered_anns = coco_sane.filter_annotations(ids=half_ids)
            assert len(filtered_anns) == half_count


class TestEvaluation:
    def test_bbox_evaluation(self, coco_data, results_data):
        coco_dataset = CocoDataset.from_dict(coco_data)
        predictions = CocoDataset.from_predictions(results_data, coco_dataset.categories)
        
        evaluator = CocoEvaluator(coco_dataset, predictions, iou_type='bbox')
        metrics = evaluator.evaluate()
        
        assert "precision" in metrics
        assert "recall" in metrics
        assert "ap" in metrics
    
    def test_segmentation_evaluation(self, coco_data, results_data):
        coco_dataset = CocoDataset.from_dict(coco_data)
        predictions = CocoDataset.from_predictions(results_data, coco_dataset.categories)
        
        evaluator = CocoEvaluator(coco_dataset, predictions, iou_type='segm')
        metrics = evaluator.evaluate()
        
        assert "precision" in metrics
        assert "recall" in metrics
        assert "ap" in metrics


class TestDataClassOperations:
    def test_dataset_creation(self, coco_data):
        images = [
            Image(id=img["id"], width=img["width"], height=img["height"], 
                  file_name=img["file_name"])
            for img in coco_data["images"]
        ]
        
        categories = [
            Category(id=cat["id"], name=cat["name"], 
                    supercategory=cat.get("supercategory"))
            for cat in coco_data["categories"]
        ]
        
        annotations = []
        for ann in coco_data["annotations"]:
            image = next(img for img in images if img.id == ann["image_id"])
            category = next(cat for cat in categories if cat.id == ann["category_id"])
            
            annotations.append(
                Annotation(
                    id=ann["id"],
                    image=image,
                    category=category,
                    bbox=ann["bbox"],
                    area=ann["area"],
                    segmentation=ann.get("segmentation"),
                    is_crowd=bool(ann.get("iscrowd", 0))
                )
            )
        
        dataset = CocoDataset(images=images, annotations=annotations, categories=categories)
        
        assert len(dataset.images) == len(coco_data["images"])
        assert len(dataset.annotations) == len(coco_data["annotations"])
        assert len(dataset.categories) == len(coco_data["categories"])
    
    def test_image_operations(self, coco_sane):
        img = coco_sane.images[0]
        
        image_anns = coco_sane.filter_annotations(image_ids=[img.id])
        assert len(image_anns) > 0
        
        assert len(img.annotations) > 0
        
        assert set(ann.id for ann in img.annotations) == set(ann.id for ann in image_anns)
    
    def test_category_operations(self, coco_sane):
        cat = coco_sane.categories[0]
        
        cat_anns = coco_sane.filter_annotations(category_ids=[cat.id])
        assert len(cat_anns) > 0
        
        assert len(cat.annotations) > 0
        
        assert set(ann.id for ann in cat.annotations) == set(ann.id for ann in cat_anns)
    
    def test_annotation_operations(self, coco_sane):
        ann = coco_sane.annotations[0]
        
        assert ann.image is not None
        assert ann.category is not None
        
        assert ann.image.id == ann.image_id
        assert ann.category.id == ann.category_id
        
        x, y, w, h = ann.bbox
        xyxy = ann.xyxy_bbox
        assert xyxy == (x, y, x + w, y + h)