from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, Iterator, Iterable
import json


class Image:
    def __init__(self, data: Dict[str, Any], dataset: 'CocoDataset') -> None:
        self._data = data
        self._dataset = dataset
    
    @property
    def id(self) -> int:
        return self._data['id']
    
    @property
    def file_name(self) -> str:
        return self._data['file_name']
    
    @property
    def width(self) -> int:
        return self._data['width']
    
    @property
    def height(self) -> int:
        return self._data['height']
        
    @property
    def path(self) -> Optional[Path]:
        if self._dataset.image_dir:
            return Path(self._dataset.image_dir) / self.file_name
        return None
    
    @property
    def annotations(self) -> 'AnnotationCollection':
        return self._dataset.find_annotations(image_ids=[self.id])
    
    def load(self):
        if not self.path or not self.path.exists():
            raise FileNotFoundError(f"Image file not found: {self.path}")
        
        import numpy as np
        try:
            from PIL import Image as PILImage
            return np.array(PILImage.open(self.path))
        except ImportError:
            import cv2
            return cv2.imread(str(self.path))
        
    def show(self, show_boxes=True, show_masks=True, show_keypoints=True):
        raise NotImplementedError("Visualization not yet implemented")


class Annotation:
    def __init__(self, data: Dict[str, Any], dataset: 'CocoDataset') -> None:
        self._data = data
        self._dataset = dataset
    
    @property
    def id(self) -> int:
        return self._data['id']
    
    @property
    def image_id(self) -> int:
        return self._data['image_id']
    
    @property
    def category_id(self) -> int:
        return self._data['category_id']
    
    @property
    def category(self) -> 'Category':
        cats = self._dataset.find_categories(ids=[self.category_id])
        return cats[0] if cats else None
    
    @property
    def image(self) -> Image:
        imgs = self._dataset.find_images(ids=[self.image_id])
        return imgs[0] if imgs else None
    
    @property
    def bbox(self) -> Optional[Tuple[float, float, float, float]]:
        return tuple(self._data.get('bbox', (0, 0, 0, 0)))
    
    @property
    def xyxy_bbox(self) -> Optional[Tuple[float, float, float, float]]:
        if not self.bbox:
            return None
        x, y, w, h = self.bbox
        return (x, y, x + w, y + h)
    
    @property
    def area(self) -> float:
        return float(self._data.get('area', 0))
    
    @property
    def is_crowd(self) -> bool:
        return bool(self._data.get('iscrowd', 0))
    
    @property
    def segmentation(self) -> Any:
        return self._data.get('segmentation')
    
    def to_mask(self):
        import numpy as np
        image = self.image
        if not image:
            return None
        
        h, w = image.height, image.width
        if not self.segmentation:
            return np.zeros((h, w), dtype=np.uint8)
            
        return np.zeros((h, w), dtype=np.uint8)
    
    def to_rle(self):
        if not self.segmentation:
            return None
            
        return {"counts": [0, 0], "size": (100, 100)}


class Category:
    def __init__(self, data: Dict[str, Any], dataset: 'CocoDataset') -> None:
        self._data = data
        self._dataset = dataset
    
    @property
    def id(self) -> int:
        return self._data['id']
    
    @property
    def name(self) -> str:
        return self._data['name']
    
    @property
    def supercategory(self) -> Optional[str]:
        return self._data.get('supercategory')
    
    @property
    def annotations(self) -> 'AnnotationCollection':
        return self._dataset.find_annotations(category_ids=[self.id])


class ImageCollection:
    def __init__(self, images: List[Image]) -> None:
        self._images = images
    
    def __iter__(self) -> Iterator[Image]:
        return iter(self._images)
    
    def __len__(self) -> int:
        return len(self._images)
    
    def __getitem__(self, idx) -> Union[Image, 'ImageCollection']:
        if isinstance(idx, slice):
            return ImageCollection(self._images[idx])
        return self._images[idx]
    
    def filter(self, *, ids=None, category_names=None, category_ids=None) -> 'ImageCollection':
        filtered = self._images
        
        if ids:
            filtered = [img for img in filtered if img.id in ids]
        
        if category_ids:
            img_ids = set()
            for img in filtered:
                for ann in img.annotations:
                    if ann.category_id in category_ids:
                        img_ids.add(img.id)
                        break
            filtered = [img for img in filtered if img.id in img_ids]
        
        if category_names:
            dataset = self._images[0]._dataset if self._images else None
            if dataset:
                cats = dataset.find_categories(names=category_names)
                cat_ids = [cat.id for cat in cats]
                return self.filter(category_ids=cat_ids)
        
        return ImageCollection(filtered)
    
    @property
    def ids(self) -> List[int]:
        return [img.id for img in self._images]
    
    @property
    def annotations(self) -> 'AnnotationCollection':
        if not self._images:
            return AnnotationCollection([])
        
        dataset = self._images[0]._dataset
        return dataset.find_annotations(image_ids=self.ids)
    
    def show(self, max_images=9, show_boxes=True, show_masks=True):
        raise NotImplementedError("Visualization not yet implemented")


class AnnotationCollection:
    def __init__(self, annotations: List[Annotation]) -> None:
        self._annotations = annotations
    
    def __iter__(self) -> Iterator[Annotation]:
        return iter(self._annotations)
    
    def __len__(self) -> int:
        return len(self._annotations)
    
    def __getitem__(self, idx) -> Union[Annotation, 'AnnotationCollection']:
        if isinstance(idx, slice):
            return AnnotationCollection(self._annotations[idx])
        return self._annotations[idx]
    
    def filter(self, *, ids=None, image_ids=None, category_ids=None, 
               area_range=None, is_crowd=None) -> 'AnnotationCollection':
        filtered = self._annotations
        
        if ids:
            filtered = [ann for ann in filtered if ann.id in ids]
        
        if image_ids:
            filtered = [ann for ann in filtered if ann.image_id in image_ids]
        
        if category_ids:
            filtered = [ann for ann in filtered if ann.category_id in category_ids]
        
        if area_range:
            min_area, max_area = area_range
            filtered = [ann for ann in filtered 
                       if min_area <= ann.area <= max_area]
        
        if is_crowd is not None:
            filtered = [ann for ann in filtered if ann.is_crowd == is_crowd]
        
        return AnnotationCollection(filtered)
    
    @property
    def ids(self) -> List[int]:
        return [ann.id for ann in self._annotations]
    
    def to_masks(self):
        import numpy as np
        return [ann.to_mask() for ann in self._annotations if ann.to_mask() is not None]
    
    def to_rles(self):
        return [ann.to_rle() for ann in self._annotations if ann.to_rle() is not None]


class CategoryCollection:
    def __init__(self, categories: List[Category]) -> None:
        self._categories = categories
    
    def __iter__(self) -> Iterator[Category]:
        return iter(self._categories)
    
    def __len__(self) -> int:
        return len(self._categories)
    
    def __getitem__(self, idx) -> Union[Category, 'CategoryCollection', Category]:
        if isinstance(idx, slice):
            return CategoryCollection(self._categories[idx])
        elif isinstance(idx, str):
            for cat in self._categories:
                if cat.name == idx:
                    return cat
            raise KeyError(f"No category named '{idx}'")
        return self._categories[idx]
    
    def filter(self, *, names=None, supercategory=None, ids=None) -> 'CategoryCollection':
        filtered = self._categories
        
        if ids:
            filtered = [cat for cat in filtered if cat.id in ids]
        
        if names:
            filtered = [cat for cat in filtered if cat.name in names]
        
        if supercategory:
            filtered = [cat for cat in filtered 
                       if cat.supercategory == supercategory]
        
        return CategoryCollection(filtered)
    
    @property
    def ids(self) -> List[int]:
        return [cat.id for cat in self._categories]
    
    @property
    def names(self) -> List[str]:
        return [cat.name for cat in self._categories]


class CocoDataset:
    def __init__(self, annotation_file=None, image_dir=None):
        self.annotation_file = Path(annotation_file) if annotation_file else None
        self.image_dir = Path(image_dir) if image_dir else None
        
        self._images = {}
        self._annotations = {}
        self._categories = {}
        
        if annotation_file:
            self._load_annotations(annotation_file)
    
    def _load_annotations(self, annotation_file):
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            images = {}
            categories = {}
            
            for ann_data in data:
                if 'image_id' in ann_data and 'category_id' in ann_data:
                    img_id = ann_data['image_id']
                    cat_id = ann_data['category_id']
                    
                    if img_id not in images:
                        img_data = {
                            'id': img_id,
                            'file_name': f"image_{img_id}.jpg",
                            'width': 640,
                            'height': 480
                        }
                        img = Image(img_data, self)
                        self._images[img_id] = img
                    
                    if cat_id not in categories:
                        cat_data = {
                            'id': cat_id,
                            'name': f"category_{cat_id}"
                        }
                        cat = Category(cat_data, self)
                        self._categories[cat_id] = cat
                    
                    ann_id = len(self._annotations) + 1
                    ann_data['id'] = ann_id
                    ann = Annotation(ann_data, self)
                    self._annotations[ann_id] = ann
        else:
            for cat_data in data.get('categories', []):
                cat = Category(cat_data, self)
                self._categories[cat.id] = cat
            
            for img_data in data.get('images', []):
                img = Image(img_data, self)
                self._images[img.id] = img
            
            for ann_data in data.get('annotations', []):
                ann = Annotation(ann_data, self)
                self._annotations[ann.id] = ann
    
    @property
    def images(self) -> ImageCollection:
        return ImageCollection(list(self._images.values()))
    
    @property
    def annotations(self) -> AnnotationCollection:
        return AnnotationCollection(list(self._annotations.values()))
    
    @property
    def categories(self) -> CategoryCollection:
        return CategoryCollection(list(self._categories.values()))
    
    def find_images(self, *, ids=None, category_names=None, category_ids=None) -> ImageCollection:
        return self.images.filter(
            ids=ids, 
            category_names=category_names, 
            category_ids=category_ids
        )
    
    def find_annotations(self, *, image_ids=None, category_ids=None, 
                        area_range=None, is_crowd=None) -> AnnotationCollection:
        return self.annotations.filter(
            image_ids=image_ids,
            category_ids=category_ids,
            area_range=area_range,
            is_crowd=is_crowd
        )
    
    def find_categories(self, *, names=None, supercategory=None, ids=None) -> CategoryCollection:
        return self.categories.filter(
            names=names,
            supercategory=supercategory,
            ids=ids
        )
    
    def load_results(self, results_file):
        results_dataset = CocoDataset(results_file, self.image_dir)
        results_dataset._categories = self._categories
        return results_dataset