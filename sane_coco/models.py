from dataclasses import dataclass
from typing import List, Tuple, Union, Dict, Any, Optional
import numpy as np
from matplotlib.path import Path
from skimage import measure, transform


@dataclass
class BBox:
    x: float
    y: float
    width: float
    height: float
    score: float | None = None

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def corners(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.width, self.height)

    def to_dict(self) -> list:
        return list(self.xywh)


class Mask:
    def __init__(self, array: np.ndarray):
        self.array = np.asarray(array, dtype=bool)

    @property
    def area(self) -> int:
        return int(np.sum(self.array))

    def to_rle(self) -> "RLE":
        mask = self.array.flatten()
        counts, last, count = [], False, 0
        for bit in mask:
            if bit != last:
                counts.append(count)
                last, count = bit, 1
            else:
                count += 1
        counts.append(count)
        return RLE(counts if len(counts) % 2 == 1 else [0] + counts, self.array.shape)

    def to_polygon(self) -> "Polygon":
        contours = measure.find_contours(self.array, 0.5)
        return (
            Polygon([])
            if not contours
            else Polygon(np.fliplr(contours[0]).flatten().tolist())
        )


class RLE:
    def __init__(self, counts: List[int], size: Tuple[int, int], area: int = None):
        self.counts, self.size = counts, size
        self.input_area = area

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RLE":
        return cls(counts=data["counts"], size=data["size"], area=data.get("area"))

    def to_dict(self) -> Dict[str, Any]:
        return {"counts": self.counts, "size": self.size}

    @property
    def area(self) -> int:
        if self.input_area is not None:
            return self.input_area
        return sum(self.counts[1::2] if self.counts[0] == 0 else self.counts[::2])

    def to_mask(self, size: Tuple[int, int] = None) -> Mask:
        h, w = size or self.size
        mask = np.zeros(h * w, dtype=bool)
        idx, val = 0, False
        for count in self.counts:
            end_idx = min(idx + count, len(mask))
            if val:
                mask[idx:end_idx] = True
            idx, val = end_idx, not val
        mask = mask.reshape(self.size)
        if size and size != self.size:
            return Mask(
                transform.resize(mask.astype(float), size, order=0, preserve_range=True)
                > 0.5
            )
        return Mask(mask)

    def to_polygon(self) -> "Polygon":
        return self.to_mask().to_polygon()


class Polygon:
    def __init__(self, points: List[float]):
        self.points = points

    @property
    def area(self) -> float:
        if len(self.points) < 6:
            return 0.0

        x, y = self.points[::2], self.points[1::2]
        return 0.5 * abs(
            sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(-1, len(x) - 1))
        )

    def to_dict(self) -> list:
        return self.points

    def to_mask(self, size: Tuple[int, int] = None) -> Mask:
        if not self.points:
            return Mask(np.zeros(size or (1, 1), dtype=bool))
        if not size:
            max_x, max_y = max(self.points[::2]), max(self.points[1::2])
            size = (int(max_y) + 1, int(max_x) + 1)
        y, x = np.mgrid[: size[0], : size[1]]
        return Mask(
            Path(np.array(self.points).reshape(-1, 2))
            .contains_points(np.vstack((x.flatten(), y.flatten())).T)
            .reshape(size)
        )

    def to_rle(self, size: Tuple[int, int]) -> RLE:
        return self.to_mask(size).to_rle()


@dataclass
class Category:
    id: int
    name: str
    supercategory: str

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "supercategory": self.supercategory}


@dataclass
class Image:
    id: int
    file_name: str
    width: int
    height: int
    annotations: List["Annotation"]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "file_name": self.file_name,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class Annotation:
    id: int
    bbox: BBox
    category: Category
    image: Image
    segmentation: Optional[Union[Polygon, RLE]] = None
    iscrowd: bool = False

    @property
    def area(self) -> float:
        if self.segmentation:
            return self.segmentation.area
        return self.bbox.area

    def compute_iou(self, other: "Annotation") -> float:
        if not (self.iscrowd or other.iscrowd):
            x1, y1, w1, h1 = self.bbox.xywh
            x2, y2, w2, h2 = other.bbox.xywh
            x_left, y_top = max(x1, x2), max(y1, y2)
            x_right, y_bottom = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
            if x_right < x_left or y_bottom < y_top:
                return 0.0
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            return intersection_area / (
                self.bbox.area + other.bbox.area - intersection_area
            )

        if not (self.segmentation and other.segmentation):
            return 0.0
        size = (self.image.height, self.image.width)
        mask1, mask2 = self.segmentation.to_mask(size), other.segmentation.to_mask(size)
        intersection = np.logical_and(mask1.array, mask2.array).sum()
        if self.iscrowd and other.iscrowd:
            return float(intersection) / max(
                np.logical_or(mask1.array, mask2.array).sum(), 1
            )
        return float(intersection) / max(mask2.area if self.iscrowd else mask1.area, 1)

    def to_dict(self) -> dict:
        result = {
            "id": self.id,
            "image_id": self.image.id,
            "category_id": self.category.id,
            "bbox": self.bbox.to_dict(),
            "area": self.area,
            "iscrowd": 1 if self.iscrowd else 0,
        }
        if self.segmentation:
            result["segmentation"] = (
                [self.segmentation.to_dict()]
                if isinstance(self.segmentation, Polygon)
                else self.segmentation.to_dict()
            )
        return result
