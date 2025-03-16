from dataclasses import dataclass
from typing import List, Tuple, Union, Dict, Any, Optional
import numpy as np


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
        if not isinstance(array, np.ndarray):
            array = np.array(array, dtype=bool)
        elif array.dtype != bool:
            array = array.astype(bool)
        self.array = array

    @property
    def area(self) -> int:
        return int(np.sum(self.array))

    def to_rle(self) -> "RLE":
        mask = self.array.flatten()
        counts = []
        last = False
        count = 0

        for bit in mask:
            if bit != last:
                counts.append(count)
                last = bit
                count = 1
            else:
                count += 1
        counts.append(count)

        if len(counts) % 2 == 0:
            counts = [0] + counts

        return RLE(counts=counts, size=self.array.shape)

    def to_polygon(self) -> "Polygon":
        from skimage import measure

        contours = measure.find_contours(self.array, 0.5)
        if not contours:
            return Polygon([])

        contour = contours[0]
        contour = np.fliplr(contour)
        points = contour.flatten().tolist()
        return Polygon(points)


class RLE:
    def __init__(self, counts: List[int], size: Tuple[int, int]):
        self.counts = counts
        self.size = size

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RLE":
        return cls(counts=data["counts"], size=data["size"])

    def to_dict(self) -> Dict[str, Any]:
        return {"counts": self.counts, "size": self.size}

    @property
    def area(self) -> int:
        counts = self.counts
        if len(counts) % 2 == 1:
            counts = counts[1:]
        return sum(counts[1::2])

    def to_mask(self, size: Tuple[int, int] = None) -> Mask:
        if size is None:
            size = self.size

        h, w = size
        # Create a flat mask array
        mask = np.zeros(h * w, dtype=bool)

        # COCO RLE format: counts represent runs of 0s and 1s in row-major order
        counts = self.counts
        val = False  # Start with 0s (False)
        idx = 0

        for count in counts:
            end_idx = min(idx + count, len(mask))
            if val:  # If current run is 1s
                mask[idx:end_idx] = True
            idx = end_idx
            val = not val  # Toggle between 0s and 1s

        # Reshape to the original size
        mask = mask.reshape(self.size)

        # Resize if needed
        if size != self.size:
            from skimage.transform import resize

            resized = resize(mask.astype(float), size, order=0, preserve_range=True)
            return Mask(resized > 0.5)

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

        # Shoelace formula for polygon area
        x = self.points[::2]
        y = self.points[1::2]
        return 0.5 * abs(
            sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(-1, len(x) - 1))
        )

    def to_dict(self) -> list:
        return self.points

    def to_mask(self, size: Tuple[int, int] = None) -> Mask:
        if size is None:
            # Default size if not provided
            max_x = max(self.points[::2]) if self.points else 0
            max_y = max(self.points[1::2]) if self.points else 0
            size = (int(max_y) + 1, int(max_x) + 1)

        mask = np.zeros(size, dtype=bool)
        if not self.points:
            return Mask(mask)

        # Convert points to polygon vertices
        vertices = np.array(self.points).reshape(-1, 2)

        # Create grid of points
        y, x = np.mgrid[: size[0], : size[1]]
        points = np.vstack((x.flatten(), y.flatten())).T

        # Use matplotlib's path to determine points inside polygon
        from matplotlib.path import Path

        path = Path(vertices)
        grid = path.contains_points(points)
        mask = grid.reshape(size)

        return Mask(mask)

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
    area: Optional[float] = None
    iscrowd: bool = False

    def compute_iou(self, other: "Annotation") -> float:
        if self.iscrowd or other.iscrowd:
            # For crowd annotations, IoU is the area of intersection / area of non-crowd
            if not self.segmentation or not other.segmentation:
                return 0.0

            # Get image dimensions for consistent mask sizes
            img_width = self.image.width
            img_height = self.image.height
            size = (img_height, img_width)

            if self.iscrowd and other.iscrowd:
                # Both are crowd, use standard IoU with masks
                crowd_mask1 = self.segmentation.to_mask(size)
                crowd_mask2 = other.segmentation.to_mask(size)

                intersection = np.logical_and(
                    crowd_mask1.array, crowd_mask2.array
                ).sum()
                union = np.logical_or(crowd_mask1.array, crowd_mask2.array).sum()

                return float(intersection) / max(union, 1)
            elif self.iscrowd:
                # Self is crowd, other is not
                # Create masks with consistent size
                crowd_mask = self.segmentation.to_mask(size)
                other_mask = other.segmentation.to_mask(size)

                intersection = np.logical_and(crowd_mask.array, other_mask.array).sum()
                return float(intersection) / max(other_mask.area, 1)
            else:
                # Other is crowd, self is not
                # Create masks with consistent size
                crowd_mask = other.segmentation.to_mask(size)
                self_mask = self.segmentation.to_mask(size)

                intersection = np.logical_and(crowd_mask.array, self_mask.array).sum()
                return float(intersection) / max(self_mask.area, 1)

        # Standard IoU calculation for non-crowd annotations
        x1, y1, w1, h1 = self.bbox.xywh
        x2, y2, w2, h2 = other.bbox.xywh

        # Calculate intersection area
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union area
        union_area = self.bbox.area + other.bbox.area - intersection_area

        return intersection_area / union_area

    def to_dict(self) -> dict:
        result = {
            "id": self.id,
            "image_id": self.image.id,
            "category_id": self.category.id,
            "bbox": self.bbox.to_dict(),
            "area": self.area if self.area is not None else self.bbox.area,
            "iscrowd": 1 if self.iscrowd else 0,
        }

        if self.segmentation:
            if isinstance(self.segmentation, Polygon):
                result["segmentation"] = [self.segmentation.to_dict()]
            elif isinstance(self.segmentation, RLE):
                result["segmentation"] = self.segmentation.to_dict()

        return result
