from dataclasses import dataclass
from typing import List


@dataclass
class BBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def corners(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass
class Category:
    id: int
    name: str
    supercategory: str


@dataclass
class Image:
    id: int
    file_name: str
    width: int
    height: int
    annotations: List["Annotation"]


@dataclass
class Annotation:
    id: int
    bbox: BBox
    category: Category
    image: Image
