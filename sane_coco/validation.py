from typing import List, Set


def validate_required_fields(data: dict, required: List[str], entity: str) -> None:
    missing = [f for f in required if f not in data]
    if missing:
        raise ValueError(f"{entity} missing required fields: {', '.join(missing)}")


def validate_bbox(bbox: List[float]) -> None:
    if len(bbox) != 4:
        raise ValueError("invalid bbox format")
    if bbox[2] <= 0 or bbox[3] <= 0:
        raise ValueError("invalid bbox dimensions")
