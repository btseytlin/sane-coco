from sane_coco import COCODataset
from sane_coco.metrics import MeanAveragePrecision


class TestAveragePrecision:
    def test_average_precision(self, sample_data):

        dataset = COCODataset.from_dict(sample_data)

        predictions = []
        for img in dataset.images:
            for ann in img.annotations:
                x, y, w, h = ann.bbox.xywh
                pred = {
                    "image_id": img.id,
                    "category_id": ann.category.id,
                    "bbox": [x + 2, y - 3, w + 1, h - 2],
                    "score": 0.85,
                }
                predictions.append(pred)

            if img.file_name == "000000001.jpg":
                dog_category = next(
                    c for c in dataset.categories.values() if c.name == "dog"
                )
                predictions.append(
                    {
                        "image_id": img.id,
                        "category_id": dog_category.id,
                        "bbox": [400, 300, 50, 40],
                        "score": 0.65,
                    }
                )

        gt_annotations = dataset.get_annotation_dicts()

        pred_annotations = []
        for img in dataset.images:
            img_pred = []
            for pred in predictions:
                if pred["image_id"] == img.id:
                    img_pred.append(
                        {
                            "category": dataset.get_category_by_id(
                                pred["category_id"]
                            ).name,
                            "bbox": pred["bbox"],
                            "score": pred["score"],
                        }
                    )
            pred_annotations.append(img_pred)

        metric = MeanAveragePrecision()
        metric.update(gt_annotations, pred_annotations)
        results = metric.compute()

        assert "map" in results
        assert 0.5 in results["ap"]
        assert 0.75 in results["ap"]
