from sane_coco import COCODataset
from sane_coco.metrics import MeanAveragePrecision


class TestAveragePrecision:
    def test_average_precision(self, sample_data):
        dataset = COCODataset.from_dict(sample_data)

        gt_annotations = dataset.get_annotation_dicts()

        pred_annotations = []
        for img in dataset.images:
            img_predictions = []
            for ann in img.annotations:
                x, y, w, h = ann.bbox.xywh
                img_predictions.append(
                    {
                        "category": ann.category.name,
                        "bbox": [x + 2, y - 3, w + 1, h - 2],
                        "score": 0.8,
                    }
                )

            if img.file_name == "000000001.jpg":
                dog_category = dataset.categories["dog"]
                img_predictions.append(
                    {
                        "category": dog_category.name,
                        "bbox": [x + 10, y - 3, w + 1, h - 2],
                        "score": 0.9,
                    }
                )

            pred_annotations.append(img_predictions)

        metric = MeanAveragePrecision()
        metric.update(gt_annotations, pred_annotations)
        print("GT:", gt_annotations)
        print("PRED:", pred_annotations)
        results = metric.compute()

        assert "map" in results
        assert 0.5 in results["ap"]
        assert 0.75 in results["ap"]

    def test_perfect_predictions(self):
        gt = [[{"category": "person", "bbox": [10, 10, 30, 40]}]]
        pred = [[{"category": "person", "bbox": [10, 10, 30, 40], "score": 1.0}]]

        metric = MeanAveragePrecision()
        metric.update(gt, pred)
        results = metric.compute()

        assert results["map"] == 1.0
        assert results["ap"][0.5] == 1.0

    def test_completely_wrong_predictions(self):
        gt = [[{"category": "person", "bbox": [10, 10, 30, 40]}]]
        pred = [[{"category": "dog", "bbox": [100, 100, 30, 40], "score": 1.0}]]

        metric = MeanAveragePrecision()
        metric.update(gt, pred)
        results = metric.compute()

        assert results["map"] == 0.0
        assert results["ap"][0.5] == 0.0

    def test_false_positives_and_negatives(self):
        gt = [
            [
                {"category": "person", "bbox": [10, 10, 30, 40]},
                {"category": "dog", "bbox": [50, 50, 20, 30]},
            ]
        ]
        pred = [
            [
                {"category": "person", "bbox": [10, 10, 30, 40], "score": 0.9},
                {"category": "person", "bbox": [100, 100, 30, 40], "score": 0.8},
                {"category": "cat", "bbox": [50, 50, 20, 30], "score": 0.7},
            ]
        ]

        metric = MeanAveragePrecision()
        metric.update(gt, pred)
        results = metric.compute()

        assert results["map"] <= 0.5

    def test_multiple_iou_thresholds(self):
        gt = [[{"category": "person", "bbox": [10, 10, 30, 40]}]]
        pred = [[{"category": "person", "bbox": [15, 15, 30, 40], "score": 1.0}]]

        metric = MeanAveragePrecision()
        metric.update(gt, pred)
        results = metric.compute()

        assert results["ap"][0.5] >= results["ap"][0.75]

    def test_empty_predictions(self):
        gt = [[{"category": "person", "bbox": [10, 10, 30, 40]}]]
        pred = [[]]

        metric = MeanAveragePrecision()
        metric.update(gt, pred)
        results = metric.compute()

        assert results["map"] == 0.0

    def test_empty_ground_truth(self):
        gt = [[]]
        pred = [[{"category": "person", "bbox": [10, 10, 30, 40], "score": 1.0}]]

        metric = MeanAveragePrecision()
        metric.update(gt, pred)
        results = metric.compute()

        assert results["map"] == 0.0

    def test_single_tp_single_fp(self):
        gt = [[{"category": "person", "bbox": [10, 10, 30, 40]}]]
        pred = [
            [
                {"category": "person", "bbox": [10, 10, 30, 40], "score": 0.9},
                {"category": "person", "bbox": [100, 100, 30, 40], "score": 0.8},
            ]
        ]

        metric = MeanAveragePrecision()
        metric.update(gt, pred)
        results = metric.compute()

        assert results["ap"][0.5] == 1.0

    def test_precision_recall_curve(self):
        gt = [
            [
                {"category": "person", "bbox": [i * 100, i * 100, 30, 40]}
                for i in range(4)
            ]
        ]
        pred = [
            [
                {"category": "person", "bbox": [0, 0, 30, 40], "score": 0.9},
                {"category": "person", "bbox": [100, 100, 30, 40], "score": 0.8},
                {"category": "person", "bbox": [200, 200, 30, 40], "score": 0.7},
                {"category": "person", "bbox": [500, 500, 30, 40], "score": 0.6},
            ]
        ]

        metric = MeanAveragePrecision()
        metric.update(gt, pred)
        results = metric.compute()

        assert abs(results["ap"][0.5] - 0.75) < 1e-6

    def test_multiple_categories(self):
        gt = [
            [
                {"category": "person", "bbox": [10, 10, 30, 40]},
                {"category": "dog", "bbox": [100, 100, 30, 40]},
            ]
        ]
        pred = [
            [
                {"category": "person", "bbox": [10, 10, 30, 40], "score": 0.9},
                {"category": "dog", "bbox": [100, 100, 30, 40], "score": 0.8},
            ]
        ]

        metric = MeanAveragePrecision()
        metric.update(gt, pred)
        results = metric.compute()

        assert results["map"] == 1.0
        assert results["ap"][0.5] == 1.0

    def test_iou_boundary(self):
        gt = [[{"category": "person", "bbox": [10, 10, 30, 40]}]]
        pred = [[{"category": "person", "bbox": [35, 35, 30, 40], "score": 1.0}]]

        metric = MeanAveragePrecision()
        metric.update(gt, pred)
        results = metric.compute()

        assert results["ap"][0.5] == 0.0

    def test_multiple_images_map(self):
        gt = [
            [{"category": "person", "bbox": [10, 10, 30, 40]}],
            [{"category": "person", "bbox": [50, 50, 30, 40]}],
        ]
        pred = [
            [{"category": "person", "bbox": [12, 12, 30, 40], "score": 0.9}],
            [{"category": "person", "bbox": [51, 51, 30, 40], "score": 0.8}],
        ]

        metric = MeanAveragePrecision()
        metric.update(gt, pred)
        results = metric.compute()

        assert results["map"] > 0.6

    def test_exact_iou_thresholds(self):
        gt = [[{"category": "person", "bbox": [10, 10, 30, 40]}]]
        pred = [[{"category": "person", "bbox": [15, 15, 30, 40], "score": 1.0}]]

        metric = MeanAveragePrecision()
        metric.update(gt, pred)
        results = metric.compute()

        assert results["ap"][0.5] == 1.0
        assert results["ap"][0.75] == 0.0

    def test_confidence_thresholds(self):
        gt = [[{"category": "person", "bbox": [10, 10, 30, 40]}]]
        pred = [
            [
                {"category": "person", "bbox": [10, 10, 30, 40], "score": 0.3},
                {"category": "person", "bbox": [10, 10, 30, 40], "score": 0.7},
                {"category": "person", "bbox": [10, 10, 30, 40], "score": 0.9},
            ]
        ]

        metric = MeanAveragePrecision()
        metric.update(gt, pred)
        results = metric.compute()

        assert results["map"] == 1.0

    def test_scale_invariance(self):
        gt = [
            [
                {"category": "person", "bbox": [10, 10, 300, 400]},
                {"category": "person", "bbox": [50, 50, 30, 40]},
            ]
        ]
        pred = [
            [
                {"category": "person", "bbox": [12, 12, 300, 400], "score": 0.9},
                {"category": "person", "bbox": [51, 51, 30, 40], "score": 0.8},
            ]
        ]

        metric = MeanAveragePrecision()
        metric.update(gt, pred)
        results = metric.compute()

        assert results["map"] >= 0.9

    def test_overlapping_predictions(self):
        gt = [[{"category": "person", "bbox": [10, 10, 30, 40]}]]
        pred = [
            [
                {"category": "person", "bbox": [11, 11, 30, 40], "score": 0.9},
                {"category": "person", "bbox": [12, 12, 30, 40], "score": 0.8},
                {"category": "person", "bbox": [13, 13, 30, 40], "score": 0.7},
            ]
        ]

        metric = MeanAveragePrecision()
        metric.update(gt, pred)
        results = metric.compute()

        assert results["ap"][0.5] == 1.0

    def test_different_aspect_ratios(self):
        gt = [
            [
                {"category": "person", "bbox": [10, 10, 100, 20]},
                {"category": "person", "bbox": [50, 50, 20, 100]},
            ]
        ]
        pred = [
            [
                {"category": "person", "bbox": [11, 11, 100, 20], "score": 0.9},
                {"category": "person", "bbox": [51, 51, 20, 100], "score": 0.8},
            ]
        ]

        metric = MeanAveragePrecision()
        metric.update(gt, pred)
        results = metric.compute()

        assert results["map"] >= 0.8

    def test_custom_iou_threshold(self):
        gt = [[{"category": "person", "bbox": [10, 10, 30, 40]}]]
        pred = [[{"category": "person", "bbox": [15, 15, 30, 40], "score": 1.0}]]

        metric = MeanAveragePrecision(iou_thresholds=[0.35])
        metric.update(gt, pred)
        results = metric.compute()

        assert results["ap"][0.35] == 1.0
        assert 0.5 not in results["ap"]
