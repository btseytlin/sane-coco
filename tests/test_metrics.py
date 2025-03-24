from sane_coco import COCODataset
from sane_coco.metrics import (
    MeanAveragePrecision,
    average_precision,
    compute_ap_ar_at_iou,
    compute_precision_recall,
)
from sane_coco.util import calculate_iou_batch
from sane_coco.numba import calculate_iou_batch_numba
import numpy as np
import pytest


class TestIOU:
    @pytest.mark.parametrize("iou_fn", [calculate_iou_batch, calculate_iou_batch_numba])
    def test_perfect_overlap(self, iou_fn):
        boxes1 = np.array([[10, 10, 30, 40]], dtype=np.float32)
        boxes2 = np.array([[10, 10, 30, 40]], dtype=np.float32)
        iou = iou_fn(boxes1, boxes2)
        assert iou.shape == (1, 1)
        assert iou[0, 0] == 1.0

    @pytest.mark.parametrize("iou_fn", [calculate_iou_batch, calculate_iou_batch_numba])
    def test_no_overlap(self, iou_fn):
        boxes1 = np.array([[0, 0, 10, 10]], dtype=np.float32)
        boxes2 = np.array([[20, 20, 10, 10]], dtype=np.float32)
        iou = iou_fn(boxes1, boxes2)
        assert iou.shape == (1, 1)
        assert iou[0, 0] == 0.0

    @pytest.mark.parametrize("iou_fn", [calculate_iou_batch, calculate_iou_batch_numba])
    def test_partial_overlap(self, iou_fn):
        boxes1 = np.array([[10, 10, 20, 20]], dtype=np.float32)
        boxes2 = np.array([[15, 15, 20, 20]], dtype=np.float32)
        iou = iou_fn(boxes1, boxes2)
        assert iou.shape == (1, 1)
        assert np.allclose(iou, 0.39130434), iou

        boxes1 = boxes1 * 100
        boxes2 = boxes2 * 100
        iou = iou_fn(boxes1, boxes2)
        assert np.allclose(iou, 0.39130434), iou

    @pytest.mark.parametrize("iou_fn", [calculate_iou_batch, calculate_iou_batch_numba])
    def test_batch_comparison(self, iou_fn):
        boxes1 = np.array([[10, 10, 30, 40]], dtype=np.float32)
        boxes2 = np.array([[10, 10, 30, 40], [45, 55, 65, 75]], dtype=np.float32)
        iou = iou_fn(boxes1, boxes2)
        assert iou.shape == (1, 2)
        assert np.allclose(iou, np.array([[1.0, 0.0]])), iou

        boxes1 = np.array([[10, 10, 30, 40], [45, 55, 65, 75]], dtype=np.float32)
        iou = iou_fn(boxes1, boxes2)
        assert iou.shape == (2, 2)
        assert np.allclose(iou, np.array([[1.0, 0.0], [0.0, 1.0]]))

        boxes1 = np.array([[10, 10, 30, 40], [10, 10, 30, 40]], dtype=np.float32)
        iou = iou_fn(boxes1, boxes1)
        assert iou.shape == (2, 2)
        assert np.allclose(iou, np.array([[1.0, 1.0], [1.0, 1.0]]))

    @pytest.mark.parametrize("iou_fn", [calculate_iou_batch, calculate_iou_batch_numba])
    def test_same_boxes_different_order(self, iou_fn):
        boxes1 = np.array([[0, 0, 10, 10], [20, 20, 10, 10]], dtype=np.float32)
        boxes2 = np.array([[20, 20, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
        iou = iou_fn(boxes1, boxes2)
        expected = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        assert np.allclose(iou, expected)

    @pytest.mark.parametrize("iou_fn", [calculate_iou_batch, calculate_iou_batch_numba])
    def test_tiny_boxes(self, iou_fn):
        boxes1 = np.array([[100, 100, 1e-5, 1e-5]], dtype=np.float32)
        boxes2 = np.array([[100, 100, 1e-5, 1e-5]], dtype=np.float32)
        iou = iou_fn(boxes1, boxes2)
        assert iou.shape == (1, 1)
        assert np.allclose(iou, 1.0)

    @pytest.mark.parametrize("iou_fn", [calculate_iou_batch, calculate_iou_batch_numba])
    def test_boundary_boxes(self, iou_fn):
        boxes1 = np.array([[0, 0, 1, 1]], dtype=np.float32)
        boxes2 = np.array([[0, 0, 1, 1]], dtype=np.float32)
        iou = iou_fn(boxes1, boxes2)
        assert iou.shape == (1, 1)
        assert np.allclose(iou, 1.0)

        boxes1 = np.array([[9999, 9999, 1, 1]], dtype=np.float32)
        boxes2 = np.array([[9999, 9999, 1, 1]], dtype=np.float32)
        iou = iou_fn(boxes1, boxes2)
        assert np.allclose(iou, 1.0)


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

    def test_false_positive(self):
        gt = [[{"category": "person", "bbox": [10, 10, 30, 40]}]]
        pred = [[{"category": "person", "bbox": [100, 100, 30, 40], "score": 1.0}]]

        metric = MeanAveragePrecision()
        metric.update(gt, pred)
        results = metric.compute()

        assert results["map"] == 0.0
        assert results["ap"][0.5] == 0.0

    def test_false_negative(self):
        gt = [[{"category": "person", "bbox": [10, 10, 30, 40]}]]
        pred = [[]]

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
                {"category": "person", "bbox": [10, 10, 30, 40], "score": 0.9},  # TP
                {"category": "person", "bbox": [100, 100, 30, 40], "score": 0.8},  # FP
                {"category": "cat", "bbox": [50, 50, 20, 30], "score": 0.7},  # FP
                # FN for dog
            ]
        ]

        metric = MeanAveragePrecision()
        metric.update(gt, pred)
        results = metric.compute()

        assert results["map"] <= 0.51

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
        """Tests precision-recall curve calculation with multiple detections.

        Creates 4 ground truth boxes at coordinates (0,0), (100,100), (200,200), (300,300).
        Makes 4 predictions:
        - 3 correct detections at (0,0), (100,100), (200,200) with decreasing confidence
        - 1 incorrect detection at (500,500) with lowest confidence

        The AP is computed using 101-point interpolation, which samples the precision values
        at 101 recall points [0, 0.01, 0.02, ..., 1.0] and averages them. This gives a close
        approximation to the true area under the precision-recall curve.

        In this case, we expect AP@0.5 to be approximately 0.75, but the actual value may
        differ slightly due to the interpolation method used.
        """
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

        assert abs(results["ap"][0.5] - 0.75) < 1e-2, results["ap"][0.5]

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
                {"category": "person", "bbox": [10, 10, 300, 400], "area": 300 * 400},
                {"category": "person", "bbox": [50, 50, 30, 40], "area": 30 * 40},
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

        assert np.allclose(results["map"], 0.9, atol=1e-2)

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

    def test_identical_scores(self):
        """
        Test that metric computation correctly penalizes duplicate predictions.
        """
        gt = [[{"category": "person", "bbox": [10, 10, 30, 40], "area": 30 * 40}]]
        pred = [
            [
                {"category": "person", "bbox": [11, 11, 30, 40], "score": 0.8},
                {"category": "person", "bbox": [12, 12, 30, 40], "score": 0.8},
                {"category": "person", "bbox": [13, 13, 30, 40], "score": 0.8},
            ]
        ]

        metric = MeanAveragePrecision()
        metric.update(gt, pred)
        results = metric.compute()
        assert np.allclose(results["ap"][0.5], 1 / 3, 1e-3)  # 1 TP, 2 FP

    def test_duplicate_predictions(self):
        gt = [[{"category": "person", "bbox": [10, 10, 30, 40]}]]
        pred = [
            [
                {"category": "person", "bbox": [10, 10, 30, 40], "score": 0.9},
                {"category": "person", "bbox": [10, 10, 30, 40], "score": 0.8},
            ]
        ]

        metric = MeanAveragePrecision(iou_thresholds=[0.5])
        metric.update(gt, pred)
        results = metric.compute()
        assert results["ap"][0.5] == 1.0

    def test_extended_area_ranges(self):
        """
        Test that area range filtering works correctly for different object sizes.
        Tests small (< 32), medium (32-96), and large (96+) objects, as well as
        an invalid range that should contain no objects.
        """
        gt = [
            [
                {"category": "person", "bbox": [10, 10, 5, 5], "area": 25},
                {"category": "person", "bbox": [10, 10, 6, 6], "area": 36},
                {"category": "person", "bbox": [20, 20, 10, 10], "area": 100},
                {"category": "person", "bbox": [30, 30, 20, 20], "area": 400},
            ]
        ]
        pred = [
            [
                {
                    "category": "person",
                    "bbox": [10, 10, 5, 5],
                    "score": 0.9,
                    "area": 25,
                },
                {
                    "category": "person",
                    "bbox": [10, 10, 6, 6],
                    "score": 0.9,
                    "area": 36,
                },
                {
                    "category": "person",
                    "bbox": [20, 20, 10, 10],
                    "score": 0.8,
                    "area": 100,
                },
                {
                    "category": "person",
                    "bbox": [30, 30, 20, 20],
                    "score": 0.7,
                    "area": 400,
                },
            ]
        ]

        # Test small objects
        results = average_precision(
            gt, pred, min_area=0, max_area=32, iou_thresholds=[0.5]
        )
        assert results["ap"][0.5] == 1.0

        # Test medium objects
        results = average_precision(
            gt, pred, min_area=32, max_area=96, iou_thresholds=[0.5]
        )
        assert results["ap"][0.5] == 1.0

        # Test large objects
        results = average_precision(
            gt, pred, min_area=96, max_area=1000, iou_thresholds=[0.5]
        )
        assert results["ap"][0.5] == 1.0

        # Test invalid range
        results = average_precision(
            gt, pred, min_area=1000, max_area=2000, iou_thresholds=[0.5]
        )
        assert results["ap"][0.5] == 0.0

    def test_numerical_precision(self):
        gt = [[{"category": "person", "bbox": [10, 10, 30, 40]}]]
        pred = [
            [
                {
                    "category": "person",
                    "bbox": [10 + 1e-10, 10 - 1e-10, 30, 40],
                    "score": 1.0,
                },
                {
                    "category": "person",
                    "bbox": [10.0000001, 10.0000001, 30, 40],
                    "score": 0.9,
                },
                {
                    "category": "person",
                    "bbox": [10 + np.finfo(np.float32).eps, 10, 30, 40],
                    "score": 0.8,
                },
            ]
        ]

        metric = MeanAveragePrecision()
        metric.update(gt, pred)
        results = metric.compute()
        assert results["ap"][0.5] == 1.0


class TestComputeAPAtIOU:
    def test_perfect_match(self):
        annotations_true = [[{"category": "person", "bbox": [10, 10, 30, 40]}]]
        annotations_pred = [
            [{"category": "person", "bbox": [10, 10, 30, 40], "score": 1.0}]
        ]
        ap, _ = compute_ap_ar_at_iou(annotations_true, annotations_pred, 0.5)
        assert ap == 1.0

    def test_no_predictions(self):
        annotations_true = [[{"category": "person", "bbox": [10, 10, 30, 40]}]]
        annotations_pred = [[]]
        ap, _ = compute_ap_ar_at_iou(annotations_true, annotations_pred, 0.5)
        assert ap == 0.0

    def test_no_ground_truth(self):
        annotations_true = [[]]
        annotations_pred = [
            [{"category": "person", "bbox": [10, 10, 30, 40], "score": 1.0}]
        ]
        ap, _ = compute_ap_ar_at_iou(annotations_true, annotations_pred, 0.5)
        assert ap == 0.0

    def test_multiple_predictions_single_truth(self):
        annotations_true = [[{"category": "person", "bbox": [10, 10, 30, 40]}]]
        annotations_pred = [
            [
                {"category": "person", "bbox": [10, 10, 30, 40], "score": 0.9},
                {"category": "person", "bbox": [100, 100, 30, 40], "score": 0.8},
            ]
        ]
        ap, _ = compute_ap_ar_at_iou(annotations_true, annotations_pred, 0.5)
        assert ap == 1.0

    def test_multiple_categories(self):
        annotations_true = [
            [
                {"category": "person", "bbox": [10, 10, 30, 40]},
                {"category": "dog", "bbox": [50, 50, 20, 30]},
            ]
        ]
        annotations_pred = [
            [
                {"category": "person", "bbox": [10, 10, 30, 40], "score": 0.9},
                {"category": "dog", "bbox": [50, 50, 20, 30], "score": 0.8},
            ]
        ]
        ap, _ = compute_ap_ar_at_iou(annotations_true, annotations_pred, 0.5)
        assert ap == 1.0

    def test_threshold_boundaries(self):
        annotations_true = [
            [{"category": "person", "bbox": [10, 10, 30, 40], "area": 30 * 40}]
        ]
        annotations_pred = [
            [
                {
                    "category": "person",
                    "bbox": [15, 15, 30, 40],
                    "score": 1.0,
                }
            ]
        ]
        ap_strict, _ = compute_ap_ar_at_iou(annotations_true, annotations_pred, 0.9)
        assert ap_strict == 0.0

        ap_lenient, _ = compute_ap_ar_at_iou(annotations_true, annotations_pred, 0.3)
        assert ap_lenient == 1.0

    def test_edge_thresholds(self):
        annotations_true = [
            [{"category": "person", "bbox": [10, 10, 30, 40], "area": 30 * 40}]
        ]
        annotations_pred = [
            [
                {
                    "category": "person",
                    "bbox": [10, 10, 30, 40],
                    "area": 30 * 40,
                    "score": 1.0,
                }
            ]
        ]

        ap_zero, _ = compute_ap_ar_at_iou(annotations_true, annotations_pred, 0.0)
        assert ap_zero == 1.0

        ap_one, _ = compute_ap_ar_at_iou(annotations_true, annotations_pred, 1.0)
        assert ap_one == 1.0

    def test_multiple_thresholds_same_prediction(self):
        annotations_true = [[{"category": "person", "bbox": [10, 10, 30, 40]}]]
        annotations_pred = [
            [{"category": "person", "bbox": [15, 15, 30, 40], "score": 1.0}]
        ]

        thresholds = [0.5, 0.7, 0.9]
        aps = []
        for t in thresholds:
            ap, _ = compute_ap_ar_at_iou(annotations_true, annotations_pred, t)
            aps.append(ap)

        assert aps[0] == 1.0
        assert aps[1] == 0.0
        assert aps[2] == 0.0

    def test_multiple_images(self):
        annotations_true = [
            [{"category": "person", "bbox": [10, 10, 30, 40]}],
            [{"category": "dog", "bbox": [50, 50, 20, 30]}],
        ]
        annotations_pred = [
            [{"category": "person", "bbox": [10, 10, 30, 40], "score": 0.9}],
            [{"category": "dog", "bbox": [50, 50, 20, 30], "score": 0.8}],
        ]
        ap, _ = compute_ap_ar_at_iou(annotations_true, annotations_pred, 0.5)
        assert ap == 1.0

    def test_multiple_images_empty_predictions(self):
        annotations_true = [
            [{"category": "person", "bbox": [10, 10, 30, 40]}],
            [{"category": "dog", "bbox": [50, 50, 20, 30]}],
            [],
        ]
        annotations_pred = [
            [{"category": "person", "bbox": [10, 10, 30, 40], "score": 0.9}],
            [],
            [{"category": "cat", "bbox": [100, 100, 25, 25], "score": 0.7}],
        ]
        ap, _ = compute_ap_ar_at_iou(annotations_true, annotations_pred, 0.5)
        assert np.allclose(ap, 0.5, atol=1e-2), (
            ap,
            0.5,
        )  # 1 correct out of 2 total ground truths

    def test_multiple_images_multiple_predictions(self):
        annotations_true = [
            [{"category": "person", "bbox": [10, 10, 30, 40]}],
            [{"category": "person", "bbox": [50, 50, 20, 30]}],
        ]
        annotations_pred = [
            [
                {"category": "person", "bbox": [10, 10, 30, 40], "score": 0.9},
                {"category": "person", "bbox": [15, 15, 30, 40], "score": 0.8},
            ],
            [
                {"category": "person", "bbox": [50, 50, 20, 30], "score": 0.95},
                {"category": "person", "bbox": [55, 55, 20, 30], "score": 0.85},
            ],
        ]
        ap, _ = compute_ap_ar_at_iou(annotations_true, annotations_pred, 0.5)
        assert ap == 1.0  # Both ground truths matched with highest scoring predictions


class TestComputeARAtIOU:
    def test_perfect_match(self):
        annotations_true = [[{"category": "person", "bbox": [10, 10, 30, 40]}]]
        annotations_pred = [
            [{"category": "person", "bbox": [10, 10, 30, 40], "score": 1.0}]
        ]
        _, ar = compute_ap_ar_at_iou(annotations_true, annotations_pred, 0.5)
        assert ar == 1.0

    def test_no_predictions(self):
        annotations_true = [[{"category": "person", "bbox": [10, 10, 30, 40]}]]
        annotations_pred = [[]]
        _, ar = compute_ap_ar_at_iou(annotations_true, annotations_pred, 0.5)
        assert ar == 0.0

    def test_no_ground_truth(self):
        annotations_true = [[]]
        annotations_pred = [
            [{"category": "person", "bbox": [10, 10, 30, 40], "score": 1.0}]
        ]
        _, ar = compute_ap_ar_at_iou(annotations_true, annotations_pred, 0.5)
        assert ar == 0.0

    def test_multiple_predictions_single_truth(self):
        annotations_true = [[{"category": "person", "bbox": [10, 10, 30, 40]}]]
        annotations_pred = [
            [
                {"category": "person", "bbox": [10, 10, 30, 40], "score": 0.9},
                {"category": "person", "bbox": [100, 100, 30, 40], "score": 0.8},
            ]
        ]
        _, ar = compute_ap_ar_at_iou(annotations_true, annotations_pred, 0.5)
        assert ar == 1.0

    def test_multiple_images(self):
        annotations_true = [
            [{"category": "person", "bbox": [10, 10, 30, 40]}],
            [{"category": "dog", "bbox": [50, 50, 20, 30]}],
        ]
        annotations_pred = [
            [{"category": "person", "bbox": [10, 10, 30, 40], "score": 0.9}],
            [{"category": "dog", "bbox": [50, 50, 20, 30], "score": 0.8}],
        ]
        _, ar = compute_ap_ar_at_iou(annotations_true, annotations_pred, 0.5)
        assert ar == 1.0

    def test_max_detections_limit(self):
        annotations_true = [[{"category": "person", "bbox": [10, 10, 30, 40]}]]
        annotations_pred = [
            [
                {"category": "person", "bbox": [100, 100, 30, 40], "score": 0.9},
                {"category": "person", "bbox": [10, 10, 30, 40], "score": 0.8},
            ]
        ]
        _, ar = compute_ap_ar_at_iou(
            annotations_true, annotations_pred, 0.5, max_detections=1
        )
        assert ar == 0.0
        _, ar = compute_ap_ar_at_iou(
            annotations_true, annotations_pred, 0.5, max_detections=2
        )
        assert ar == 1.0


class TestComputePrecisionRecall:
    def test_perfect_predictions(self):
        tp = np.array([1, 1, 1])
        fp = np.array([0, 0, 0])
        total_true = 3
        precision, recall = compute_precision_recall(tp, fp, total_true)
        assert np.allclose(precision, 1.0)
        assert np.allclose(recall, np.linspace(0, 1, 101))

    def test_no_predictions(self):
        tp = np.array([])
        fp = np.array([])
        total_true = 1
        precision, recall = compute_precision_recall(tp, fp, total_true)
        assert np.allclose(precision, 0.0)
        assert np.allclose(recall, np.linspace(0, 1, 101))

    def test_all_false_positives(self):
        tp = np.array([0, 0, 0])
        fp = np.array([1, 1, 1])
        total_true = 3
        precision, recall = compute_precision_recall(tp, fp, total_true)
        assert np.allclose(precision, 0.0)
        assert np.allclose(recall, np.linspace(0, 1, 101))

    def test_mixed_predictions(self):
        tp = np.array([1, 0, 1])
        fp = np.array([0, 1, 0])
        total_true = 2
        precision, recall = compute_precision_recall(tp, fp, total_true)
        expected_recall = np.linspace(0, 1, 101)
        assert len(precision) == 101
        assert np.allclose(recall, expected_recall)
        assert precision[0] == 1.0

    def test_no_ground_truth(self):
        tp = np.array([0, 0])
        fp = np.array([1, 1])
        total_true = 0
        precision, recall = compute_precision_recall(tp, fp, total_true)
        assert np.allclose(precision, 0.0)
        assert np.allclose(recall, np.linspace(0, 1, 101))

    def test_monotonic_precision(self):
        tp = np.array([1, 0, 1, 0])
        fp = np.array([0, 1, 0, 1])
        total_true = 2
        precision, recall = compute_precision_recall(tp, fp, total_true)
        assert np.all(np.diff(precision) <= 0)

    def test_numerical_stability(self):
        tp = np.array([1e-10, 1e-10])
        fp = np.array([1e-10, 1e-10])
        total_true = 2
        precision, recall = compute_precision_recall(tp, fp, total_true)
        assert not np.any(np.isnan(precision))
        assert not np.any(np.isnan(recall))

    def test_interpolation_points(self):
        tp = np.array([1, 1])
        fp = np.array([0, 0])
        total_true = 2
        precision, recall = compute_precision_recall(tp, fp, total_true)
        assert len(precision) == 101
        assert len(recall) == 101
        assert recall[0] == 0.0
        assert recall[-1] == 1.0

    def test_precision_endpoint(self):
        tp = np.array([1, 1])
        fp = np.array([0, 0])
        total_true = 2
        precision, recall = compute_precision_recall(tp, fp, total_true)
        assert precision[-1] == 1.0

    def test_large_numbers(self):
        tp = np.array([1000, 2000, 3000])
        fp = np.array([100, 200, 300])
        total_true = 3000
        precision, recall = compute_precision_recall(tp, fp, total_true)
        assert not np.any(np.isnan(precision))
        assert not np.any(precision > 1.0)
