from sane_coco import COCODataset
from sane_coco.metrics import MeanAveragePrecision, compute_ap_at_iou, compute_ar_at_iou
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


class TestComputeAPAtIOU:
    def test_perfect_match(self):
        annotations_true = [[{"category": "person", "bbox": [10, 10, 30, 40]}]]
        annotations_pred = [
            [{"category": "person", "bbox": [10, 10, 30, 40], "score": 1.0}]
        ]
        ap = compute_ap_at_iou(annotations_true, annotations_pred, 0.5)
        assert ap == 1.0

    def test_no_predictions(self):
        annotations_true = [[{"category": "person", "bbox": [10, 10, 30, 40]}]]
        annotations_pred = [[]]
        ap = compute_ap_at_iou(annotations_true, annotations_pred, 0.5)
        assert ap == 0.0

    def test_no_ground_truth(self):
        annotations_true = [[]]
        annotations_pred = [
            [{"category": "person", "bbox": [10, 10, 30, 40], "score": 1.0}]
        ]
        ap = compute_ap_at_iou(annotations_true, annotations_pred, 0.5)
        assert ap == 0.0

    def test_multiple_predictions_single_truth(self):
        annotations_true = [[{"category": "person", "bbox": [10, 10, 30, 40]}]]
        annotations_pred = [
            [
                {"category": "person", "bbox": [10, 10, 30, 40], "score": 0.9},
                {"category": "person", "bbox": [100, 100, 30, 40], "score": 0.8},
            ]
        ]
        ap = compute_ap_at_iou(annotations_true, annotations_pred, 0.5)
        assert ap == 1.0

    def test_wrong_category(self):
        annotations_true = [[{"category": "person", "bbox": [10, 10, 30, 40]}]]
        annotations_pred = [
            [{"category": "dog", "bbox": [10, 10, 30, 40], "score": 1.0}]
        ]
        ap = compute_ap_at_iou(annotations_true, annotations_pred, 0.5)
        assert ap == 0.0

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
        ap = compute_ap_at_iou(annotations_true, annotations_pred, 0.5)
        assert ap == 1.0

    def test_threshold_boundaries(self):
        annotations_true = [[{"category": "person", "bbox": [10, 10, 30, 40]}]]
        annotations_pred = [
            [{"category": "person", "bbox": [15, 15, 30, 40], "score": 1.0}]
        ]

        ap_strict = compute_ap_at_iou(annotations_true, annotations_pred, 0.9)
        assert ap_strict == 0.0

        ap_lenient = compute_ap_at_iou(annotations_true, annotations_pred, 0.3)
        assert ap_lenient == 1.0

    def test_multiple_thresholds_same_prediction(self):
        annotations_true = [[{"category": "person", "bbox": [10, 10, 30, 40]}]]
        annotations_pred = [
            [{"category": "person", "bbox": [15, 15, 30, 40], "score": 1.0}]
        ]

        thresholds = [0.5, 0.7, 0.9]
        aps = []
        for t in thresholds:
            ap = compute_ap_at_iou(annotations_true, annotations_pred, t)
            aps.append(ap)

        assert aps[0] == 1.0
        assert aps[1] == 0.0
        assert aps[2] == 0.0

    def test_edge_thresholds(self):
        annotations_true = [[{"category": "person", "bbox": [10, 10, 30, 40]}]]
        annotations_pred = [
            [{"category": "person", "bbox": [10, 10, 30, 40], "score": 1.0}]
        ]

        ap_zero = compute_ap_at_iou(annotations_true, annotations_pred, 0.0)
        assert ap_zero == 1.0

        ap_one = compute_ap_at_iou(annotations_true, annotations_pred, 1.0)
        assert ap_one == 1.0

    def test_multiple_images(self):
        annotations_true = [
            [{"category": "person", "bbox": [10, 10, 30, 40]}],
            [{"category": "dog", "bbox": [50, 50, 20, 30]}],
        ]
        annotations_pred = [
            [{"category": "person", "bbox": [10, 10, 30, 40], "score": 0.9}],
            [{"category": "dog", "bbox": [50, 50, 20, 30], "score": 0.8}],
        ]
        ap = compute_ap_at_iou(annotations_true, annotations_pred, 0.5)
        assert ap == 1.0

    def test_multiple_images_mixed_results(self):
        annotations_true = [
            [{"category": "person", "bbox": [10, 10, 30, 40]}],
            [{"category": "dog", "bbox": [50, 50, 20, 30]}],
            [{"category": "cat", "bbox": [100, 100, 25, 25]}],
        ]
        annotations_pred = [
            [{"category": "person", "bbox": [12, 12, 30, 40], "score": 0.9}],
            [
                {"category": "cat", "bbox": [51, 51, 20, 30], "score": 0.8}
            ],  # wrong category
            [{"category": "cat", "bbox": [100, 100, 25, 25], "score": 0.7}],
        ]
        ap = compute_ap_at_iou(annotations_true, annotations_pred, 0.5)
        assert abs(ap - 0.556) < 1e-3  # Area under precision-recall curve

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
        ap = compute_ap_at_iou(annotations_true, annotations_pred, 0.5)
        assert ap == 0.5  # 1 correct out of 2 total ground truths

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
        ap = compute_ap_at_iou(annotations_true, annotations_pred, 0.5)
        assert ap == 1.0  # Both ground truths matched with highest scoring predictions


class TestComputeARAtIOU:
    def test_perfect_match(self):
        annotations_true = [[{"category": "person", "bbox": [10, 10, 30, 40]}]]
        annotations_pred = [
            [{"category": "person", "bbox": [10, 10, 30, 40], "score": 1.0}]
        ]
        ar = compute_ar_at_iou(annotations_true, annotations_pred, 0.5)
        assert ar == 1.0

    def test_no_predictions(self):
        annotations_true = [[{"category": "person", "bbox": [10, 10, 30, 40]}]]
        annotations_pred = [[]]
        ar = compute_ar_at_iou(annotations_true, annotations_pred, 0.5)
        assert ar == 0.0

    def test_no_ground_truth(self):
        annotations_true = [[]]
        annotations_pred = [
            [{"category": "person", "bbox": [10, 10, 30, 40], "score": 1.0}]
        ]
        ar = compute_ar_at_iou(annotations_true, annotations_pred, 0.5)
        assert ar == 0.0

    def test_multiple_predictions_single_truth(self):
        annotations_true = [[{"category": "person", "bbox": [10, 10, 30, 40]}]]
        annotations_pred = [
            [
                {"category": "person", "bbox": [10, 10, 30, 40], "score": 0.9},
                {"category": "person", "bbox": [100, 100, 30, 40], "score": 0.8},
            ]
        ]
        ar = compute_ar_at_iou(annotations_true, annotations_pred, 0.5)
        assert ar == 1.0

    def test_area_range_filtering(self):
        annotations_true = [[{"category": "person", "bbox": [10, 10, 5, 5]}]]
        annotations_pred = [
            [{"category": "person", "bbox": [10, 10, 5, 5], "score": 1.0}]
        ]
        ar = compute_ar_at_iou(
            annotations_true, annotations_pred, 0.5, area_range=(0, 30)
        )
        assert ar == 1.0
        ar = compute_ar_at_iou(
            annotations_true, annotations_pred, 0.5, area_range=(30, 100)
        )
        assert ar == 0.0

    def test_multiple_images(self):
        annotations_true = [
            [{"category": "person", "bbox": [10, 10, 30, 40]}],
            [{"category": "dog", "bbox": [50, 50, 20, 30]}],
        ]
        annotations_pred = [
            [{"category": "person", "bbox": [10, 10, 30, 40], "score": 0.9}],
            [{"category": "dog", "bbox": [50, 50, 20, 30], "score": 0.8}],
        ]
        ar = compute_ar_at_iou(annotations_true, annotations_pred, 0.5)
        assert ar == 1.0

    def test_max_detections_limit(self):
        annotations_true = [[{"category": "person", "bbox": [10, 10, 30, 40]}]]
        annotations_pred = [
            [
                {"category": "person", "bbox": [100, 100, 30, 40], "score": 0.9},
                {"category": "person", "bbox": [10, 10, 30, 40], "score": 0.8},
            ]
        ]
        ar = compute_ar_at_iou(
            annotations_true, annotations_pred, 0.5, max_detections=1
        )
        assert ar == 0.0
        ar = compute_ar_at_iou(
            annotations_true, annotations_pred, 0.5, max_detections=2
        )
        assert ar == 1.0

    def test_mixed_categories(self):
        annotations_true = [
            [
                {"category": "person", "bbox": [10, 10, 30, 40]},
                {"category": "dog", "bbox": [50, 50, 20, 30]},
            ]
        ]
        annotations_pred = [
            [
                {"category": "person", "bbox": [10, 10, 30, 40], "score": 0.9},
                {"category": "cat", "bbox": [50, 50, 20, 30], "score": 0.8},
            ]
        ]
        ar = compute_ar_at_iou(annotations_true, annotations_pred, 0.5)
        assert ar == 0.5
