import pytest

from sane_coco.metrics import MeanAveragePrecision


@pytest.fixture
def torchmetrics_data():
    preds = [
        {
            "boxes": [[258.15, 41.29, 606.41, 285.07]],
            "scores": [0.236],
            "labels": [4],
        },  # coco image id 42
        {
            "boxes": [[61.00, 22.75, 565.00, 632.42], [12.66, 3.32, 281.26, 275.23]],
            "scores": [0.318, 0.726],
            "labels": [3, 2],
        },  # coco image id 73
        {
            "boxes": [
                [87.87, 276.25, 384.29, 379.43],
                [0.00, 3.66, 142.15, 316.06],
                [296.55, 93.96, 314.97, 152.79],
                [328.94, 97.05, 342.49, 122.98],
                [356.62, 95.47, 372.33, 147.55],
                [464.08, 105.09, 495.74, 146.99],
                [276.11, 103.84, 291.44, 150.72],
            ],
            "scores": [0.546, 0.3, 0.407, 0.611, 0.335, 0.805, 0.953],
            "labels": [4, 1, 0, 0, 0, 0, 0],
        },  # coco image id 74
        {
            "boxes": [
                [72.92, 45.96, 91.23, 80.57],
                [45.17, 45.34, 66.28, 79.83],
                [82.28, 47.04, 99.66, 78.50],
                [59.96, 46.17, 80.35, 80.48],
                [75.29, 23.01, 91.85, 50.85],
                [71.14, 1.10, 96.96, 28.33],
                [61.34, 55.23, 77.14, 79.57],
                [41.17, 45.78, 60.99, 78.48],
                [56.18, 44.80, 64.42, 56.25],
            ],
            "scores": [
                0.532,
                0.204,
                0.782,
                0.202,
                0.883,
                0.271,
                0.561,
                0.204
                + 1e-8,  # There are some problems with sorting at the moment. When sorting score with the same values, they give different indexes. # noqa: E501
                0.349,
            ],
            "labels": [49, 49, 49, 49, 49, 49, 49, 49, 49],
        },  # coco image id 987 category_id 49
    ]

    target = [
        {
            "boxes": [[214.1500, 41.2900, 562.4100, 285.0700]],
            "labels": [4],
        },  # coco image id 42
        {
            "boxes": [
                [13.00, 22.75, 548.98, 632.42],
                [1.66, 3.32, 270.26, 275.23],
            ],
            "labels": [2, 2],
        },  # coco image id 73
        {
            "boxes": [
                [61.87, 276.25, 358.29, 379.43],
                [2.75, 3.66, 162.15, 316.06],
                [295.55, 93.96, 313.97, 152.79],
                [326.94, 97.05, 340.49, 122.98],
                [356.62, 95.47, 372.33, 147.55],
                [462.08, 105.09, 493.74, 146.99],
                [277.11, 103.84, 292.44, 150.72],
            ],
            "labels": [4, 1, 0, 0, 0, 0, 0],
        },  # coco image id 74
        {
            "boxes": [
                [72.92, 45.96, 91.23, 80.57],
                [50.17, 45.34, 71.28, 79.83],
                [81.28, 47.04, 98.66, 78.50],
                [63.96, 46.17, 84.35, 80.48],
                [75.29, 23.01, 91.85, 50.85],
                [56.39, 21.65, 75.66, 45.54],
                [73.14, 1.10, 98.96, 28.33],
                [62.34, 55.23, 78.14, 79.57],
                [44.17, 45.78, 63.99, 78.48],
                [58.18, 44.80, 66.42, 56.25],
            ],
            "labels": [49, 49, 49, 49, 49, 49, 49, 49, 49, 49],
        },  # coco image id 987 category_id 49
    ]

    return preds, target


class TestTorchmetricsInterface:
    def test_torchmetrics_evaluate(self, torchmetrics_data):
        preds, target = torchmetrics_data
        metric = MeanAveragePrecision()
        metric.update(target, preds)
        results = metric.compute()

        assert "map" in results
        assert "ap" in results
        assert 0.5 in results["ap"]
        assert 0.75 in results["ap"]
