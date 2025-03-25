import pytest
import numpy as np
from sane_coco.matching import (
    match_predictions_to_ground_truth,
)
from sane_coco.util import calculate_iou_batch


class TestMatching:
    def test_perfect_match(self):
        true_anns = [[10.0, 10.0, 20.0, 20.0]]
        pred_anns = [[10.0, 10.0, 20.0, 20.0]]
        scores = [0.9]

        tp, fp, scores = match_predictions_to_ground_truth(
            true_anns, pred_anns, scores, 0.5
        )

        assert tp == [1]
        assert fp == [0]
        assert scores == [0.9]

    def test_no_match_low_iou(self):
        true_anns = [[10, 10, 20, 20]]
        pred_anns = [[50, 50, 20, 20]]
        scores = [0.9]

        tp, fp, scores = match_predictions_to_ground_truth(
            true_anns, pred_anns, scores, 0.5
        )

        assert tp == [0]
        assert fp == [1]
        assert scores == [0.9]

    def test_multiple_predictions(self):
        true_anns = [[10, 10, 20, 20]]
        pred_anns = [[10, 10, 20, 20], [15, 15, 20, 20]]
        scores = [0.9, 0.8]

        tp, fp, scores = match_predictions_to_ground_truth(
            true_anns, pred_anns, scores, 0.5
        )

        assert tp == [1, 0]
        assert fp == [0, 1]
        assert scores == [0.9, 0.8]

    def test_multiple_ground_truth(self):
        true_anns = [[10, 10, 20, 20], [50, 50, 20, 20]]
        pred_anns = [[10, 10, 20, 20], [50, 50, 20, 20]]
        scores = [0.9, 0.8]

        tp, fp, scores = match_predictions_to_ground_truth(
            true_anns, pred_anns, scores, 0.5
        )

        assert tp == [1, 1]
        assert fp == [0, 0]
        assert scores == [0.9, 0.8]

    def test_empty_predictions(self):
        true_anns = [[10, 10, 20, 20]]
        pred_anns = []
        scores = []

        tp, fp, scores = match_predictions_to_ground_truth(
            true_anns, pred_anns, scores, 0.5
        )

        assert tp == []
        assert fp == []
        assert scores == []

    def test_empty_ground_truth(self):
        true_anns = []
        pred_anns = [[10, 10, 20, 20]]
        scores = [0.9]

        tp, fp, scores = match_predictions_to_ground_truth(
            true_anns, pred_anns, scores, 0.5
        )

        assert tp == [0]
        assert fp == [1]
        assert scores == [0.9]

    def test_score_sorting(self):
        true_anns = [[10, 10, 20, 20]]
        pred_anns = [[12, 12, 20, 20], [10, 10, 20, 20]]
        scores = [0.7, 0.9]

        tp, fp, scores = match_predictions_to_ground_truth(
            true_anns, pred_anns, scores, 0.5
        )

        assert tp == [1, 0]
        assert fp == [0, 1]
        assert scores == [0.9, 0.7]

    def test_coco_single_match_per_ground_truth(self):
        true_anns = [[10, 10, 20, 20]]
        pred_anns = [[10, 10, 20, 20], [11, 11, 20, 20], [12, 12, 20, 20]]
        scores = [0.9, 0.8, 0.7]

        tp, fp, scores = match_predictions_to_ground_truth(
            true_anns, pred_anns, scores, 0.5
        )

        assert sum(tp) == 1
        assert sum(fp) == 2

    def test_coco_predictions_sorted_by_score(self):
        true_anns = [[10, 10, 20, 20]]
        pred_anns = [[30, 30, 20, 20], [10, 10, 20, 20], [20, 20, 20, 20]]
        scores = [0.5, 0.8, 0.6]

        tp, fp, scores = match_predictions_to_ground_truth(
            true_anns, pred_anns, scores, 0.5
        )

        assert scores == [0.8, 0.6, 0.5]
        assert tp == [1, 0, 0]

    def test_border_case_iou_exactly_at_threshold(self):
        true_anns = [[0, 0, 10, 10]]
        pred_anns = [[5, 0, 10, 10]]
        scores = [0.9]

        tp_above, fp_above, _ = match_predictions_to_ground_truth(
            true_anns, pred_anns, scores, 0.34
        )
        tp_below, fp_below, _ = match_predictions_to_ground_truth(
            true_anns, pred_anns, scores, 0.33
        )

        assert tp_above == [0]
        assert fp_above == [1]
        assert tp_below == [1]
        assert fp_below == [0]

    def test_different_sized_boxes_same_center(self):
        true_anns = [[5, 5, 10, 10]]
        pred_anns = [[0, 0, 20, 20]]
        scores = [0.9]

        iou = float(calculate_iou_batch(np.array(true_anns), np.array(pred_anns))[0][0])
        tp, fp, scores = match_predictions_to_ground_truth(
            true_anns, pred_anns, scores, iou
        )
        assert tp == [1]
        assert fp == [0]

    def test_invalid_box_coordinates(self):
        true_anns = [[10, 10, 20, 20]]
        pred_anns = [[20, 20, 10, 10]]
        scores = [0.9]
        tp, fp, scores = match_predictions_to_ground_truth(
            true_anns, pred_anns, scores, 0.5
        )
        assert tp == [0]
        assert fp == [1]

    def test_zero_iou(self):
        true_anns = [[0, 0, 10, 10]]
        pred_anns = [[20, 20, 10, 10]]
        scores = [0.9]
        tp, fp, scores = match_predictions_to_ground_truth(
            true_anns, pred_anns, scores, 0.5
        )
        assert tp == [0]
        assert fp == [1]

    def test_overlapping_boxes_same_category(self):
        true_anns = [[10, 10, 20, 20], [15, 15, 20, 20]]
        pred_anns = [[10, 10, 20, 20], [15, 15, 20, 20]]
        scores = [0.9, 0.8]
        tp, fp, scores = match_predictions_to_ground_truth(
            true_anns, pred_anns, scores, 0.5
        )
        assert tp == [1, 1]
        assert fp == [0, 0]

    def test_extremely_large_box(self):
        true_anns = [[10, 10, 20, 20]]
        pred_anns = [[0, 0, 1000, 1000]]
        scores = [0.9]
        tp, fp, scores = match_predictions_to_ground_truth(
            true_anns, pred_anns, scores, 0.5
        )
        assert tp == [0]
        assert fp == [1]

    def test_extremely_small_box(self):
        true_anns = [[10, 10, 0.001, 0.001]]
        pred_anns = [[10, 10, 0.001, 0.001]]
        scores = [0.9]
        tp, fp, scores = match_predictions_to_ground_truth(
            true_anns, pred_anns, scores, 0.5
        )
        assert tp == [1]
        assert fp == [0]
