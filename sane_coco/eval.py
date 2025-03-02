import numpy as np
from typing import List, Dict, Any, Union, Tuple, Optional, Sequence
from collections import defaultdict

from .dataset import BBox, Mask, CocoDataset, Category, Annotation


def evaluate(
    gt_annotations: List[Annotation] = None,
    predictions: List[Tuple[BBox, Category, float, Optional[Mask]]] = None,
    gt_bboxes: Optional[List[BBox]] = None,
    gt_categories: Optional[List[Category]] = None,
    pred_bboxes: Optional[List[BBox]] = None,
    pred_categories: Optional[List[Category]] = None,
    pred_scores: Optional[List[float]] = None,
    gt_masks: Optional[List[Mask]] = None,
    pred_masks: Optional[List[Mask]] = None,
    iou_thresholds: List[float] = [0.5, 0.75],
    max_detections: int = 100,
    eval_bbox: bool = True,
    eval_segm: bool = False,
    return_per_class: bool = False
) -> Dict[str, Any]:
    if gt_annotations and predictions:
        gt_bboxes = [ann.bbox for ann in gt_annotations]
        gt_categories = [ann.category for ann in gt_annotations]
        gt_masks = [ann.mask for ann in gt_annotations if ann.mask]
        
        pred_bboxes = [p[0] for p in predictions]
        pred_categories = [p[1] for p in predictions]
        pred_scores = [p[2] for p in predictions]
        pred_masks = [p[3] for p in predictions if p[3]]
    
    if eval_bbox and (gt_bboxes is None or pred_bboxes is None):
        raise ValueError("gt_bboxes and pred_bboxes must be provided for bbox evaluation")
    
    if eval_segm and (gt_masks is None or pred_masks is None):
        raise ValueError("gt_masks and pred_masks must be provided for segmentation evaluation")
    
    if not eval_bbox and not eval_segm:
        eval_bbox = True
    
    results = {}
    
    if eval_bbox:
        bbox_results = _evaluate_detections(
            gt_bboxes=gt_bboxes,
            gt_categories=gt_categories,
            pred_bboxes=pred_bboxes,
            pred_categories=pred_categories,
            pred_scores=pred_scores,
            iou_thresholds=iou_thresholds,
            max_detections=max_detections,
            return_per_class=return_per_class
        )
        
        if eval_segm:
            results["bbox"] = bbox_results
        else:
            results = bbox_results
    
    if eval_segm:
        segm_results = _evaluate_segmentations(
            gt_masks=gt_masks,
            gt_categories=gt_categories,
            pred_masks=pred_masks,
            pred_categories=pred_categories,
            pred_scores=pred_scores,
            iou_thresholds=iou_thresholds,
            max_detections=max_detections,
            return_per_class=return_per_class
        )
        
        if eval_bbox:
            results["segm"] = segm_results
        else:
            results = segm_results
    
    return results


def _evaluate_detections(
    gt_bboxes: List[BBox],
    gt_categories: List[Category],
    pred_bboxes: List[BBox],
    pred_categories: List[Category],
    pred_scores: List[float],
    iou_thresholds: List[float] = [0.5, 0.75],
    max_detections: int = 100,
    return_per_class: bool = False
) -> Dict[str, Any]:
    if len(gt_categories) != len(gt_bboxes):
        raise ValueError("gt_categories and gt_bboxes must have the same length")
    
    if not (len(pred_bboxes) == len(pred_categories) == len(pred_scores)):
        raise ValueError("pred_bboxes, pred_categories, and pred_scores must have the same length")
    
    sorted_indices = np.argsort(-np.array(pred_scores))
    pred_bboxes = [pred_bboxes[i] for i in sorted_indices[:max_detections]]
    pred_categories = [pred_categories[i] for i in sorted_indices[:max_detections]]
    pred_scores = [pred_scores[i] for i in sorted_indices[:max_detections]]
    
    category_set = set(cat.name for cat in gt_categories) | set(cat.name for cat in pred_categories)
    
    results = {
        "precision": {},
        "recall": {},
        "map": 0.0,
        "map_per_category": {},
        "per_class": {} if return_per_class else None
    }
    
    ap_values = []
    
    for iou_threshold in iou_thresholds:
        tp = []
        fp = []
        scores = []
        
        gt_matched = [False] * len(gt_bboxes)
        
        for pred_box, pred_cat, pred_score in zip(pred_bboxes, pred_categories, pred_scores):
            max_iou = -1
            max_idx = -1
            
            for i, (gt_box, gt_cat) in enumerate(zip(gt_bboxes, gt_categories)):
                if gt_matched[i] or gt_cat.name != pred_cat.name:
                    continue
                
                iou = pred_box.iou(gt_box)
                if iou > max_iou:
                    max_iou = iou
                    max_idx = i
            
            scores.append(pred_score)
            if max_idx >= 0 and max_iou >= iou_threshold:
                tp.append(1)
                fp.append(0)
                gt_matched[max_idx] = True
            else:
                tp.append(0)
                fp.append(1)
        
        tp = np.array(tp)
        fp = np.array(fp)
        scores = np.array(scores)
        
        if len(scores) == 0:
            results["precision"][iou_threshold] = np.array([])
            results["recall"][iou_threshold] = np.array([])
            results["map_per_category"][iou_threshold] = 0.0
            continue
        
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        
        precision = tp_cum / (tp_cum + fp_cum + np.finfo(float).eps)
        recall = tp_cum / (len(gt_bboxes) + np.finfo(float).eps)
        
        ap = 0.0
        for r in np.arange(0, 1.1, 0.1):
            if np.any(recall >= r):
                ap += np.max(precision[recall >= r]) / 11
        
        results["precision"][iou_threshold] = precision
        results["recall"][iou_threshold] = recall
        results["map_per_category"][iou_threshold] = ap
        ap_values.append(ap)
        
        if return_per_class:
            for cat_name in category_set:
                if cat_name not in results["per_class"]:
                    results["per_class"][cat_name] = {"ap": {}}
                
                cat_gt_indices = [i for i, c in enumerate(gt_categories) if c.name == cat_name]
                cat_pred_indices = [i for i, c in enumerate(pred_categories) if c.name == cat_name]
                
                if not cat_gt_indices or not cat_pred_indices:
                    results["per_class"][cat_name]["ap"][iou_threshold] = 0.0
                    continue
                
                cat_tp = tp[cat_pred_indices]
                cat_fp = fp[cat_pred_indices]
                cat_scores = scores[cat_pred_indices]
                
                if len(cat_scores) == 0:
                    results["per_class"][cat_name]["ap"][iou_threshold] = 0.0
                    continue
                
                cat_indices = np.argsort(-cat_scores)
                cat_tp = cat_tp[cat_indices]
                cat_fp = cat_fp[cat_indices]
                
                cat_tp_cum = np.cumsum(cat_tp)
                cat_fp_cum = np.cumsum(cat_fp)
                
                cat_precision = cat_tp_cum / (cat_tp_cum + cat_fp_cum + np.finfo(float).eps)
                cat_recall = cat_tp_cum / (len(cat_gt_indices) + np.finfo(float).eps)
                
                cat_ap = 0.0
                for r in np.arange(0, 1.1, 0.1):
                    if np.any(cat_recall >= r):
                        cat_ap += np.max(cat_precision[cat_recall >= r]) / 11
                
                results["per_class"][cat_name]["ap"][iou_threshold] = cat_ap
    
    results["map"] = np.mean(ap_values)
    
    if return_per_class:
        for cat_name in category_set:
            if cat_name in results["per_class"]:
                results["per_class"][cat_name]["map"] = np.mean(list(results["per_class"][cat_name]["ap"].values()))
    
    return results


def _evaluate_segmentations(
    gt_masks: List[Mask],
    gt_categories: List[Category],
    pred_masks: List[Mask],
    pred_categories: List[Category],
    pred_scores: List[float],
    iou_thresholds: List[float] = [0.5, 0.75],
    max_detections: int = 100,
    return_per_class: bool = False
) -> Dict[str, Any]:
    if len(gt_categories) != len(gt_masks):
        raise ValueError("gt_categories and gt_masks must have the same length")
    
    if not (len(pred_masks) == len(pred_categories) == len(pred_scores)):
        raise ValueError("pred_masks, pred_categories, and pred_scores must have the same length")
    
    sorted_indices = np.argsort(-np.array(pred_scores))
    pred_masks = [pred_masks[i] for i in sorted_indices[:max_detections]]
    pred_categories = [pred_categories[i] for i in sorted_indices[:max_detections]]
    pred_scores = [pred_scores[i] for i in sorted_indices[:max_detections]]
    
    category_set = set(cat.name for cat in gt_categories) | set(cat.name for cat in pred_categories)
    
    results = {
        "precision": {},
        "recall": {},
        "map": 0.0,
        "map_per_category": {},
        "per_class": {} if return_per_class else None
    }
    
    ap_values = []
    
    for iou_threshold in iou_thresholds:
        tp = []
        fp = []
        scores = []
        
        gt_matched = [False] * len(gt_masks)
        
        for pred_mask, pred_cat, pred_score in zip(pred_masks, pred_categories, pred_scores):
            max_iou = -1
            max_idx = -1
            
            for i, (gt_mask, gt_cat) in enumerate(zip(gt_masks, gt_categories)):
                if gt_matched[i] or gt_cat.name != pred_cat.name:
                    continue
                
                try:
                    iou = pred_mask.iou(gt_mask)
                    if iou > max_iou:
                        max_iou = iou
                        max_idx = i
                except ValueError:
                    continue
            
            scores.append(pred_score)
            if max_idx >= 0 and max_iou >= iou_threshold:
                tp.append(1)
                fp.append(0)
                gt_matched[max_idx] = True
            else:
                tp.append(0)
                fp.append(1)
        
        tp = np.array(tp)
        fp = np.array(fp)
        scores = np.array(scores)
        
        if len(scores) == 0:
            results["precision"][iou_threshold] = np.array([])
            results["recall"][iou_threshold] = np.array([])
            results["map_per_category"][iou_threshold] = 0.0
            continue
        
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        
        precision = tp_cum / (tp_cum + fp_cum + np.finfo(float).eps)
        recall = tp_cum / (len(gt_masks) + np.finfo(float).eps)
        
        ap = 0.0
        for r in np.arange(0, 1.1, 0.1):
            if np.any(recall >= r):
                ap += np.max(precision[recall >= r]) / 11
        
        results["precision"][iou_threshold] = precision
        results["recall"][iou_threshold] = recall
        results["map_per_category"][iou_threshold] = ap
        ap_values.append(ap)
        
        if return_per_class:
            for cat_name in category_set:
                if cat_name not in results["per_class"]:
                    results["per_class"][cat_name] = {"ap": {}}
                
                cat_gt_indices = [i for i, c in enumerate(gt_categories) if c.name == cat_name]
                cat_pred_indices = [i for i, c in enumerate(pred_categories) if c.name == cat_name]
                
                if not cat_gt_indices or not cat_pred_indices:
                    results["per_class"][cat_name]["ap"][iou_threshold] = 0.0
                    continue
                
                cat_tp = tp[cat_pred_indices]
                cat_fp = fp[cat_pred_indices]
                cat_scores = scores[cat_pred_indices]
                
                if len(cat_scores) == 0:
                    results["per_class"][cat_name]["ap"][iou_threshold] = 0.0
                    continue
                
                cat_indices = np.argsort(-cat_scores)
                cat_tp = cat_tp[cat_indices]
                cat_fp = cat_fp[cat_indices]
                
                cat_tp_cum = np.cumsum(cat_tp)
                cat_fp_cum = np.cumsum(cat_fp)
                
                cat_precision = cat_tp_cum / (cat_tp_cum + cat_fp_cum + np.finfo(float).eps)
                cat_recall = cat_tp_cum / (len(cat_gt_indices) + np.finfo(float).eps)
                
                cat_ap = 0.0
                for r in np.arange(0, 1.1, 0.1):
                    if np.any(cat_recall >= r):
                        cat_ap += np.max(cat_precision[cat_recall >= r]) / 11
                
                results["per_class"][cat_name]["ap"][iou_threshold] = cat_ap
    
    results["map"] = np.mean(ap_values)
    
    if return_per_class:
        for cat_name in category_set:
            if cat_name in results["per_class"]:
                results["per_class"][cat_name]["map"] = np.mean(list(results["per_class"][cat_name]["ap"].values()))
    
    return results


def evaluate_detections(
    predictions: List[Dict[str, Any]],
    ground_truth: CocoDataset,
    iou_thresholds: List[float] = [0.5, 0.75],
    max_detections: int = 100
) -> Dict[str, Any]:
    pred_bboxes = [pred["bbox"] for pred in predictions]
    pred_categories = [ground_truth.get_category_by_id(pred["category_id"]) for pred in predictions]
    pred_scores = [pred["score"] for pred in predictions]
    
    gt_bboxes = [ann.bbox for ann in ground_truth.annotations]
    gt_categories = [ann.category for ann in ground_truth.annotations]
    
    return evaluate(
        gt_bboxes=gt_bboxes,
        gt_categories=gt_categories,
        pred_bboxes=pred_bboxes,
        pred_categories=pred_categories,
        pred_scores=pred_scores,
        iou_thresholds=iou_thresholds,
        max_detections=max_detections
    )


def evaluate_segmentations(
    predictions: List[Dict[str, Any]],
    ground_truth: CocoDataset,
    iou_thresholds: List[float] = [0.5, 0.75],
    max_detections: int = 100
) -> Dict[str, Any]:
    pred_masks = [pred["mask"] for pred in predictions if "mask" in pred]
    pred_categories = [ground_truth.get_category_by_id(pred["category_id"]) for pred in predictions if "mask" in pred]
    pred_scores = [pred["score"] for pred in predictions if "mask" in pred]
    
    gt_masks = []
    gt_categories = []
    
    for ann in ground_truth.annotations:
        if ann.segmentation:
            gt_masks.append(ann.mask)
            gt_categories.append(ann.category)
    
    return evaluate(
        gt_masks=gt_masks,
        gt_categories=gt_categories,
        pred_masks=pred_masks,
        pred_categories=pred_categories,
        pred_scores=pred_scores,
        iou_thresholds=iou_thresholds,
        max_detections=max_detections,
        eval_bbox=False,
        eval_segm=True
    )


def plot_precision_recall(precision, recall, title="Precision-Recall Curve"):
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 8))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.grid(True)
        
        if isinstance(precision, dict) and isinstance(recall, dict):
            for threshold in precision.keys():
                if threshold in recall:
                    plt.plot(recall[threshold], precision[threshold], 
                            label=f'IoU={threshold}')
        else:
            plt.plot(recall, precision)
            
        plt.legend()
        plt.show()
    except ImportError:
        print("Matplotlib is required for plotting precision-recall curves")