from typing import Dict, Any, List, Optional, Union
import numpy as np

from .dataset import CocoDataset


class CocoEvaluator:
    def __init__(self, ground_truth: CocoDataset, predictions: CocoDataset, iou_type: str = 'bbox'):
        self.ground_truth = ground_truth
        self.predictions = predictions
        self.iou_type = iou_type
        self._results = {"precision": np.zeros(10), "recall": np.zeros(10), "AP": [0.0]}
    
    def evaluate(self):
        self._results = {
            "precision": np.zeros(10),
            "recall": np.zeros(10),
            "AP": [0.0]
        }
    
    def summarize(self):
        if self._results is None:
            self.evaluate()
        
        for metric, value in self.results.items():
            if isinstance(value, (int, float)):
                print(f"{metric}: {value:.3f}")
            elif isinstance(value, np.ndarray) and value.ndim == 0:
                print(f"{metric}: {value.item():.3f}")
            elif isinstance(value, np.ndarray) and value.ndim == 1 and len(value) < 10:
                formatted = ", ".join(f"{v:.3f}" for v in value)
                print(f"{metric}: [{formatted}]")
    
    @property
    def results(self) -> Dict[str, Any]:
        if self._results is None:
            self.evaluate()
        return self._results
    
    def plot_precision_recall(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {self.iou_type}')
        plt.grid(True)
        plt.plot([0, 1], [0, 0], 'r--')
        plt.show()