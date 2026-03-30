import motmetrics as mm
import numpy as np

class MOTMetrics:
    def __init__(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def update(self, gt_boxes, tracks):
        """
        gt_boxes: list of [x1,y1,x2,y2]
        tracks: list of (bbox, id)
        """

        gt_ids = list(range(len(gt_boxes)))
        pred_ids = [tid for _, tid in tracks]

        if len(gt_boxes) == 0 or len(tracks) == 0:
            self.acc.update(gt_ids, pred_ids, [])
            return

        cost_matrix = np.zeros((len(gt_boxes), len(tracks)))

        for i, gt in enumerate(gt_boxes):
            for j, (pred_box, _) in enumerate(tracks):

                iou = self._iou(gt, pred_box)

                cost_matrix[i, j] = 1 - iou

        self.acc.update(gt_ids, pred_ids, cost_matrix)

    def _iou(self, a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (a[2]-a[0])*(a[3]-a[1])
        area2 = (b[2]-b[0])*(b[3]-b[1])

        union = area1 + area2 - inter + 1e-6

        return inter / union

    def compute(self):
        mh = mm.metrics.create()

        summary = mh.compute(
            self.acc,
            metrics=['mota', 'idf1', 'precision', 'recall'],
            name='tracking'
        )

        print("\n===== METRICS =====")
        print(summary)