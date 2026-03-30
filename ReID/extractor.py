import numpy as np
from torchreid.reid.utils import FeatureExtractor
import torch

class ReIDExtractor:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path='osnet_x1_0_msmt17.pt',
            device=self.device
        )

    def extract(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None
        
        feat = self.model(crop)[0]        
        feat = feat.detach().cpu().numpy()
        feat = feat / (np.linalg.norm(feat) + 1e-6)

        return feat