import numpy as np

class KalmanFilter:
    def __init__(self, bbox):
        self.state = np.array(bbox)

    def predict(self):
        return self.state

    def update(self, bbox):
        self.state = np.array(bbox)