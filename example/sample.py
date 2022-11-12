import cv2
import numpy as np

class SampleImage:
    keypoints: list[cv2.KeyPoint] = []
    descriptors = []
    image: cv2.Mat
    path: str

    def __init__(self, path: str):
        self.path = path

    def load(self):
        self.image = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)

    def sharpen(self):
        # filter = np.array([[ 0, -1,  0], [-1, 5, -1], [ 0, -1,  0]])
        filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        self.image = cv2.filter2D(self.image, -1, filter)

    def compute(self, orb):
        kp, desc = orb.detectAndCompute(self.image, None)
        self.keypoints = kp
        self.descriptors = desc
