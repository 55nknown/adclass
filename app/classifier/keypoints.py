import cv2
import numpy as np

def get_keypoints(buf: bytes) -> np.ndarray:
    orb = cv2.ORB_create()
    image = cv2.imdecode(np.asarray(bytearray(buf), dtype=np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, desc = orb.detectAndCompute(image, None)

    return desc