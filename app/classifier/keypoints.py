import cv2
import numpy as np

def get_keypoints(buf: bytes, resize=False) -> np.ndarray:
    orb = cv2.ORB_create()
    image = cv2.imdecode(np.asarray(bytearray(buf), dtype=np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if resize:
        image = cv2.resize(image, (1000, int(1000 / (image.shape[1] / image.shape[0]))))

    _, desc = orb.detectAndCompute(image, None)

    return desc
