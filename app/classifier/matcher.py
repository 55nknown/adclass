import cv2
import numpy as np

from typing import Optional

MIN_MATCH_COUNT = 10
MIN_SCORE_THRESHOLD = 300
FLANN_INDEX_LSH = 6
FLANN_INDEX_KDTREE = 0

def __create_flann_matcher():
        index_params = dict(algorithm = FLANN_INDEX_LSH,
                    table_number = 6,
                    key_size = 12,
                    multi_probe_level = 1)
        search_params = dict(checks = 100)

        return cv2.FlannBasedMatcher(index_params, search_params)


def calculate_score(query: np.ndarray, train: np.ndarray) -> Optional[float]:
    try:
        matcher = __create_flann_matcher()
        matches = matcher.knnMatch(query, train, k=2)

        good = 0
        score = 0
        for m in matches:
            if m.__len__() < 2:
                continue
            if m[0].distance < 0.8 * m[1].distance:
                good += 1
                score += 0.8 * m[1].distance - m[0].distance

        if good > MIN_MATCH_COUNT and score > MIN_SCORE_THRESHOLD:
            return score
    except Exception as e:
        print(f"Error: {e}")

    return None
