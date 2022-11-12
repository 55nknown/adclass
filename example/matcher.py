import os

import cv2
import numpy as np

from sample import SampleImage

MIN_MATCH_COUNT = 10
MIN_SCORE_THRESHOLD = 500
FLANN_INDEX_LSH = 6
FLANN_INDEX_KDTREE = 0

class FBTMatcher:
    def __create_flann_matcher():
        index_params = dict(algorithm = FLANN_INDEX_LSH,
                    table_number = 6,
                    key_size = 12,
                    multi_probe_level = 1)
        search_params = dict(checks = 100)

        return cv2.FlannBasedMatcher(index_params, search_params)

    orb = cv2.ORB_create() # Fast & Free keypoint detector
    # matcher = cv2.BFMatcher(cv2.NORM_HAMMING) # Brute-force desc matcher
    matcher = __create_flann_matcher() # Fast Library for Approximate Nearest Neighbors

    samples: list[SampleImage] = [] # Query images
    train_img: SampleImage # Train image

    smatch = None
    squery = None

    # Load sample query images
    def load_samples(self):
        basepath = "./samples/ads/"
        sample_files = os.listdir(basepath)

        for sample in sample_files:
            # Only allow .png files
            if not sample.endswith(".png"):
                continue
            sample = SampleImage(f"{basepath}{sample}")
            sample.load()
            # sample.sharpen()
            sample.compute(self.orb)
            self.samples.append(sample)

        print(f"Loaded {self.samples.__len__()} samples")

    # Load input from path
    def load_input(self, path):
        self.train_img = SampleImage(path)
        self.train_img.load()
        self.train_img.sharpen()
        self.train_img.compute(self.orb)
        print(f"Loaded the input sample ({self.train_img.keypoints.__len__()} features)")

    # Use camera as input
    def load_camera_frame(self):
        video = cv2.VideoCapture(0)
        _, frame = video.read()
        self.train_img = SampleImage("camera://0")
        self.train_img.image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        self.train_img.sharpen()
        self.train_img.compute(self.orb)

    # Visualize keypoint matches
    def visualize(self, *, wait = False):
        if self.smatch == None or self.squery == None:
            return

        try:
            src_pts = np.float32([ self.squery.keypoints[m.queryIdx].pt for m in self.smatch ]).reshape(-1,1,2)
            dst_pts = np.float32([ self.train_img.keypoints[m.trainIdx].pt for m in self.smatch ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            h, w = self.squery.image.shape
            pts = np.float32([ [0,0], [0,h-1], [w-1,h-1], [w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts, M)

            img2 = cv2.polylines(self.train_img.image, [np.int32(dst)], True, color=(255, 0, 255), thickness=5, lineType=cv2.LINE_AA)

            draw_params = dict(matchColor=(0,255,0), # draw matches in green color
                            singlePointColor=None,
                            matchesMask=matchesMask, # draw only inliers
                            flags=2)

            img3 = cv2.drawMatches(self.squery.image, self.squery.keypoints, img2, self.train_img.keypoints, self.smatch, None, **draw_params)

            cv2.imshow("train", img3)
        except:
            pass

        # cv2.imshow("train", self.train_img.image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit(0)

        if wait:
            cv2.waitKey(0)

    # Match keypoints between images
    def match(self):
        results = []

        for query in self.samples:
            try:
                matches = self.matcher.knnMatch(query.descriptors, self.train_img.descriptors, k=2)
            except Exception as e:
                print(f"ERROR FBTMatcher.find: {e}")
                continue

            good = []
            score = 0
            for m in matches:
                if m.__len__() < 2:
                    continue
                if m[0].distance < 0.8 * m[1].distance:
                    score += 0.8 * m[1].distance - m[0].distance
                    good.append(m[0])

            if good.__len__() > MIN_MATCH_COUNT:
                results.append((score, good, query))

        # Find best result
        if results.__len__() > 0:
            scores = [r[0] for r in results]
            highscore = max(scores)

            # Filter false positives
            if highscore < MIN_SCORE_THRESHOLD:
                return -1

            bestindex = scores.index(highscore)
            best = results[bestindex]
            print(f"Best score: {best[0]}")

            # Only for debugging
            self.smatch = best[1]
            self.squery = best[2]

            return bestindex

        return -1