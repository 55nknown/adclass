import cv2
import numpy as np
import matplotlib.pyplot as plt

MIN_MATCH_COUNT = 10
MIN_SCORE_THRESHOLD = 500
FLANN_INDEX_LSH = 6
FLANN_INDEX_KDTREE = 0
MIN_RES_W = 300
SHARPEN_FILTER = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

def get_kp(mat):
    orb = cv2.ORB_create()
    _, desc = orb.detectAndCompute(mat, None)
    return desc

def __create_flann_matcher():
    index_params = dict(algorithm = FLANN_INDEX_LSH,
                table_number = 6,
                key_size = 12,
                multi_probe_level = 1)
    search_params = dict(checks = 100)

    return cv2.FlannBasedMatcher(index_params, search_params)

# Load query
query_img = cv2.imread("benchmark/query.jpg")
query_img = cv2.cvtColor(query_img, cv2.COLOR_RGB2GRAY)
query_img = cv2.resize(query_img, ((query_img.shape[1]/3).__floor__(), (query_img.shape[0]/3).__floor__()), interpolation=cv2.INTER_NEAREST)

print(f"Initial query dim: {query_img.shape[1]} x {query_img.shape[0]}")
query_kp = get_kp(query_img)

# Load train
train_img = cv2.imread("benchmark/train.jpg")
train_img = cv2.cvtColor(train_img, cv2.COLOR_RGB2GRAY)
train_img = cv2.resize(train_img, ((train_img.shape[1]/2).__floor__(), (train_img.shape[0]/2).__floor__()), interpolation=cv2.INTER_NEAREST)
train_img = cv2.filter2D(train_img, -1, SHARPEN_FILTER)
train_img = cv2.filter2D(train_img, -1, SHARPEN_FILTER)

print(f"Initial train dim: {train_img.shape[1]} x {train_img.shape[0]}")
train_kp = get_kp(train_img)

matcher = __create_flann_matcher()

data = []
cimg = query_img

while cimg.shape[1] >= MIN_RES_W:
    try:
        matches = matcher.knnMatch(get_kp(cimg), train_kp, k=2)
    except Exception as e:
        print(f"ERROR matcher: {e}")
        continue

    good = []
    score = 0
    for m in matches:
        if m.__len__() < 2:
            continue
        if m[0].distance < 0.8 * m[1].distance:
            score += 0.8 * m[1].distance - m[0].distance
            good.append(m[0])

    data.append(((cimg.shape[1], cimg.shape[0]), score))

    print(f"cimg dim: {cimg.shape[1]} x {cimg.shape[0]}, score: {score.__round__()}")

    cimg = cv2.resize(cimg, ((cimg.shape[1]/1.02).__floor__(),(cimg.shape[0]/1.02).__floor__()), interpolation=cv2.INTER_NEAREST)

data.sort(key=lambda x: x[0][0], reverse=True)

plt.plot([f"{x[0][0]} x {x[0][1]}" for x in data], [x[1] for x in data])
plt.xlabel("res")
plt.ylabel("score")
plt.xticks(rotation=90)
plt.show()