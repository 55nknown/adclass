import os

from flask import Flask, request, Response
from pymongo import MongoClient
import numpy as np

from classifier import keypoints
from classifier import matcher

router = Flask(__name__)

MONGO_CONNECTION = os.environ["MONGO"]
MONGO_DATABASE = os.environ["MONGO_DB"]


def get_database():
    client = MongoClient(MONGO_CONNECTION)
    database = client[MONGO_DATABASE]
    collection = database['index']
    return collection

@router.route('/api/images/find', methods=["POST"])
def find_image_index():
    inputs = []

    # Json input with client-side processing
    if request.content_type == "application/json":
        inputs = request.json["inputs"]

    # Raw input with server-side processing
    elif request.content_type.split(';')[0] == "multipart/form-data":
        for key in request.files.keys():
            buf = request.files[key].stream.read()
            kp = keypoints.get_keypoints(buf)
            inputs.append({"keypoints": kp.tolist()})

    db = get_database()
    indexes = list(db.find())

    all_results = []

    # Match all inputs
    for input_kp in inputs:
        input_results = []
        train = np.asarray(input_kp["keypoints"], dtype=np.uint8)

        # Match each image for the input
        for index_kp in indexes:
            query = np.asarray(index_kp["keypoints"], dtype=np.uint8)

            # Calculate match score based on keypoint distance
            score = matcher.calculate_score(query, train)

            if not score is None:
                input_results.append({"id": index_kp["id"], "score": score})

        if input_results.__len__() == 0:
            continue

        # Sort by score (descending)
        input_results.sort(key=lambda x: x["score"], reverse=True)

        all_results.append(input_results)

    images = {}

    # Group results by images
    for result in all_results:
        best_index = result[0]["id"]
        best_score = result[0]["score"]

        if not best_index in images:
            images[best_index] = {"count": 0, "scores": []}

        images[best_index]["count"] += 1
        images[best_index]["scores"].append(best_score)

    results = []

    for index_id in images.keys():
        scores = images[index_id]["scores"]
        count = images[index_id]["count"]

        # Average scores for index
        final_score = sum(scores) / scores.__len__()

        # Final result (multiply score by matches)
        results.append({"id": index_id, "score": final_score * count, "matches": count})

    # Sort by wins (descending)
    results.sort(key=lambda x: x["score"], reverse=True)    

    return {"results": results}

@router.route('/api/images/index', methods=["GET"])
def get_image_index():
    db = get_database()
    indexes = db.find()

    # Return index ids
    results = [i["id"] for i in indexes]

    return {"images": results}

@router.route('/api/images/index', methods=["POST"])
def index_image():
    db = get_database()
    index_id = request.form.get("id")

    # Check if id already exists
    count = db.count_documents({"id": index_id})
    if count > 0:
        return Response("An index with this id already exists", status=409)

    # Generate keypoints
    buf = request.files['data'].stream.read()
    kp = keypoints.get_keypoints(buf, resize=True)

    # Store keypoints
    doc = {"keypoints": kp.tolist(), "id": index_id}
    db.insert_one(doc)

    return Response(status=200)

@router.route('/api/images/deindex', methods=["POST"])
def remove_index():
    db = get_database()
    index_id = request.json["id"]

    # Delete keypoints
    res = db.delete_one({"id": index_id})

    if res.deleted_count > 0:
        status_code = 200
    else:
        status_code = 404

    return Response(status=status_code)
