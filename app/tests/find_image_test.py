import pytest
from flask.testing import FlaskClient
from classifier.keypoints import get_keypoints

@pytest.mark.order(3)
def test_find(test_client: FlaskClient, sample_id: str, sample_images: list[tuple[str, bytes]]):
    for i in range(sample_images.__len__()):
        # Attach sample image as input
        inputs = [
            {
                "keypoints": get_keypoints(sample_images[i][1]).tolist()
            }
        ]

        # Find image index
        res = test_client.post('/api/images/find', json={"inputs": inputs})

        # Check if succeeded
        assert res.status_code == 200

        print(res.json)

        # Check if match is correct
        assert "results" in res.json
        assert type(res.json["results"]) == list
        assert res.json["results"].__len__() > 0
        assert "id" in res.json["results"][0]
        assert res.json["results"][0]["id"] == f"{sample_id}_{i}"
        assert res.json["results"][0]["score"] > 1000


@pytest.mark.order(4)
def test_find_wrong(test_client: FlaskClient, sample_id: str, sample_images: list[tuple[str, bytes]]):
    # Attach sample image as input
    inputs = [
        {
            "keypoints": get_keypoints(x[1]).tolist()
        }
        for x in sample_images[-3:]
    ]

    # Find image index
    res = test_client.post('/api/images/find', json={"inputs": inputs})

    # Check if succeeded
    assert res.status_code == 200

    # Check if match is correct
    assert "results" in res.json
    assert type(res.json["results"]) == list
    assert res.json["results"].__len__() > 0
    assert "id" in res.json["results"][0]

    best_score = res.json["results"][0]["score"]

    img = sample_images[int(res.json["results"][0]["id"].split("_")[1])]

    # Attach sample image as input
    inputs = [
        {
            "keypoints": get_keypoints(img[1]).tolist()
        },
        {
            "keypoints": get_keypoints(img[1]).tolist()
        },
        {
            "keypoints": get_keypoints(img[1]).tolist()
        }
    ]

    # Find image index
    res = test_client.post('/api/images/find', json={"inputs": inputs})

    assert res.json["results"][0]["score"] > best_score