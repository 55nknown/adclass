import pytest
from flask.testing import FlaskClient

@pytest.mark.order(2)
def test_get(test_client: FlaskClient, sample_id: str, sample_images: list[tuple[str, bytes]]):
    # Get image index
    res = test_client.get('/api/images/index')
    
    # Check if succeeded
    assert res.status_code == 200

    # Check if every sample is uploaded
    assert "images" in res.json
    assert type(res.json["images"]) == list

    count = 0
    
    for img in res.json["images"]:
        if img.startswith(sample_id):
            count += 1

    assert count == sample_images.__len__()